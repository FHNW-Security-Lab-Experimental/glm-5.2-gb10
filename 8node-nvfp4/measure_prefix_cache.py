#!/usr/bin/env python3
"""Prefix-caching probe: send an IDENTICAL long prompt twice, measure cold vs warm
TTFT (= prefill time), the vllm:prefix_cache_hits_total delta, and needle correctness.

If APC works on this backend: run2 TTFT << run1 TTFT and hits_delta jumps to ~prompt_tokens.
If APC is a no-op (DSA/fp8_ds_mla incompat): run2 TTFT ~= run1 TTFT and hits_delta ~= 0.

Usage: measure_prefix_cache.py [approx_tokens] [base_url]
"""
import sys, time, json, urllib.request

APPROX = int(sys.argv[1]) if len(sys.argv) > 1 else 131000
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://192.168.88.101:8000"
MODEL = "glm-5.2-nvfp4"
NEEDLE_CODE = "PLUM-4731-OWL"


def metrics():
    m = urllib.request.urlopen(BASE + "/metrics", timeout=15).read().decode()
    q = h = 0.0
    for ln in m.splitlines():
        if ln.startswith("vllm:prefix_cache_queries_total"):
            q = float(ln.split()[-1])
        elif ln.startswith("vllm:prefix_cache_hits_total"):
            h = float(ln.split()[-1])
    return q, h


def build_prompt(approx_tokens):
    # filler sentence below tokenizes to ~12 tokens; calibrate reps to hit approx_tokens.
    sent = "The quiet harbor town kept careful ledgers each passing season. "
    reps = max(1, approx_tokens // 12)
    half = reps // 2
    body = (sent * half) + f"\n\nIMPORTANT FACT: the secret passcode is {NEEDLE_CODE}.\n\n" + (sent * (reps - half))
    return body + "\n\nQuestion: What is the secret passcode? Answer with only the code."


def run(prompt, label):
    q0, h0 = metrics()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    content, reasoning = [], []
    ptoks = None
    with urllib.request.urlopen(req, timeout=1200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if obj.get("usage"):
                ptoks = obj["usage"].get("prompt_tokens", ptoks)
            for ch in obj.get("choices", []):
                d = ch.get("delta", {}) or {}
                pc = d.get("content")
                pr = d.get("reasoning") or d.get("reasoning_content")
                if (pc or pr) and ttft is None:
                    ttft = time.time() - t0
                if pc:
                    content.append(pc)
                if pr:
                    reasoning.append(pr)
    wall = time.time() - t0
    q1, h1 = metrics()
    full = "".join(content) + " " + "".join(reasoning)
    ans = ("".join(content).strip() or "".join(reasoning).strip())
    needle_ok = NEEDLE_CODE in full
    print(f"[{label}] prompt_tokens={ptoks} TTFT={ttft:.1f}s wall={wall:.1f}s "
          f"hits_delta={h1-h0:.0f} queries_delta={q1-q0:.0f} needle={'PASS' if needle_ok else 'FAIL'} "
          f"answer={ans[:60]!r}")
    return {"ttft": ttft, "ptoks": ptoks, "hits_delta": h1 - h0, "needle": needle_ok}


def main():
    prompt = build_prompt(APPROX)
    print(f"# prefix-cache probe: approx_tokens~={APPROX}, base={BASE}")
    r1 = run(prompt, "run1-cold")
    time.sleep(2)
    r2 = run(prompt, "run2-warm")
    if r1["ttft"] and r2["ttft"]:
        sp = r1["ttft"] / r2["ttft"] if r2["ttft"] else float("inf")
        print(f"# VERDICT: cold_TTFT={r1['ttft']:.1f}s warm_TTFT={r2['ttft']:.1f}s "
              f"speedup={sp:.2f}x  warm_hits_delta={r2['hits_delta']:.0f}  "
              f"-> APC {'WORKS' if (r2['hits_delta'] > 1000 and sp > 1.5) else 'NO-OP/ineffective'}")


if __name__ == "__main__":
    main()
