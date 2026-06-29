#!/usr/bin/env python3
"""Aggregate batch-curve probe: fire N concurrent identical decode streams, measure each
stream's REAL decode rate (completion_tokens-1)/(t_last-t_first), sum = aggregate tok/s.

Tests whether the engine AMORTIZES the dominant MoE-weight read across concurrent rows
(the 'use the 4 slots' win). Small context + long gen isolates DECODE and forces overlap.
If aggregate grows ~linearly with N -> batching amortizes; if flat -> no aggregate win.

Usage: concurrent_decode.py [ctx_tokens] [max_tokens]   (loops N=1,2,4)
"""
import sys, json, time, threading, urllib.request

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 600
BASE = "http://192.168.88.101:8000"
MODEL = "glm-5.2-nvfp4"

sent = "The quiet harbor town kept careful ledgers each passing season. "
PROMPT = (sent * max(1, CTX // 12)) + "\n\nWrite a long detailed essay about distributed systems."


def one(idx, results):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAXTOK, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_last = None; ctoks = None
    with urllib.request.urlopen(req, timeout=1200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                obj = json.loads(p)
            except Exception:
                continue
            if obj.get("usage"):
                ctoks = obj["usage"].get("completion_tokens", ctoks)
            for ch in obj.get("choices", []):
                d = ch.get("delta", {}) or {}
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
    rate = (ctoks - 1) / (t_last - t_first) if (ctoks and t_first and t_last and t_last > t_first) else 0.0
    results[idx] = {"rate": rate, "ctoks": ctoks, "ttft": (t_first - t0) if t_first else None}


def run(n):
    results = {}
    threads = [threading.Thread(target=one, args=(i, results)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    rates = [results[i]["rate"] for i in range(n)]
    agg = sum(rates)
    per = ", ".join(f"{r:.1f}" for r in rates)
    print(f"N={n}: aggregate={agg:.1f} tok/s | per-stream=[{per}] | "
          f"mean_per_stream={agg/n:.1f} | wall={wall:.0f}s")
    return agg


def main():
    print(f"# concurrent decode batch-curve: ctx~={CTX}, max_tokens={MAXTOK}, base={BASE}")
    base1 = None
    for n in (1, 2, 4):
        a = run(n)
        if n == 1:
            base1 = a
    if base1:
        print(f"# (compare aggregate(N) vs N=1={base1:.1f}; ~linear growth => batching amortizes the MoE read)")


if __name__ == "__main__":
    main()
