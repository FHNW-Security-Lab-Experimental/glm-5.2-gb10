#!/usr/bin/env python3
"""GLM-5.2 long-context validation harness (coherence + needle-in-haystack).

DSA sparse attention can corrupt SILENTLY — /health stays 200 while output degrades —
so a passing /health is NOT evidence the model is correct at long context. This script
exercises the actual generation path. It runs against ANY OpenAI-compatible endpoint, so
use it as a baseline against the live 512k endpoint today, and as the acceptance gate for
the experimental DCP endpoint later (see 1M-SPARSE-CP-KV-PLAN.md, validation tests 4-6).

Examples:
  # coherence + 32k needle against the live 512k head
  ./validate_dcp.py --base http://192.168.88.101:8000/v1 --model glm-5.2-nvfp4 --needle-tokens 32000

  # deep needle sweep at 512k (depths 10/50/90%)
  ./validate_dcp.py --base http://192.168.88.101:8000/v1 --model glm-5.2-nvfp4 \
      --needle-tokens 500000 --depths 0.1,0.5,0.9

  # temp-0 A/B equivalence between two endpoints (replicated vs DCP)
  ./validate_dcp.py --base http://host-a:8000/v1 --base-b http://host-b:8000/v1 \
      --model glm-5.2-nvfp4 --ab-tokens 16000
"""
import argparse
import json
import sys
import urllib.request

# ~4 chars/token is a safe over-estimate for English filler; we pad and trust the
# server's max_model_len, not an exact tokenizer count.
CHARS_PER_TOKEN = 4
FILLER = (
    "The quarterly maintenance log records routine checks of the cooling loop, the "
    "power distribution units, and the network fabric across all eight nodes. "
)


def _post(base, model, messages, api_key, max_tokens, temperature):
    url = base.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.loads(r.read())
    msg = data["choices"][0]["message"]
    # GLM is a reasoning model: `content` is None while the answer is still in
    # `reasoning_content` (or when max_tokens is hit mid-think). Prefer content,
    # fall back to reasoning so the checks see the model's text either way.
    return (msg.get("content") or msg.get("reasoning_content") or "")


def _filler_to_tokens(n_tokens):
    target_chars = n_tokens * CHARS_PER_TOKEN
    reps = max(1, target_chars // len(FILLER))
    return FILLER * reps


def coherence(base, model, api_key):
    print("== coherence ==")
    out = _post(
        base, model,
        [{"role": "user", "content": "In one sentence, what is 17 * 23? Then say DONE."}],
        api_key, max_tokens=2048, temperature=0.0,
    )
    ok = "391" in out and "DONE" in out.upper()
    print(("  PASS" if ok else "  FAIL") + ": " + out.strip().replace("\n", " ")[:200])
    return ok


def needle(base, model, api_key, n_tokens, depths):
    print(f"== needle-in-haystack ({n_tokens} tokens) ==")
    secret = "The vault access code for spark-a175 is PLUM-4731-OWL."
    question = (
        "What is the vault access code for spark-a175? Reply with only the code."
    )
    all_ok = True
    for depth in depths:
        filler = _filler_to_tokens(n_tokens)
        cut = int(len(filler) * depth)
        haystack = filler[:cut] + "\n\n" + secret + "\n\n" + filler[cut:]
        out = _post(
            base, model,
            [
                {"role": "user", "content": haystack + "\n\n" + question},
            ],
            api_key, max_tokens=1024, temperature=0.0,
        )
        ok = "PLUM-4731-OWL" in out.upper()
        all_ok = all_ok and ok
        print(f"  depth {depth:>4}: " + ("PASS" if ok else "FAIL") + " -> " + out.strip()[:80])
    return all_ok


def ab_equiv(base_a, base_b, model, api_key, n_tokens):
    print(f"== temp-0 A/B equivalence ({n_tokens} tokens) ==")
    prompt = _filler_to_tokens(n_tokens) + "\n\nSummarize the above in exactly 3 sentences."
    msgs = [{"role": "user", "content": prompt}]
    a = _post(base_a, model, msgs, api_key, max_tokens=512, temperature=0.0)
    b = _post(base_b, model, msgs, api_key, max_tokens=512, temperature=0.0)
    ok = a.strip() == b.strip()
    print("  PASS: identical" if ok else "  FAIL: outputs differ (CP path diverges from replicated)")
    if not ok:
        print("  --- A ---\n  " + a.strip()[:300] + "\n  --- B ---\n  " + b.strip()[:300])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="OpenAI-compatible /v1 base URL")
    ap.add_argument("--base-b", default=None, help="second endpoint for A/B equivalence")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--needle-tokens", type=int, default=0, help="0 disables the needle test")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--ab-tokens", type=int, default=0, help="0 disables the A/B test")
    ap.add_argument("--skip-coherence", action="store_true")
    args = ap.parse_args()

    results = {}
    if not args.skip_coherence:
        results["coherence"] = coherence(args.base, args.model, args.api_key)
    if args.needle_tokens > 0:
        depths = [float(x) for x in args.depths.split(",")]
        results["needle"] = needle(args.base, args.model, args.api_key, args.needle_tokens, depths)
    if args.ab_tokens > 0:
        if not args.base_b:
            print("ERROR: --ab-tokens requires --base-b", file=sys.stderr)
            sys.exit(2)
        results["ab"] = ab_equiv(args.base, args.base_b, args.model, args.api_key, args.ab_tokens)

    print("\n== summary ==")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
