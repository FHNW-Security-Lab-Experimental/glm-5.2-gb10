# Many-MCP-tools HTTP 500 / "reply starts then stops" — xgrammar (2026-07-02)

A tool-calling failure distinct from the three GB10 crash/wedge classes. **The engine-side fix is still pending
(see bottom); this doc records the root cause + the reverted proxy experiment so nothing is lost.**

## Symptom
OpenCode sessions with many MCP servers (many tools) → HTTP **500** with the **reply starting then stopping**;
the "replay"/retry re-hits it. Only tool-heavy requests; plain chat is fine.

## Root cause: xgrammar guided-decoding FSM rejection (deterministic)
Tool / structured requests engage vLLM's **guided decoding via xgrammar** (the default structured-output
backend). Mid-generation GLM-5.2 emits **token 154842**, which xgrammar's schema-derived FSM cannot accept →
`Failed to advance FSM` (in `vllm/v1/structured_output/backend_xgrammar.py`) → vLLM **terminates the request
mid-stream** → the client sees a broken 200 SSE stream → surfaced as "500". Deterministic (same token every time:
11:18 ×3, 12:53). Explains everything: HTTP stays 200 (headers already sent), **zero nginx 5xx/499**, zero
EngineCore deaths, not memory, not the proxy. **Note:** `--enable-auto-tool-choice` (`tool_choice: auto`, which
is how OpenCode calls the API) **still engages xgrammar internally** — the proxy cannot intercept that path.

## What was TRIED and REVERTED: the proxy structured-output strip
`tools/vllm_keepalive_proxy.py` gained an opt-in `VLLM_PROXY_STRIP_STRUCTURED_OUTPUT` (default **0/off**, commit
`1589846`): it removes `response_format` and downgrades a **strict** `tool_choice` (`required`/named) → `auto` so
xgrammar never engages; the `glm47` parser still extracts tool calls. Deploy drop-in kept for reference:
`deploy/proxy-auth/sparks-vllm-proxy.service.d/strip-structured.conf`.

**It did NOT work and is reverted** (env back to `0` on the prod VM). Two reasons:
1. **OpenCode sends `tool_choice: auto` with no `response_format`** — exactly the shape the strip leaves untouched
   (verified in the proxy log: strip fired only on manual test requests, **zero** real OpenCode requests). And
   `auto` still triggers xgrammar internally, on a path the proxy can't touch without breaking tool-calling.
2. On the requests it *did* affect (`required`/`response_format`), it **broke** schema-enforced structured output.

So it's a bad trade in both directions. Left in the code (opt-in, off) only so the mechanism is documented.

## The real fix (pending — engine-side, keeps everything): graceful-fallback
Patch xgrammar so a `Failed to advance FSM` **drops the grammar constraint for that one request and keeps
generating** instead of terminating (same philosophy as the cuBLAS retry wrapper — degrade, don't die). It:
- covers **`auto`, `required`, and `response_format`** alike (matches how OpenCode actually calls the API),
- **keeps** structured decoding wherever it works (no regression — unlike the strip),
- is **head-only** (an in-container patch in `glm52-sparse-patches.sh`), no prod-VM access,
- is safe regardless: fires only on the exact failure that is currently fatal; a no-op otherwise.
Cost: **one model relaunch**. This is the correct fix — apply at the next relaunch window, then the strip can be
deleted entirely.

## Caveat (why one aggregate tweak never lands)
The observed "500s/stops" are a **mix**: the xgrammar termination (above) **+** long-prefill TTFT on ~100k
contexts (9-10s → up to 237s to first token; prefix cache already 83% — sparse attention still scans full
context) **+** normal OpenCode turn-cancellations (`saw_done=False`). Confirm the mode live (reproduce + read the
proxy `request→first-chunk→upstream-closed` line + engine log at that timestamp) before attributing a given stop.
