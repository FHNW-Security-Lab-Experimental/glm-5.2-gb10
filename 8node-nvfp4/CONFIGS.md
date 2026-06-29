# GLM-5.2-NVFP4 on 8× GB10 — serving configs (non-DCP 512k · DCP 512k · DCP 1M)

Three validated configs, one launcher (`start_glm52_config.sh <profile>`). All share: full
`nvidia/GLM-5.2-NVFP4`, TP=8, native sm_121 DSA sparse, `fp8_ds_mla` KV, in-checkpoint MTP
(k=3), MARLIN NVFP4 MoE, `--enforce-eager`, util 0.78.

| Profile | Context | KV layout | Decode tok/s¹ | Parallel slots² | Use when |
|---|---|---|---|---|---|
| **`prod`** (non-DCP) | 512k | replicated per rank | **~22.5** | 4 cap / **~1.1 full-512k** | Default. Fastest. Many *short* concurrent calls ≤512k. |
| **`dcp-512k`** | 512k | sharded (DCP=8) | ~20–21 | **8** (pool ~9 × 512k) | 512k with real long-context concurrency (8× pool). |
| **`dcp-1m`** | **1,048,576** | sharded (DCP=8) | ~17–18 | **4** (pool **4.12× full-1M**, vLLM-reported) | True 1M context, up to 4 parallel 1M streams. |

² Parallel slots = `--max-num-seqs` default (set per profile in `start_glm52_config.sh`; override with
`MAX_NUM_SEQS=N ...`). The *real* ceiling is the KV pool: **prod's replicated pool (601,856 tok) holds
only ~1.1 full-512k contexts** — its 4 slots are usable only for shorter/mixed requests whose combined
tokens stay under ~600k. **The DCP pools are ~8× larger** (sharding stores 1/8 of the KV per rank): dcp-512k
≈ 9 full-512k streams, dcp-1m ≈ 4.12 full-1M streams (vLLM logs "Maximum concurrency for 1,048,576 tokens
per request: 4.12x"). So DCP buys both longer context *and* far more long-request concurrency. Tuning note:
with many slots + MTP, `--max-num-batched-tokens` (4096) can bottleneck concurrent *prefill* — raise it
(`MAX_NUM_BATCHED_TOKENS=...` if you wire it) if prefill throughput matters more than memory.

¹ Single-stream, warm, measured 2026-06-28. The non-DCP path is materially faster — **prefer
`prod` for everything that fits in 512k**; use DCP only to exceed 512k (or for the larger KV pool).

## Start a config (from the head)

```bash
~/vllm-glm52/runtime/start_glm52_config.sh prod        # non-DCP 512k (fastest, production)
~/vllm-glm52/runtime/start_glm52_config.sh dcp-512k    # DCP 512k
MAX_NUM_SEQS=4 ~/vllm-glm52/runtime/start_glm52_config.sh dcp-1m   # DCP, true 1M, 4 parallel slots
```

`dcp-1m` already defaults to `MAX_NUM_SEQS=4` (shown explicit above); the KV pool holds ~4.12× a
full-1M context, so 4 concurrent 1M streams fit. This is the **current production target** (true 1M
+ 4 slots). Each command stops the prior model containers, drops caches on all 8, applies the
in-container patches, and serves OpenAI-compatible on `192.168.88.101:8000` as `glm-5.2-nvfp4`.
First boot ~12–15 min (465 GB load). Validate coherence + a long-context needle before trusting it
(sparse attention can corrupt silently — never gate on `/health` alone): `validate_dcp.py`.

## Speed knobs on the non-DCP `prod` path only (cudagraph + prefix caching)

**Why `prod` is faster (~24 vs ~18.8 tok/s decode):** non-DCP has no per-step cross-rank work — no
local-top-2048 all-gather / union / owned-mask / LSE-recombine, and KV is local (replicated, not
sharded-then-gathered). The cost is the 512k cap (1M of replicated MLA KV overflows a 121 GiB node →
1M *requires* DCP). Measured 2026-06-29, prod baseline (eager): **23.9 / 23.7 tok/s @4k (warm), 23.4
@90k**; dcp-1m baseline ~18.8. Two further toggles exist **only** on this non-DCP path (both are
env-overridable defaults in `start_glm52_config.sh`, so production behavior is unchanged unless set):

| Knob | How | Effect | DCP? |
|---|---|---|---|
| **CUDA graphs** | `ENFORCE_EAGER=0 start_glm52_config.sh prod` | launcher drops `--enforce-eager`, captures graphs; the in-container b12x probe selects **FULL** (confirmed `GLM52_CUDAGRAPH_MODE=FULL` on the prod boot). Est. +2–6% decode (bandwidth-bound caps it; MTP already amortizes launch overhead). | **NO** — cudagraph is a confirmed NO-GO on DCP (DCP collectives + `NCCL_CUMEM_ENABLE=0`). |
| **Prefix caching** | `PROD_EXTRA_ARGS="--enable-prefix-caching" start_glm52_config.sh prod` | adds the flag (default passes none → vLLM-v1 default). Targets agentic context reuse (re-prefill drops to delta-only). | **NO/ineffective** — on dcp-1m the metric showed **67,540 queries / 0 hits** (sharded KV ⇒ no block reuse). |

**Validation status (honest — these were wired + baselined this session but the on-toggle runs were
not completed before pivoting back to dcp-1m):**
- **cudagraph capture + actual speedup: NOT yet measured.** The toggle works and FULL is available
  per the b12x probe, but the captured run (Boot C) was not run. Capture allocates extra workspace on
  top of util 0.78 → watch for OOM (the cluster has an all-8-node OOM-wedge history; power-cycle to
  recover); if capture-OOMs, retry with `GPU_MEMORY_UTILIZATION=0.74`. Re-gate correctness after
  capture: top-k **set-equality** + 16k/131k needle at temp=0, never `/health`.
- **prefix-cache effectiveness on the non-DCP DSA/`fp8_ds_mla` backend: UNVALIDATED.** Boot A
  confirmed `--no-enable-prefix-caching` is a clean no-op (cold≈warm TTFT, hits_delta=0 ✓); the
  `--enable-prefix-caching` run (Boot B) was aborted before the test ran. The critic's open risk is
  that vLLM auto-disables APC on the custom sparse-indexer backend (flag accepted, never hits). Probe
  staged on the head: `python3 ~/measure_prefix_cache.py <approx_tokens>` — sends an identical prompt
  twice and reports cold-vs-warm TTFT + the `vllm:prefix_cache_hits_total` delta (PASS = hits jump
  ~prompt_tokens AND warm TTFT ≪ cold).

These knobs cannot speed up `dcp-1m` (both are non-DCP-only). **To make `dcp-1m` faster (keeping 1M +
4 slots) see [`DCP1M-FASTER-RESEARCH.md`](DCP1M-FASTER-RESEARCH.md)** (full 2026-06-29 study). Headlines:
the **4 slots ARE the win** — measured aggregate **N=4 = 40.3 tok/s (2.13×** the 18.9 single-stream;
`concurrent_decode.py`), already live for <256k traffic; extend to concurrent ≥256k by raising the proxy
`VLLM_PROXY_LONG_CONTEXT_TOKENS`. Real per-token decode lever = **`num_experts_per_tok` 8→6** (+8–13%,
recall-gated). Prefill lever = **`max-num-batched-tokens` 4096→8192/16384**. **CORRECTION:** `index_topk`
2048→1024 is NOT a bandwidth lever (the indexer scans full shard-K before top-k) → only ~+2–5% with full
1M-recall risk → cheap-maybe, not a headline.

## What makes the DCP configs work (sparse-aware context-parallel KV)

vLLM replicates the MLA latent KV on every TP rank (MLA has one KV head), so 1M context overflows
a 121 GiB node. The DCP configs shard the KV across the 8 ranks (`--decode-context-parallel-size 8`)
— but stock DCP is incompatible with the DSA sparse path, so three in-container patches
(`glm52-dcp-patches.sh`, applied after the base `glm52-sparse-patches.sh`) make it correct:

- **A** — allow fp8 KV under DCP (relax the `mla_attention.py` gate; all-gather the bf16 query, keep KV fp8).
- **B (decode)** — the DSA indexer scores its local KV shard, so each rank picks a *local* top-2048.
  Fix (**PERF#2, production**): each rank takes its LOCAL top-2048, all-gathers only those candidates
  (a **FIXED-SIZE** `[rows, 8×2048]` exchange, context-independent), unions them, and re-selects the
  global top-2048 by a strict (score desc, global-pos asc) order → writes each rank its **owned**
  subset (`p%8`, local `p//8`, else −1). The sparse decode kernel returns its per-shard softmax LSE
  so vLLM's `cp_lse_ag_out_rs` recombines correctly. **+12% decode** (validated, real tokens) vs the
  original all-gather-full-logits reassembly, which is preserved as `glm52-dcp-patches-itemb.sh`.
- **C (prefill)** — same idea for prefill: each rank gathers its KV shard, `all_gatherv` the shard-K,
  reassemble to global per-request key order, run the prefill logits/top-k over the full K, owned-mask.

Empirically: DCP KV pool ≈ **4.3–4.8M tokens** (≈8× the 601k replicated pool) → 1M fits ~4.3×
over with no OOM. Coherence + 16k needle (3 depths) + 131k needle (3 depths) all pass; the 1M
config loads and serves coherently at **~18–19 tok/s** (PERF#2 decode; +12% over item-B, validated
real-token: working 15.7/16.9 → perf2 17.7/19.0 at 4k/256k).

### Caveat — prefill latency at extreme context (DCP only)
Decode is fast, but **prefill of very long prompts is slow** under DCP: the indexer K all-gather +
reassembly runs per-chunk-per-layer. ~131k prompts are practical; ~400k+ prefill is impractically
slow today (a 600k-token prefill exceeded a 30-min client timeout — the engine was healthy, just
slow). The decode optimization is **done** (PERF#2 above). The prefill caching attempt (**PERF#1**:
cache the reassembly index/gatherv sizes across layers) was a **NO-GO** — it deadlocked the
multi-chunk prefill in a variable-size all-gatherv (see `PERF-RESULTS.md`); revisit needs per-chunk
sizing + a gatherv-size-agreement assertion. Prefill slowness is **not** a correctness or
decode-speed issue. For prompts that fit 512k, `prod` both avoids this and is faster.

## DCP staging (one-time / after patch edits)

The DCP profiles need, on **all 8 nodes**:
- `~/glm-triton-dcp/` — copy of `~/glm-triton/` with the LSE-returning `flashmla_sparse.py`.
- `~/vllm-glm52/runtime/glm52-sparse-patches-dcp.sh` — combined base+DCP patch.

Regenerate + stage both after any patch change:
```bash
~/vllm-glm52/runtime/build-dcp-patch.sh --stage
```
(`build-dcp-patch.sh` concatenates `glm52-sparse-patches.sh` + `glm52-dcp-patches.sh`; the DCP
kernels are just `~/glm-triton` + the branch `kernels/flashmla_sparse.py`.)

## Watchdog

`sparks-glm52-watchdog` restarts via `launch.sh` with **production** env, so it would clobber a DCP
run. `start_glm52_config.sh` **stops** it for `dcp-*` and leaves it for manual re-enable on `prod`:
```bash
sudo systemctl start sparks-glm52-watchdog.timer    # ONLY when running prod (non-DCP 512k)
```
DCP runs are managed manually (no auto-restart).

## Recovery / redeploy from the repo (if a node or the head is rebuilt)

Everything needed is in this repo (`remote/glm52-gb10/`) — nothing is irreproducibly node-only.
To reconstruct a node (or all 8) from a fresh checkout:

1. **Base GLM-5.2 stack** (image, NCCL 2.30.4, b12x wheel, kernels, model weights) per
   `README.md` "Stage" + **`EXTERNAL-DEPS.md`** (the binaries are sha-pinned there; the image
   and `~/glm-triton/` kernels also exist on the surviving nodes — copy over RDMA is fastest).
2. **Deploy the launcher + patch toolchain to the head** `~/vllm-glm52/runtime/`:
   `launch.sh` (= `start_glm52_8node.sh`), `glm52-sparse-patches.sh`, `glm52-dcp-patches.sh`,
   `start_glm52_config.sh`, `build-dcp-patch.sh`, `watchdog_glm52_cluster.sh`. (The same four
   DCP files are also on every worker so any node can rebuild/launch.)
3. **Stage the DCP artifacts to all 8** (idempotent; regenerates the combined patch + the
   `~/glm-triton-dcp` kernels from the repo):
   ```bash
   ~/vllm-glm52/runtime/build-dcp-patch.sh --stage     # do NOT run while a DCP engine is live (it rm -rf's ~/glm-triton-dcp)
   ```
4. **Launch** any config: `~/vllm-glm52/runtime/start_glm52_config.sh {prod|dcp-512k|dcp-1m}`.

Reconciliation (2026-06-28): repo == deployed for `flashmla_sparse.py` (LSE),
`glm52-dcp-patches.sh`, `glm52-sparse-patches.sh`, `start_glm52_config.sh`, `build-dcp-patch.sh`;
combined patch + `~/glm-triton-dcp/flashmla_sparse.py` byte-identical across all 8 nodes; the head
regenerates the combined patch from repo sources (recovery-tested). The non-DCP `prod` path is
fully reproducible from `main` alone (the DCP additions are purely additive + branch-only).

## Status (2026-06-28)
Experimental DCP path on branch `glm52-1m-sparse-cp`; `prod` (non-DCP 512k) is the committed
production fallback on `main`. The cluster is currently running `dcp-1m`. Engineering writeup:
`ITEM-B-SPEC.md`, `PREFILL-ALLGATHER-SPEC.md`, `1M-SPARSE-CP-KV-PLAN.md`. Offline validators:
`validate_dcp_reassembly.py`, `validate_lse_recombine.py`; endpoint validator: `validate_dcp.py`.
