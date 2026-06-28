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
~/vllm-glm52/runtime/start_glm52_config.sh dcp-1m      # DCP, true 1M context
```

Each stops the prior model containers, drops caches on all 8, applies the in-container patches,
and serves OpenAI-compatible on `192.168.88.101:8000` as `glm-5.2-nvfp4`. First boot ~12–15 min
(465 GB load). Validate coherence + a long-context needle before trusting it (sparse attention
can corrupt silently — never gate on `/health` alone): `validate_dcp.py`.

## What makes the DCP configs work (sparse-aware context-parallel KV)

vLLM replicates the MLA latent KV on every TP rank (MLA has one KV head), so 1M context overflows
a 121 GiB node. The DCP configs shard the KV across the 8 ranks (`--decode-context-parallel-size 8`)
— but stock DCP is incompatible with the DSA sparse path, so three in-container patches
(`glm52-dcp-patches.sh`, applied after the base `glm52-sparse-patches.sh`) make it correct:

- **A** — allow fp8 KV under DCP (relax the `mla_attention.py` gate; all-gather the bf16 query, keep KV fp8).
- **B (decode)** — the DSA indexer scores its local KV shard, so each rank picks a *local* top-2048.
  Fix: all-gather the indexer logits across the DCP group → reassemble to global key order → global
  top-2048 → write each rank its **owned** subset (`p%8`, local `p//8`, else −1). Plus the sparse
  decode kernel returns its per-shard softmax LSE so vLLM's `cp_lse_ag_out_rs` recombines correctly.
- **C (prefill)** — same idea for prefill: each rank gathers its KV shard, `all_gatherv` the shard-K,
  reassemble to global per-request key order, run the prefill logits/top-k over the full K, owned-mask.

Empirically: DCP KV pool ≈ **4.3–4.8M tokens** (≈8× the 601k replicated pool) → 1M fits ~4.3×
over with no OOM. Coherence + 16k needle (3 depths) + 131k needle (3 depths) all pass; the 1M
config loads and serves coherently at ~17–18 tok/s.

### Caveat — prefill latency at extreme context (DCP only)
Decode is fast, but **prefill of very long prompts is slow** under DCP: the indexer K all-gather +
reassembly runs per-chunk-per-layer. ~131k prompts are practical; ~400k+ prefill is impractically
slow today (a 600k-token prefill exceeded a 30-min client timeout — the engine was healthy, just
slow). This is an optimization target (cache the reassembly index across layers; lighten the
collective), **not** a correctness or decode-speed issue. For prompts that fit 512k, `prod` both
avoids this and is faster.

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
