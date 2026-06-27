# GLM-5.2-NVFP4 + MTP on 8× DGX Spark (GB10, sm_121) — WORKING

Serves the **full `nvidia/GLM-5.2-NVFP4`** (465 GB, `glm_moe_dsa`: sparse-MLA +
lightning indexer + 256-expert MoE) across **all 8 GB10 nodes** at **TP=8** with
**native sm_121 DSA sparse attention** and **in-checkpoint MTP speculative decode**.

Ported from [`CosmicRaisins/glm-5.2-gb10`](https://github.com/CosmicRaisins/glm-5.2-gb10)
(proven on 4 nodes, TP=4, AWQ-INT4 15%-pruned + a *separate* reconstructed INT4 MTP
draft). Our adaptation runs the **full NVFP4 model on 8 nodes (no prune)** and uses
the **in-checkpoint layer-78 MTP draft** the nvidia export ships.

## Measured (first boot, 2026-06-27, eager mode, single-stream)

| Metric | Result |
|---|---|
| Decode throughput | **~22 tok/s** (eager, no cudagraph) — ~4.6× the old PP=8 dense (~4.8 tok/s) |
| MTP acceptance | 799/1323 draft tokens (~60%), **~2.8 accepted tokens/step** (k=3; matches GLM-5.2's ~2.76) |
| Coherence | clean short output + **31,246-token needle retrieved correctly** |
| Attention | native `FLASHMLA_SPARSE` on sm_121 (NOT the dense carve-out) |
| MoE | `FLASHINFER_CUTLASS` NVFP4 |
| Context | 262144 (256k) validated; 512k safe ceiling, 1M experimental (see below) |

PIECEWISE/FULL cudagraph (b12x) should lift decode further (~26–34 tok/s); eager is
the safe first-boot mode (cudagraph FULL is a known GB10 wedge for this sparse-MLA
family — vLLM #40969).

## How it works (what differs from a stock vLLM run)

1. **Image** `vllm-node-tf5-glm52:base` — eugr/spark-vllm-docker built at pinned vLLM
   `ab666069935c1f23e8ef56038b4659ac9e8f19f8` (post-0.23.0, GlmMoeDsa + indexer/MTP),
   `--tf5`. Built on the head, `docker save | load` fanned to all 8 over RDMA.
2. **Triton sparse-MLA kernels** (`kernels/`, from CosmicRaisins/jasl, Apache-2.0 —
   see `KERNELS-LICENSE`) deployed to `~/glm-triton/` and **mounted RO** over the
   image's `vllm/v1/attention/backends/mla/` + `ops/deepseek_v4_ops/` at launch.
3. **`glm52-sparse-patches.sh`** — the two non-vendored CosmicRaisins mods, applied
   **at container start** (idempotent, anchor-guarded against ab66606):
   - `glm52-sm12x-sparse`: patches `vllm/utils/deep_gemm.py` so
     `fp8_fp4_mqa_logits`/`fp8_fp4_paged_mqa_logits`/`tf32_hc_prenorm_gemm`
     short-circuit to the `sm12x_*` Triton fallbacks **before** the DeepGEMM
     `_missing()` gate; and relaxes `SparseAttnIndexer` so sm_121 doesn't require
     `has_deep_gemm()`.
   - `glm52-b12x-sparse`: `pip install --no-deps b12x==0.23.0`, then **probes the
     real decode import** and writes `/workspace/glm52-b12x.env` (FULL if b12x
     imports, else PIECEWISE — the cudagraph-capture-safe choice).
4. **NCCL 2.30.4** aarch64 `LD_PRELOAD`ed (fixes the 2.29.7 `shm_broadcast` wedge);
   `--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1` so NCCL uses
   RoCE/IB (NET/IB, not TCP — ~30 vs ~12 tok/s). **This is what makes TP=8 stable**
   where our prior bare TP=8 GLOO-wedged.
5. **In-checkpoint MTP**: `--speculative-config '{"method":"mtp",
   "num_speculative_tokens":3,"attention_backend":"FLASHMLA_SPARSE"}'` — no separate
   draft dir; vLLM self-resolves the draft to the served model's layer-78 nextn head.
6. **Per-node RDMA env** (head `enp1s0f0np0`/`rocep1s0f0`, workers
   `enP2p1s0f0np0`/`roceP2p1s0f0`) — `GLOO/TP/UCX_SOCKET_IFNAME` set to each node's
   OWN rail iface; never collapsed.
7. `--kv-cache-dtype fp8_ds_mla`, `--hf-overrides '{"qk_rope_head_dim":64}'`
   (the head_dim=192/qk_rope=64 alias collision), `--reasoning-parser glm45
   --tool-call-parser glm47`. GLM returns reasoning in the `reasoning` field.

## Stage (once, on the head; fans to all 8)

```bash
# NCCL 2.30.4 -> ~/models/nccl-2.30.4/libnccl.so.2 on every node
# kernels/*.py -> ~/glm-triton/ on every node
# glm52-sparse-patches.sh -> ~/vllm-glm52/runtime/ on every node (verify byte-identical sha256!)
# b12x-0.23.0 wheel -> ~/models/wheels/ on every node
# image vllm-node-tf5-glm52:base -> loaded on every node
# nvidia/GLM-5.2-NVFP4 weights -> ~/models/GLM-5.2-NVFP4 on every node
```

## Launch (from the head)

```bash
# Safe first boot — eager, single stream, conservative util (isolates the
# distributed/sparse/MTP bring-up from the cudagraph wedge). Stops the live model
# containers incl. model-router (which holds :8000).
for h in 192.168.88.{101..108}; do ssh blacksheeep@$h 'sync; echo 3|sudo tee /proc/sys/vm/drop_caches>/dev/null'; done
CONFIRM_GLM52=YES ENFORCE_EAGER=1 MAX_NUM_SEQS=1 GPU_MEMORY_UTILIZATION=0.72 \
  ~/vllm-glm52/runtime/launch.sh

# Production step-up (after a coherence soak): cudagraph + more slots
CONFIRM_GLM52=YES MAX_NUM_SEQS=4 ~/vllm-glm52/runtime/launch.sh     # cudagraph from the b12x probe
~/vllm-glm52/runtime/launch.sh --stop                              # reap all 8
~/vllm-glm52/runtime/launch.sh --dry-run                           # preview per-rank commands
```

Serves OpenAI-compatible on `192.168.88.101:8000` as `glm-5.2-nvfp4`.

## Gotchas (learned the hard way)

- **`model-router` holds `:8000`.** GLM binds `:8000` with `--network host`; the
  router MUST be stopped first or rank0 dies `OSError Errno 98 Address already in
  use` AND the `/health` poll gets a false 200 from the router. It's now in the
  launcher's `STOP_CONTAINERS`.
- **First boot eager, not cudagraph.** cudagraph FULL + sparse-MLA + MTP on sm_121
  can hang after ~6 requests (#40969). Step to PIECEWISE then FULL only after a soak.
- **sparse can corrupt silently** — gate acceptance on a coherence + long-context
  needle test, not just `/health` 200.
- Keep `GPU_MEMORY_UTILIZATION ≤ 0.78` for TP on GB10 unified memory (the launcher
  refuses higher without `ALLOW_HIGH_GPU_UTIL=YES`) — a high-util OOM wedges the
  whole cluster (ping-only, power-cycle to recover).

## Context: 512k / 1M

See **"Update 2026-06-27"** below for the validated 512k production config and the
1M feasibility verdict (this section's earlier pre-fix guidance was superseded).

## Attribution

Kernels and the overall recipe: **CosmicRaisins/glm-5.2-gb10** (Apache-2.0), itself
building on jasl's V4 sparse-MLA, the eugr/spark-vllm-docker image, and Z.ai's
GLM-5.2. This directory adapts it to the full NVFP4 model on 8 nodes with the
in-checkpoint MTP draft. See `KERNELS-LICENSE`.

## Update 2026-06-27 — 512k context unlocked (persistent_topk fix)

The DSA **decode** indexer crashed at configured `max_model_len > ~397k` in
`torch.ops._C.persistent_topk` (`topk.cu`): on GB10 it needs `total_ctas =
ceil(max_model_len/8472)` cooperative CTAs (31@256k, **62@512k, 124@1M**) >
`num_sms*occupancy=48`, and the `FilteredTopK` fallback needs ≥128 KB smem vs
sm_121's 99 KB → `RuntimeError`, killing the engine. It's gated by **configured
max_model_len (stride), not prompt length** (so 256k was always fine).

**Fix (runtime, in `glm52-sparse-patches.sh` Step A4 — standalone, sentinel
`# GLM52_SM12X_DECODE_REROUTE`):** add `and not
current_platform.is_device_capability_family(120)` to the decode gate in
`sparse_attn_indexer.py`, so GB10 takes the in-tree `else` →
`ops.top_k_per_row_decode` (set-equivalent top-k, 8 KB smem, one block/row, no
SM/smem wall). **Validated at 512k**: decode works, 0 `persistent_topk` errors,
~22.5 tok/s (no regression), and a **300,040-token needle retrieved correctly**.

**512k is the safe production ceiling.** Validated, serving config:
```bash
CONFIRM_GLM52=YES MAX_MODEL_LEN=524288 MAX_NUM_SEQS=4 GPU_MEMORY_UTILIZATION=0.78 \
  ENFORCE_EAGER=1 ~/vllm-glm52/runtime/launch.sh
```
No `ALLOW_HIGH_GPU_UTIL` needed — the topk fix makes 512k fit at the safe 0.78 util.
KV pool ≈ **601k tokens** shared across the 4 slots (one full-512k stream, or 4
concurrent shorter requests; typical AnythingLLM turns leave large headroom).
~656k is reachable single-stream at util 0.85 with **no quality change** (it's just
more memory, soak-test vs the wedge). fp8_ds_mla KV throughout — no MXFP4.

**1M is a SOFTWARE limit, not hardware — and NOT reachable via vLLM's DCP.** The
cluster physically has the memory (8×121 = 968 GiB; 1M needs ~555 GiB *with sharded
KV*), but vLLM **replicates** the MLA KV on all 8 ranks (~50 GiB/node at 1M) because
MLA's single KV head can't be TP-sharded — that 8× duplication overflows a 121 GiB
node, not a real shortage. vLLM's `--decode-context-parallel-size` (DCP) is a **dead
end here, verified 3 ways in the live container**: (1) DCP hard-refuses fp8 KV
(`mla_attention.py:788 assert not fp8_attention`); (2) the DSA indexer has **no
cross-rank top-k**, so sharding the context makes each rank pick top-2048 of its own
1/8 → silently wrong attention; (3) sparse `forward_mqa` returns `None` for the
softmax LSE DCP requires. **Do not use `--decode-context-parallel-size`.**

True 1M needs a **sparse-aware context-parallel KV** implementation: shard the
fp8_ds_mla KV + a distributed indexer doing a global cross-rank top-2048 + an
all-to-all gather of the selected KV + the fp8 LSE combine. With that, 1M fits at a
**safe ~0.66 util keeping fp8 (no quality loss)** — but it is ~weeks of
upstream/research-grade work (the correctness-critical piece is the distributed
top-k; cf. vLLM RFC #30055). Until then, **512k is the production ceiling**
(≤~656k single-stream via util, quality-neutral).

## Update 2026-06-27 (b) — NVFP4 MoE illegal-memory-access + watchdog

The `FLASHINFER_CUTLASS` NVFP4 MoE kernel (TensorRT-LLM `cutlass_fused_moe` →
`cudaMemsetAsync(final_output…)`) can **intermittently** hit a CUDA illegal memory
access that kills EngineCore (seen after hours of uptime, incl. surviving the 300k
needle, then a shape/concurrency edge tripped it). The container stays `Up`
(entrypoint `sleep infinity`) but `vllm serve` exits → `/health` dies.

**Note:** you can NOT force a different NVFP4 MoE kernel with the global
`--moe-backend marlin` — that flag is global and GLM's **unquantized MTP MoE**
rejects it (`ValueError: moe_backend='marlin' is not supported for unquantized
MoE`). A targeted NVFP4-only backend swap (patch `fused_moe/oracle/nvfp4.py`'s
`AVAILABLE_BACKENDS` to drop the FlashInfer entries → VLLM_CUTLASS/MARLIN) is the
deeper root-fix, but is unvalidated; pursue only if the IMA recurs often.

**Mitigation in place: `sparks-glm52-watchdog`** (`tools/watchdog_glm52_cluster.sh`
+ `tools/systemd/sparks-glm52-watchdog.{service,timer}`). Every 60s it restarts
GLM **only** when `/health` is dead AND the in-container `vllm serve` process is
gone (true engine death — never during a slow long-context prefill, which keeps the
process alive). Two consecutive dead checks required; shares `restart.lock` (flock)
with manual launches. The stale `sparks-vllm-watchdog` / `sparks-kimi-watchdog`
timers are disabled so they cannot fight GLM.

## Update 2026-06-27 (c) — NVFP4 MoE root-fix: force MARLIN (VALIDATED)

The intermittent FlashInfer-CUTLASS IMA is now **root-fixed** (the watchdog from (b)
is kept only as a backstop). `glm52-sparse-patches.sh` **STEP C** (sentinel
`# GLM52_NVFP4_MOE_BACKEND`) edits **only** `fused_moe/oracle/nvfp4.py`: on
capability-family 120 it strips the FlashInfer backends **and** `VLLM_CUTLASS` from
`AVAILABLE_BACKENDS` so the NVFP4 MoE auto-selects **MARLIN**. The unquantized
in-checkpoint MTP MoE (`oracle/unquantized.py`) is untouched — which is why this
surgical patch works where the global `--moe-backend marlin` flag fails.

Why MARLIN (not VLLM_CUTLASS): VLLM_CUTLASS shares the **same broken SM120
block-scaled grouped-GEMM** as FlashInfer → ~5 tok/s **garbage**. MARLIN is **w4a16**
(dequantizes the NVFP4 weights in-GEMM with bf16 activations) — a different codepath
that never touches the broken FP4 grouped-GEMM, and it discards activation scales by
design so the NVFP4 global-scale-factor concern is moot. vLLM itself already falls
MXFP8 MoE back to Marlin on sm_121.

**Validated 2026-06-27:** `Using 'MARLIN' NvFp4 MoE backend → ['MARLIN','EMULATION']`;
coherent + correct output (reasoning/code/instruction, temp 0 — NOT garbage);
**~22.5 tok/s decode (no regression)**; KV pool 601,600 tokens (512k fits, 1.15×).

**Memory note:** MARLIN's w4a16 weights are slightly larger, so 512k needs
**`GPU_MEMORY_UTILIZATION=0.78`** (at 0.72 the KV pool was ~0.7 GiB short of the
26.72 GiB one 512k seq needs — clean `ValueError` at init, no wedge). 0.78 gives
32.89 GiB KV. Production config is therefore:
```bash
CONFIRM_GLM52=YES ENFORCE_EAGER=1 MAX_MODEL_LEN=524288 MAX_NUM_SEQS=4 \
  GPU_MEMORY_UTILIZATION=0.78 ~/vllm-glm52/runtime/launch.sh
```
