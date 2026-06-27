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

- **512k single-stream** is the safe long-context ceiling:
  `CONFIRM_GLM52=YES MAX_MODEL_LEN=524288 MAX_NUM_SEQS=1 GPU_MEMORY_UTILIZATION=0.85
  ALLOW_HIGH_GPU_UTIL=YES ~/vllm-glm52/runtime/launch.sh` (coherence-soak first).
- **1M** (`MAX_MODEL_LEN=1048576`) needs gmu ~0.90 and is experimental — collides
  with the GB10 unified-mem OOM-at-load wedge; attempt supervised with Shelly
  power-cycle recovery armed.

## Attribution

Kernels and the overall recipe: **CosmicRaisins/glm-5.2-gb10** (Apache-2.0), itself
building on jasl's V4 sparse-MLA, the eugr/spark-vllm-docker image, and Z.ai's
GLM-5.2. This directory adapts it to the full NVFP4 model on 8 nodes with the
in-checkpoint MTP draft. See `KERNELS-LICENSE`.
