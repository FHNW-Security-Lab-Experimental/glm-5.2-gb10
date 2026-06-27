# GLM-5.2-NVFP4 + MTP on 8× DGX Spark (GB10, sm_121)

Serves the **full `nvidia/GLM-5.2-NVFP4`** (465 GB, `glm_moe_dsa`: sparse-MLA +
lightning indexer + 256-expert MoE) across **all 8 GB10 nodes** at **TP=8**, with
**native sm_121 DSA sparse attention**, **in-checkpoint MTP speculative decode**, and
the **MARLIN** NVFP4 MoE backend. Served as `glm-5.2-nvfp4` on the head `:8000`.

Ported from [`CosmicRaisins/glm-5.2-gb10`](https://github.com/CosmicRaisins/glm-5.2-gb10)
(proven on 4 nodes / TP=4 / AWQ-INT4-15%-pruned + a *separate* INT4 MTP draft); this
runs the **full NVFP4 model on 8 nodes (no prune)** and uses the **in-checkpoint
layer-78 MTP draft** the nvidia export ships (no separate draft model).

## Performance

| Metric | Value |
|---|---|
| Decode throughput | **~22.5 tok/s** single-stream (eager) |
| MTP acceptance | **~2.8 accepted tokens/step** (k=3; ~60% of drafts) |
| Context | **512k** (`max_model_len=524288`), KV pool ≈ 601k tokens |
| Coherence | correct short + reasoning/code output; **300k-token needle retrieved** |
| Attention | native `FLASHMLA_SPARSE` on sm_121 |
| MoE | `MARLIN` NVFP4 (w4a16) |
| KV | `fp8_ds_mla` |
| Stability | survives concurrent load (6-way burst: 67 reqs, 0 errors, 0 MoE crashes) |

Decode is memory-bandwidth-bound (~40 B active params), so the MoE backend choice has
negligible throughput impact; MTP is the decode lever.

## How it works

1. **Image** `vllm-node-tf5-glm52:base` — eugr/spark-vllm-docker built at pinned vLLM
   `ab666069935c1f23e8ef56038b4659ac9e8f19f8` (post-0.23.0: `GlmMoeDsa` + indexer/MTP),
   `--tf5`. Built on the head, `docker save | load`-fanned to all 8 over RDMA.
2. **Triton sparse-MLA kernels** (`kernels/`, from CosmicRaisins/jasl, Apache-2.0 — see
   `KERNELS-LICENSE`) deployed to `~/glm-triton/`, **mounted RO** over the image's
   `vllm/v1/attention/backends/mla/` + `ops/deepseek_v4_ops/`.
3. **`glm52-sparse-patches.sh`** runs in each container before `vllm serve` (idempotent,
   anchor-guarded vs `ab66606`, byte-identical on all 8). Three sm_121 fixes + b12x:
   - **Sparse wiring (`glm52-sm12x-sparse`)** — patches `vllm/utils/deep_gemm.py` so
     `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` / `tf32_hc_prenorm_gemm`
     short-circuit to the `sm12x_*` Triton fallbacks *before* the DeepGEMM `_missing()`
     gate, and relaxes `SparseAttnIndexer` so sm_121 doesn't require `has_deep_gemm()`.
     Without it the DSA sparse path has no working sm_121 backend.
   - **Decode top-k reroute (`# GLM52_SM12X_DECODE_REROUTE`)** — on capability-family 120
     the DSA decode indexer takes the in-tree `ops.top_k_per_row_decode` branch instead of
     `torch.ops._C.persistent_topk`. The compiled `persistent_topk` needs a cooperative
     grid of `ceil(max_model_len/8472)` CTAs (62 @512k, 124 @1M) > GB10's
     `num_sms*occupancy=48`, and its `FilteredTopK` fallback needs ≥128 KB smem vs sm_121's
     99 KB → it would `RuntimeError` and kill the engine for any decode once
     `max_model_len > ~397k`. `top_k_per_row_decode` is the set-equivalent top-k (8 KB
     smem, one block/row) — this is what makes 512k work.
   - **NVFP4 MoE → MARLIN (`# GLM52_NVFP4_MOE_BACKEND`)** — on capability-family 120,
     strips the FlashInfer backends **and** `VLLM_CUTLASS` from
     `fused_moe/oracle/nvfp4.py`'s `AVAILABLE_BACKENDS` so the NVFP4 MoE auto-selects
     **MARLIN**. `FLASHINFER_CUTLASS` (the default) routes into TRT-LLM `cutlass_fused_moe`
     and **intermittently CUDA-illegal-memory-accesses** (`cudaMemsetAsync(final_output)`),
     killing EngineCore; `VLLM_CUTLASS` shares the same broken SM120 grouped-GEMM →
     garbage. MARLIN is **w4a16** (dequant-in-GEMM, bf16 activations) — a different
     codepath that never touches the broken FP4 kernel, and correct (it discards activation
     scales by design, so the NVFP4 global-scale-factor concern is moot). The edit is
     **NVFP4-oracle-only** so the unquantized in-checkpoint MTP MoE is untouched — which is
     why the global `--moe-backend marlin` flag (rejected by the unquantized MoE) can't be
     used here.
   - **b12x** — `pip install --no-deps b12x==0.23.0` + a probe that picks the cudagraph
     mode (only relevant if cudagraph is enabled; production runs eager).
4. **NCCL 2.30.4** aarch64 `LD_PRELOAD`ed (2.29.7 has an aarch64 `shm_broadcast` wedge) +
   `--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1` so NCCL uses
   RoCE/IB (`NET/IB`, not TCP). With NCCL 2.30.4 this is what makes **TP=8 stable** on
   GB10 (bare TP=8 GLOO-wedged before).
5. **In-checkpoint MTP** — `--speculative-config '{"method":"mtp",
   "num_speculative_tokens":3,"attention_backend":"FLASHMLA_SPARSE"}'` (no separate draft
   dir; vLLM resolves the draft to the served model's layer-78 nextn head).
6. **Per-node RDMA env** — head `enp1s0f0np0`/`rocep1s0f0`, workers
   `enP2p1s0f0np0`/`roceP2p1s0f0`; `GLOO/TP/UCX_SOCKET_IFNAME` = each node's own rail
   iface, never collapsed.
7. **Engine settings** — `--kv-cache-dtype fp8_ds_mla`, `--hf-overrides
   '{"qk_rope_head_dim":64}'` (the `head_dim=192`/`qk_rope=64` alias collision would
   otherwise die "704 exceeds 576" at load), `--enforce-eager`, `--reasoning-parser glm45
   --tool-call-parser glm47` (GLM returns reasoning in the `reasoning` field).

## Memory & context

512k runs at **`GPU_MEMORY_UTILIZATION=0.78`** (the launcher's TP ceiling; no
`ALLOW_HIGH_GPU_UTIL` needed). That gives **32.89 GiB KV** = a ~601k-token pool, enough
for one full 512k stream (1.15×) or several shorter concurrent ones. MARLIN's w4a16
weights are slightly larger than the FlashInfer layout, so 0.72 is too low (the pool
falls ~0.7 GiB short of one 512k seq's 26.72 GiB — a clean `ValueError` at init, no
wedge); 0.78 is the right value. `fp8_ds_mla` KV throughout — no MXFP4.

**512k is the production ceiling. True 1M is a *software* gap, not hardware.** The
cluster has the memory (8×121 = 968 GiB; 1M needs ~555 GiB with *sharded* KV), but vLLM
**replicates** the MLA latent KV on every rank (~50 GiB/node @1M — MLA has one KV head, so
TP can't shard it); that 8× duplication overflows a 121 GiB node. The KV-sharding lever,
`--decode-context-parallel-size` (DCP), is a **dead end for DSA** (verified in the live
container): DCP hard-refuses fp8 KV (`mla_attention.py:788`), the DSA indexer has no
cross-rank top-k (sharding would silently corrupt attention), and the sparse path returns
no softmax LSE. **Do not use `--decode-context-parallel-size`.** Real 1M needs a
**sparse-aware context-parallel KV** implementation (shard fp8 KV + distributed global
top-2048 + cross-rank KV gather + fp8 LSE combine) — upstream/research-grade work (cf.
vLLM RFC #30055), which would fit 1M at a safe ~0.66 util keeping fp8. ~656k is reachable
single-stream at util 0.85 with no quality change if needed (soak-test vs the OOM-wedge).

## Stage (once, on the head; fans to all 8)

```bash
# NCCL 2.30.4         -> ~/models/nccl-2.30.4/libnccl.so.2   on every node
# kernels/*.py        -> ~/glm-triton/                        on every node
# glm52-sparse-patches.sh -> ~/vllm-glm52/runtime/            on every node (verify byte-identical sha256!)
# b12x-0.23.0 wheel   -> ~/models/wheels/                     on every node
# image vllm-node-tf5-glm52:base                              loaded on every node
# nvidia/GLM-5.2-NVFP4 weights -> ~/models/GLM-5.2-NVFP4       on every node
```

## Launch / operate (from the head)

```bash
# Production launch (drop caches first — the 465 GB mmap saturates the unified pool):
for h in 192.168.88.{101..108}; do ssh blacksheeep@$h 'sync; echo 3|sudo tee /proc/sys/vm/drop_caches>/dev/null'; done
CONFIRM_GLM52=YES ENFORCE_EAGER=1 MAX_MODEL_LEN=524288 MAX_NUM_SEQS=4 \
  GPU_MEMORY_UTILIZATION=0.78 ~/vllm-glm52/runtime/launch.sh

~/vllm-glm52/runtime/launch.sh --dry-run   # preview the per-rank docker/serve commands
~/vllm-glm52/runtime/launch.sh --stop      # reap the container on all 8
```

The launcher stops the prior model containers incl. **`model-router`** (which holds
`:8000`; if left running, rank0 dies `OSError Errno 98 Address already in use` and the
`/health` poll gets a false 200 from the router). First boot loads ~58 GB/node — allow
~12 min cold, then a coherence + long-context needle check before trusting it (sparse
attention can corrupt silently, so don't gate on `/health` 200 alone). Serves
OpenAI-compatible on `192.168.88.101:8000` as `glm-5.2-nvfp4`.

## Watchdog (safety net)

`sparks-glm52-watchdog` (`watchdog_glm52_cluster.sh` + `systemd/sparks-glm52-watchdog.{service,timer}`,
deployed + enabled on the head) restarts GLM via the launcher **only** when `/health` is
dead AND the in-container `vllm serve` process is gone (true engine death — never during a
slow prefill, which keeps the process alive). Two-strike, shares `restart.lock` (flock); a
restart re-applies all patches incl. MARLIN. The stale `sparks-vllm-watchdog` /
`sparks-kimi-watchdog` timers are disabled so they can't fight GLM. With the MARLIN MoE
fix the crash class is removed at the source; the watchdog is a backstop for anything
unforeseen.

## Attribution

Kernels and the overall recipe: **CosmicRaisins/glm-5.2-gb10** (Apache-2.0), itself
building on jasl's V4 sparse-MLA, the eugr/spark-vllm-docker image, and Z.ai's GLM-5.2.
This directory adapts it to the full NVFP4 model on 8 nodes with the in-checkpoint MTP
draft and the MARLIN MoE backend. See `KERNELS-LICENSE`.
