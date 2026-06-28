# GLM-5.2-NVFP4 on 8× DGX Spark (GB10, sm_121)

Run the **full `nvidia/GLM-5.2-NVFP4`** (465 GB, **no prune**) across **8× DGX Spark GB10** at
**TP=8** — native sm_121 DSA sparse attention, in-checkpoint MTP speculative decode, MARLIN NVFP4
MoE — from **512k** production up to **true 1M context** via a **sparse-aware context-parallel KV**.

This is the **8-node, full-model** line of work. The **entire runtime for all three serving configs**
— every patch, kernel, launcher, the base-image build recipe, the engineering writeups, and the
reproduction info — lives in **[`8node-nvfp4/`](8node-nvfp4/)**. The original 4-node / AWQ-pruned
upstream is credited at the bottom.

## Three serving configs — one launcher

```bash
8node-nvfp4/start_glm52_config.sh prod        # non-DCP 512k  — fastest, production default
8node-nvfp4/start_glm52_config.sh dcp-512k    # DCP 512k      — more long-context concurrency
8node-nvfp4/start_glm52_config.sh dcp-1m      # DCP true 1M context
```

| Config | Context | Decode tok/s¹ | Parallel streams (KV pool) | KV layout |
|---|---|---|---|---|
| **`prod`** (non-DCP) | 512k | **~22.5** | ~1 full-512k (4-slot cap) | MLA KV replicated per rank |
| **`dcp-512k`** | 512k | ~20–21 | **~9 × 512k** (8 slots) | MLA KV sharded 8-way |
| **`dcp-1m`** | **1,048,576** | ~17–18 | **4 × 1M** (vLLM: 4.12×) | MLA KV sharded 8-way |

¹ single-stream, warm, GB10, 2026-06-28. Slots ≈ `min(--max-num-seqs, KV-pool ÷ context)`. Full
table (slots vs context × config), tuning, recovery: **[`8node-nvfp4/CONFIGS.md`](8node-nvfp4/CONFIGS.md)**.

Use **`prod`** for anything ≤512k (it's materially faster); use the DCP configs to exceed 512k or for
far more long-request concurrency (sharding gives ~8× the KV pool).

## What runs

| | |
|---|---|
| Model | full `nvidia/GLM-5.2-NVFP4` — 465 GB, no prune (~58 GB weights/node) |
| Hardware | 8× DGX Spark GB10 (sm_121a, **121 GiB unified** mem/node), 200G RoCE fabric |
| Parallelism | TP=8; **DCP=8** (decode context-parallel) for the sharded-KV configs |
| Attention | native sm_121 **DSA**: sparse-MLA + lightning indexer (top-2048/query) |
| Speculative | **in-checkpoint MTP** (the nvidia export's layer-78 nextn head), k=3 |
| KV / MoE | `fp8_ds_mla` / **MARLIN** NVFP4 (w4a16) |
| Serving | OpenAI-compatible on the head `:8000` as `glm-5.2-nvfp4` |

## Highlights — what was solved on 8× GB10

- **Full NVFP4 on all 8 nodes, no prune** — the entire 465 GB model at TP=8 (not a reduced/pruned variant).
- **Native DSA sparse on sm_121** — ported Triton sparse-MLA + lightning indexer off the Hopper-only
  `_flashmla_C` path, with a DeepGEMM arch-gate bypass.
- **In-checkpoint MTP** — ~2.8 accepted tokens/step → ~22.5 tok/s (≈4.6× the old PP=8 dense baseline).
- **512k** — fixed the GB10 `persistent_topk` cooperative-grid / 99 KB-smem limit (reroute decode to
  `top_k_per_row_decode`); validated with a 300k-token needle.
- **NVFP4 MoE crash root-fixed** — FlashInfer-CUTLASS illegal-memory-access eliminated by forcing
  **MARLIN** (NVFP4-oracle patch), with a watchdog backstop.
- **TRUE 1M context** — a **sparse-aware context-parallel KV**: shard the MLA latent KV across the 8
  ranks (DCP), a **distributed global top-2048** for the DSA indexer (decode *and* prefill), and a
  per-shard softmax-LSE recombine. Memory closes (KV pool ~4.3M tokens, ~4.3× a full 1M) and the model
  serves **4 parallel 1M streams** coherently. Needles pass at 16k/131k across depths.

## How the 1M unlock works (sparse-aware context-parallel KV)

vLLM replicates the MLA latent KV on every TP rank (MLA has one KV head), so 1M overflows a 121 GiB
node. The DCP configs shard it across ranks — but stock DCP is incompatible with the DSA sparse path,
so three in-container patches (`8node-nvfp4/glm52-dcp-patches.sh`, applied after the base
`glm52-sparse-patches.sh`) make it correct:

- **A** — allow fp8 KV under DCP (relax the gate; all-gather the bf16 query, keep KV fp8).
- **B (decode)** — each rank scores only its KV shard, so it would pick a *local* top-2048. Fix:
  all-gather the indexer logits → reassemble to global order → global top-2048 → each rank keeps its
  *owned* subset (`p%8`, local `p//8`, else −1); the sparse decode kernel returns its per-shard LSE so
  vLLM's `cp_lse_ag_out_rs` recombines exactly.
- **C (prefill)** — same for prefill: gather each rank's KV shard, `all_gatherv` it, reassemble to
  global per-request key order, run the prefill logits/top-k over the full key set, owned-mask.

Engineering writeups (in `8node-nvfp4/`): `1M-SPARSE-CP-KV-PLAN.md`, `ITEM-B-SPEC.md`,
`PREFILL-ALLGATHER-SPEC.md`, `ITEM-B-DESIGN.md`, `CUTOVER-2026-06-27.md`, `DECODE-RESEARCH-2026-06-27.md`.
Offline validators: `validate_dcp_reassembly.py`, `validate_lse_recombine.py`; endpoint validator:
`validate_dcp.py`.

## Quickstart (from the head)

```bash
# 1) stage runtime to all 8 nodes (image, NCCL 2.30.4, kernels, b12x, patches, model) — see 8node-nvfp4/README.md
#    DCP configs also need: 8node-nvfp4/build-dcp-patch.sh --stage   (builds + stages the combined patch + DCP kernels)
# 2) launch any config:
~/vllm-glm52/runtime/start_glm52_config.sh dcp-1m            # or prod / dcp-512k
curl -s http://192.168.88.101:8000/v1/models                 # served as glm-5.2-nvfp4
# 3) validate (sparse attention can corrupt silently — never trust /health alone):
8node-nvfp4/validate_dcp.py --base http://192.168.88.101:8000/v1 --model glm-5.2-nvfp4 \
  --needle-tokens 131072 --depths 0.1,0.5,0.9
```

Full runbook (staging, per-node RDMA env, operate, watchdog): **[`8node-nvfp4/README.md`](8node-nvfp4/README.md)**.

## Reproducibility — everything is here

The whole 8× runtime is in `8node-nvfp4/`: the launcher (`start_glm52_8node.sh`), the 3-config
starter (`start_glm52_config.sh`), the base patches (`glm52-sparse-patches.sh`) + the DCP A/B/C
patches (`glm52-dcp-patches.sh`), all Triton kernels (`kernels/`), the watchdog (`watchdog_glm52_cluster.sh`
+ `systemd/`), validators, the b12x wheel (`wheels/`), and the model-distribution helper (`scripts/`).

- **Base image** — the build recipe is vendored at `8node-nvfp4/image/spark-vllm-docker/` with
  `image/BUILD-IMAGE.md` (exact command + the `--vllm-ref ab66606 --tf5` pins). All source build-deps
  are forked + ref-pinned under `FHNW-Security-Lab-Dependencies` (vLLM `ab66606`, flashinfer `v0.6.13`,
  NCCL `v2.30u1`, DeepGEMM `nv_dev`) for cluster-gone rebuilds.
- **Binary deps** — image / NCCL shas + provenance in `8node-nvfp4/EXTERNAL-DEPS.md`; b12x wheel vendored.
- **Not in git**: only the 465 GB weights (HF `nvidia/GLM-5.2-NVFP4`) — back them up off-cluster.

## Caveats

- **`prod` (non-DCP 512k) is the fast path** (~22.5 tok/s); DCP trades ~10–20% decode for >512k
  context / ~8× the KV pool. Pick per workload.
- **DCP prefill of very long prompts is slow** (the per-chunk-per-layer indexer all-gather): ~131k
  prompts are practical, ~400k+ prefill is currently impractically slow — a known optimization target.
  Decode speed and correctness are unaffected.
- The DCP/1M path is **experimental**; gate any rollout on the long-context needle + a coherence soak.

## Attribution / upstream

Ported from [`CosmicRaisins/glm-5.2-gb10`](https://github.com/CosmicRaisins/glm-5.2-gb10) — proven on
**4 nodes / TP=4 / AWQ-INT4 with a data-free 15% expert prune + a separate reconstructed INT4 MTP
draft, at 256k**. **This fork runs the full NVFP4 model on 8 nodes (no prune)**, uses the
**in-checkpoint** MTP draft the nvidia export ships, and extends context to **512k → 1M** with the
sparse-aware context-parallel KV above. Builds on jasl's V4 sparse-MLA, the `eugr/spark-vllm-docker`
image, and Z.ai's GLM-5.2. (The upstream 15% prune is coherence-checked, not benchmarked — not
relevant here; this fork is unpruned.)
