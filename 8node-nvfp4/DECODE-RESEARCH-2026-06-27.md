# GLM-5.2-NVFP4 on 8× DGX Spark — max-decode serving research (2026-06-27)

Goal: serve the **full `nvidia/GLM-5.2-NVFP4`** (465 GB) across **all 8 GB10 nodes**
(no prune / no REAP) for **maximum decode tokens/sec**. This consolidates four
parallel investigations: the in-repo research workflow + adversarial review, the
`CosmicRaisins/glm-5.2-gb10` recipe (the breakthrough), KTransformers, and
`bird/GLM-spark`.

## Operational status (this session)

- **Download:** `nvidia/GLM-5.2-NVFP4` (465 GB) + `nvidia/MiniMax-M3-NVFP4` (250 GB)
  downloading to the head `~/models/` via `~/dl-models.sh` (hf_transfer, Xet disabled).
  HuggingFace throttles this IP to **~42 MB/s sustained** (aria2 `-x16` bursts to
  ~520 MB/s but 403s on Xet's 596 MiB byte-range-locked signed URLs; `hf_xet` ~1 MB/s;
  hf_transfer ~42 MB/s with zero errors = the reliable choice). ETA ~4.5 h.
- **Distribution:** `scripts/distribute_models_rdma.sh` (deployed to head) parallel-
  rsyncs both models from the head to 102–108 over the **200G RDMA fabric (10.0.0.x)**
  once the download completes.

## The checkpoint choice — full NVFP4 on 8 nodes (settled), with one honest caveat

- The **full 465 GB NVFP4 model fits 8 nodes** (~58 GB weights/node, large KV
  headroom) — so **no prune/REAP is needed** (that is only forced at 3–4 nodes).
  `bird/GLM-spark` (REAP-469B, 3 nodes, DSA disabled, ~4.4 tok/s) is the path we are
  explicitly NOT taking.
- **Caveat for the record:** on GB10, decode is **memory-bandwidth-bound** (~40 B
  active params over ~273 GB/s LPDDR5X), so the quant *format* barely moves decode
  speed — `CosmicRaisins` measured **NVFP4 ~10 tok/s vs AWQ-INT4 ~13 tok/s** (NVFP4
  slightly slower + some instability) pre-MTP. **The real lever is MTP, not the quant.**
  We proceed with NVFP4 as instructed; the upside is that the **nvidia NVFP4 export
  ships the MTP draft head** (see below), so MTP is available.

## What the nvidia checkpoint unlocks vs the old Mapika export

| Aspect | Mapika (old, deployed) | nvidia/GLM-5.2-NVFP4 (downloading) |
|---|---|---|
| MTP / nextn draft | **absent** (0 nextn tensors, max layer 77) → MTP impossible | **present**: full DeepSeek-MTP layer-78 head (791 tensors, `eh_proj`, enorm/hnorm, 256-expert MoE, `num_nextn_predict_layers=1`, bf16 draft) → **MTP viable** |
| Dense carve-out | required | **still required** — `glm_moe_dsa`, `index_topk=2048` ⇒ vLLM forces `FLASHMLA_SPARSE`, which has no clean sm_121 build |
| `qk_rope_head_dim=64` override | required | **still required** (head_dim=192 alias collision → "704 exceeds 576") |
| KV dtype | bf16 only (no k/v scales) | **bf16 only** — zero `k_scale`/`v_scale` tensors despite card recommending `fp8_e4m3`. **Do NOT use fp8_e4m3 (word-salad trap)** |
| generation_config | manual patch needed | ships `temp 1.0 / top_p 0.95 / do_sample` — no patch |

## Two serving paths

### Path A — SAFE baseline: PP=8 dense (proven, ~4.8 tok/s)
The only topology that has ever stayed up under the 465 GB / all-8-nodes constraint.
`TP=1 PP=8`, dense-MLA carve-out, bf16 KV, `qk_rope_head_dim=64`, `num_stages=1` MLA
patch, skip-mm-profiling, via `remote/glm52-vllm/start_glm52_vllm_cluster.sh` on
`vllm-node-dsv4-official:latest`.
- **Single-stream ~4.8 tok/s** (network-bound on 7 inter-node hops/token over RoCE;
  CUDA graphs are ~flat here, do not promise more). **Aggregate** grows with
  `MAX_NUM_SEQS` (raise 4 → 8 → 12–16) — the only real throughput lever on this path;
  projected ~18–30 tok/s at 8 concurrent (UNMEASURED — verify).
- **No MTP, no TP** (MTP forces TP via `deepseek_mtp` lacking `SupportsPP` →
  `NotImplementedError` under PP; and our TP=8 GLOO-wedges ~30 min post-load).
- Boot hygiene mandatory: `drop_caches` on all 8 nodes first (465 GB mmap saturates
  the 121 GB unified pool). A 4-node pre-flight cell is **impossible** (model doesn't
  fit 4 nodes) — de-risk with a staged eager→graphs 8-node boot + node-local abort guards.

### Path B — MAX decode: port the CosmicRaisins GB10 recipe (~22–27 tok/s)
`CosmicRaisins/glm-5.2-gb10` (NVIDIA-forum "22 tok/s, 256k ctx" recipe) cracks **both**
historical blockers on the *same RoCE fabric as us* (identical HCAs
`rocep1s0f0`/`roceP2p1s0f0`, ifaces `enP7s7,enp1s0f0np0,enP2p1s0f0np0`):

1. **Native DSA sparse on sm_121** — ports jasl's V4 Triton sparse-MLA kernels as
   drop-in V3.2 replacements (the Hopper-only `_flashmla_C` path), plus **DeepGEMM
   arch-gate fallbacks** (`fp8_fp4_mqa_logits` / `paged_mqa_logits`) so the lightning
   indexer runs on sm12x (vLLM #41063 gate bypassed). Mods live in
   `eugr/spark-vllm-docker`: `mods/glm52-sm12x-sparse` + `mods/glm52-b12x-sparse`.
   Backend `FLASHMLA_SPARSE`, KV `fp8_ds_mla`.
2. **MTP speculative decode** — `--speculative-config '{"method":"mtp",
   "num_speculative_tokens":3,"attention_backend":"FLASHMLA_SPARSE"}'`. CosmicRaisins
   *reconstructed* an INT4 draft (their AWQ base dropped layer 78); **our nvidia NVFP4
   export ships layer 78**, so we may load MTP from the in-checkpoint draft head or
   build an NVFP4-aligned draft (their `mtp/` pipeline: dequant → requant → re-key to
   `DeepSeekMTP` layout → verify-load). **MTP ≈ doubles decode (~8→~22 tok/s).**
3. **`b12x==0.23.0`** (`pip install --no-deps`, `GLM52_B12X_MLA=1`) → cudagraph-capture-
   safe sparse decode → `--compilation-config '{"cudagraph_mode":"FULL"}'`. Without
   b12x you must use `PIECEWISE` or it crashes on capture. Avoid `flashinfer_b12x`
   (crashes on sm_121).
4. **NCCL 2.30.4 aarch64** `LD_PRELOAD` (2.29.7 has an aarch64 `shm_broadcast` deadlock)
   + `docker run --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1`
   (or NCCL silently drops to TCP → ~12 vs 30+ tok/s). **This IB passthrough + NCCL
   2.30.4 is likely what avoids our prior GLOO/TP wedge.**

Measured: **4 nodes TP=4 + MTP k=3 ≈ 22 tok/s** decode @256k; a commenter ran
**unpruned on 8× GB10 ≈ 26–27 tok/s with MTP**; a tuned variant (draft at TP=1) hit
**~34 tok/s**. Pinned vLLM `ab66606…` (post-0.23.0), env `GLM52_BIND_HOST_TRITON=1`,
`GLM52_MQA_LOGITS_TRITON=1`, `GLM52_PAGED_MQA_TRITON=1`,
`GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192`, `--max-num-seqs 1`, gpu-util 0.93 (4-node).

### Topology for the FULL model on 8 nodes (the open engineering question)
CosmicRaisins is **TP=4 on 4 nodes** with a *pruned/AWQ* model. The full 465 GB NVFP4
does **not** fit 4 nodes, so two TP=4 replicas of the full model are impossible. The
full model on 8 nodes must be ONE job:
- **TP=8** + CosmicRaisins stack — the 8-node commenter's ~26–27 tok/s suggests it
  works *with* NCCL 2.30.4 + IB passthrough (the missing pieces in our prior wedged
  TP=8). Highest expected decode; needs validation that MTP composes with TP=8.
- **Hybrid PP=2 × TP=4** — keeps each TP group at CosmicRaisins' proven size (4) and
  lets MTP live inside the TP=4 group while spanning 8 nodes. Likely the safest way to
  get MTP on the full model. Lower aggregate than TP=8 (KV duplicated across the PP
  boundary) but avoids an unproven TP=8.
- **PP=8 dense** — Path A fallback (~4.8 tok/s), MTP/sparse off.

## Recommendation

1. **Now (while weights land):** nothing blocks — download + distribute are running.
2. **First serve = Path A** (PP=8 dense) on the nvidia checkpoint to validate weights
   + coherence end-to-end with zero new risk (~4.8 single / aggregate via `MAX_NUM_SEQS`).
3. **Then pursue Path B for max decode**, in a maintenance window with a power-cycle
   plan: build/adapt the `eugr/spark-vllm-docker` image with `glm52-sm12x-sparse` +
   `b12x`, add NCCL 2.30.4 + `/dev/infiniband` passthrough, and bring up **hybrid
   PP=2×TP=4 + MTP + FLASHMLA_SPARSE** (fall back to TP=8 if MTP needs one TP group;
   PP=8 dense if sparse/MTP misbehave). Target **~22–27 tok/s** decode.
   - Verify the in-checkpoint layer-78 draft loads via `--speculative-config` against
     the NVFP4 base; if not, run CosmicRaisins' `mtp/` pipeline to build an
     NVFP4-aligned draft.
   - sm_121 sparse can corrupt **silently** (wrong index_topk) → coherence-soak, not
     just a health check.

## Sources
- `CosmicRaisins/glm-5.2-gb10` · forum: developer.nvidia.com/t/.../374125 ·
  drafts: hf.co/CosmicRaisins/GLM-5.2-MTP-INT4-aligned · `eugr/spark-vllm-docker`
- KTransformers GLM-5.2 tutorial — **not viable on GB10** (kt-kernel CPU experts are
  x86 AMX/AVX-only; our CPU is ARM; single-node). Useful only as: GLM-5.2 sparse also
  runs via SGLang `--attention-backend nsa`; parsers glm45/glm47; temp 1.0/top_p 0.95.
- `bird/GLM-spark` — the "avoid" template (REAP-469B, 3 nodes, DSA off, ~4.4 tok/s).
- vLLM issues #45317 (sm_121 sparse gap), #43477 (FlashInfer SM120 sparse), #46726
  (indexer assert), #41063 (DeepGEMM arch gate), #40082 (b12x SM121 NVFP4 MoE).
- In-repo: `remote/glm52-vllm/{start_glm52_vllm_cluster.sh,patch-glm-dense.py,
  patch-glm-sparse.py,README.md}`, memories `glm52-qk-rope-head-dim-override`,
  `glm51-dsa-gb10-dense-carveout`, `gb10-tp-unified-memory-oom`.
