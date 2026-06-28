# GLM-5.2-NVFP4 — fp8 + 1M via sparse-aware context-parallel KV (engineering plan)

Status: **in progress on branch `glm52-1m-sparse-cp`.** Production stays 512k
(`remote/glm52-gb10/` on `main`) and is untouched. This is the build to lift fp8 KV to
1M context on 8× GB10, keeping fp8 (no MXFP4). Anchors below were read from the **live**
vLLM `ab66606` source in the running container on 2026-06-28.

## Why this is the only path

1M fp8 is a **software gap, not a memory wall**. vLLM replicates the MLA latent KV (and
the DSA indexer cache) on every TP rank because MLA has one KV head (`num_kv_heads==1`), so
TP cannot shard it. At 1M that replicated state overflows the 121 GiB node:

| per node @1M, fp8 | replicated (today) | main-KV sharded, indexer **replicated** (M1) | both sharded (M2) |
|---|---|---|---|
| NVFP4/MARLIN weights | 58 | 58 | 58 |
| main MLA latent KV | 54.7 | 6.8 (÷8) | 6.8 |
| DSA indexer KV | 11.0 | 11.0 | 1.4 (÷8) |
| activations / MTP / workspace + prefill workspaces | ~17 | ~17 | ~13 (CP-aware) |
| overhead (NCCL/ctx/framework) | ~5 | ~6 | ~6 |
| **total** | **~145 → overflow** | **~99 → fits, tight** | **~85 → fits, headroom** |

So sharding the KV across ranks (vLLM's "decode context parallel", DCP) is the lever. The
problem is that stock DCP does not work for the DSA sparse path. The good news from reading
the live code: **most of DCP is already wired** — the hard part is one item (global top-k).

## What the live source already gives us (verified 2026-06-28)

- **The CP attention recombine is already implemented** in
  `vllm/model_executor/layers/attention/mla_attention.py`:
  - L271-272 import `cp_lse_ag_out_rs` and `dcp_a2a_lse_reduce`.
  - L797 already calls `attn_out, lse = self.impl.forward_mqa(...)`.
  - L800-809: when `dcp_world_size>1` it already does `dcp_a2a_lse_reduce(attn_out, lse, …,
    is_lse_base_on_e=True)` (or `cp_lse_ag_out_rs`). Contract (from `v1/attention/ops/common.py`):
    `out:[B,H,D]`, `lse:[B,H]`, base-e. **→ item E is free.**
- **The decode kernel already returns the LSE.** `kernels/flashmla_sparse.py`
  `_fp8_flash_mla_kernel` returns `(out, lse)` (base-e); it was only discarded at
  `forward_mqa` (returned `lse=None`). **→ item C is plumbing (now done on this branch).**
- **The merge math already ships.** `kernels/sparse_mla_kernels.py`
  `merge_two_sparse_mla_subsets_with_sink`, `finish_gathered_sparse_mla_attention`.
- **The indexer block table is already CP-sized.** `v1/attention/backends/mla/indexer.py:311`
  `cdiv(max_model_len, block_size * get_total_cp_world_size())` — so when DCP is on, the
  indexer already addresses a 1/N shard. That is exactly why item B is needed: each rank's
  indexer would otherwise pick a **local** top-2048.
- **CP token ownership is interleaved**, not contiguous: `v1/worker/gpu/cp_utils.py`
  distributes token `i` to rank `i % (dcp_size*cp_interleave)`, round-robin
  (`cp_kv_cache_interleave_size` default 1). Any ownership mask in item B MUST use
  `slot % cp_world_size == rank`, not `slot // capacity`.

## Empirical results — DCP=8 diagnostic boot (2026-06-28)

A diagnostic boot on all 8 nodes at 512k with `--decode-context-parallel-size 8` + items
A & C (staged in isolated dirs; production untouched; watchdog disabled; rolled back to 512k
after). What it proved:

- **Memory thesis CONFIRMED.** The KV pool went from 601,600 tokens (replicated) to
  **4,849,664 tokens** at the same util 0.78 — almost exactly 8×. KV sharding is real, and
  **1M fp8 is comfortably memory-feasible** (1M needs 1M pool tokens; we have ~4.85M). This is
  the decisive result.
- **Item A works.** The engine boots under DCP with fp8_ds_mla (the `assert not fp8_attention`
  is gone; the query-tuple path is taken); MTP loads; the MoE autotunes; warmup completes.
- **The recombine (E) is genuinely on the hot path.** `mla_attention.py` calls
  `cp_lse_ag_out_rs` for every mqa step under DCP.
- **C was incomplete (now fixed) — and it was plumbing after all.** The first real request
  crashed with `cp_attn_lse … NoneType.contiguous()`: sparse routes **prefill and mixed-batch**
  through `forward_mqa` too (`num_mqa_tokens = q.size(0)`), but those paths returned `lse=None`.
  A first read suggested the bf16 prefill kernel had no LSE — **that was wrong**: the real
  prefill kernel `flash_mla_sparse_fwd_triton` already returns `(out, max_logits, lse)` (base-e,
  `sm12x_sparse_mla_attn.py:263`); the earlier `(out, valid)` was a *different* helper
  (`_gather_dequant_fp8ds`). So C is genuinely plumbing for prefill too — **now implemented on
  this branch**: `_bf16_flash_mla_kernel`, `_forward_fp8_kv_mixed_batch`, and
  `_forward_fp8_kv_separate_prefill_decode` all take `return_lse=True` and thread the LSE;
  `forward_mqa` routes both fp8 paths through them under DCP. Production (`return_lse=False`) is
  byte-for-byte unchanged.
- **Not yet tested:** (a) that the LSE shapes/base are exactly what `cp_lse_ag_out_rs` wants
  (validation test #1), and (b) correctness of the sharded-indexer top-k (item B).

### Empirical results — DCP=8 boot #2 (2026-06-28, with the prefill-LSE fix)

- ✅ **No crash.** The exact request that died with `NoneType` now serves (HTTP 200). Item C
  plumbing is complete — the LSE shapes are accepted by `cp_lse_ag_out_rs`. KV pool 4,751,360
  tokens (sharding active).
- ❌ **Output is corrupted** even at 17 tokens: `"OKOKOK…"` + nonsense reasoning. Production
  (non-DCP) is coherent for the same prompt → this is DCP-specific.
- **LSE is NOT the cause.** Source confirms both fp8 kernels return base-e LSE in `[b, H, T]`:
  `flash_mla_with_kvcache_triton` natively, `b12x_glm_mla_attention` converts base-2→e
  (`lse.mul(math.log(2.0))`). The C reshape to `[T, H]` matches `cp_lse_ag_out_rs`. So the LSE
  plumbing is correct.
- **Root cause = the sharded-KV gather/index correctness (item B), and it corrupts at ANY
  length.** Under DCP each rank holds an *interleaved* 1/8 KV shard, but the sparse
  `topk_indices` are GLOBAL slot ids and the indexer picks a per-rank LOCAL top-2048. So each
  rank gathers the wrong KV and the recombine stitches wrong partials → garbage. This is not
  context-length-dependent (wrong at 17 tokens), confirming it's the fundamental sharded-gather
  problem, not a long-context edge case.

### Revised remaining work (empirically grounded)

1. ~~Prefill-under-DCP LSE~~ — **done** (item C; engine serves under DCP without crashing).
2. **Item B — the sharded-gather correctness (the wall).** Two coupled pieces, both required:
   (a) distributed cross-rank global top-2048 for the indexer (gather + re-top-k, interleaved
   ownership `slot % cp_world_size`); (b) DCP-aware index→local-slot mapping so each rank gathers
   the correct KV for the global ids it owns. This is the research-grade core.
   - *Optional de-risk first:* the LSE-recombine unit check (validation #1) to 100%-confirm the
     LSE path is innocent before investing in B, and to have a regression harness.
3. Then the validation gates below at 512k (A/B vs replicated) → 1M.

## Two milestones

### Milestone 1 — main-KV sharded, indexer **replicated** (no distributed top-k)

The shortcut. If the DSA indexer cache is **not** sharded (every rank holds the full
indexer KV and computes logits over the whole sequence), then each rank already computes the
**correct global top-2048 locally** — no cross-rank top-k needed. Each rank then attends to
the subset of that global top-2048 whose **main** KV it physically owns, and the existing
LSE recombine (E) stitches the shards. Memory still closes (~99 GiB/node, table above) at a
tighter util (~0.80–0.85). This proves the end-to-end CP path (boot, recombine, coherence,
1M) **without** the research-grade item B.

The work for M1:
- **A** (done, `glm52-dcp-patches.sh`): allow fp8 KV under DCP.
- **C** (done, `kernels/flashmla_sparse.py`): `forward_mqa` returns the per-shard base-e LSE
  for fp8 pure-decode when `dcp_world_size>1`; non-DCP path byte-unchanged.
- **D-lite**: make the indexer cache opt **out** of DCP sharding (stay replicated) while the
  main MLA KV opts in. Concretely, the indexer's block-table sizing at `indexer.py:311` uses
  `get_total_cp_world_size()`; for M1 it must size for the full sequence and the indexer KV
  must be written replicated (every rank writes every token's indexer key). This is the main
  open coding task for M1 and needs live experimentation (is the indexer a separate
  kv_cache_group that can be excluded from DCP, or does it need a write-replicated override?).
- **G**: launcher `--decode-context-parallel-size 8`, run `glm52-dcp-patches.sh` in-container
  after `glm52-sparse-patches.sh`, `MAX_NUM_SEQS=1`, eager, `cp_kv_cache_interleave_size=1`.

### Milestone 2 — co-shard the indexer (the research-grade item B)

Reclaim the last ~14 GiB/node and run at safe util. This is the distributed cross-rank
**global top-2048** for the lightning indexer.

- **B**: in `model_executor/layers/sparse_attn_indexer.py` (decode top-k is at L335-365,
  already rerouted to `ops.top_k_per_row_decode` by `glm52-sparse-patches.sh` step A4): make
  it a **two-pass gather + re-top-k** (NOT an allreduce — summing sparse logits is
  meaningless):
  1. each rank scores its KV shard, takes local top-2048 carrying **global** slot ids;
  2. NCCL `all_gather` over the DCP group → `[num_q, 8×2048]` candidates;
  3. re-run `top_k_per_row` to get the exact global top-2048 (exact, because the global
     winners are a subset of the per-shard top-2048 unions);
  4. each rank masks to slots it owns (`slot % cp_world_size == rank`) and attends those.
  All ranks must agree bit-for-bit on the id set or the shards desync.
- **D**: co-shard the indexer KV with the main KV; make the two `max_model_len`-scaled
  prefill workspaces (`indexer.py:229` `max_model_len*40`; sparse `5*max_model_len`)
  DCP-aware to reclaim ~10.8 GiB/node.

## Constraints / invariants

- **F**: `cp_kv_cache_interleave_size` MUST stay 1. With MTP (`--speculative-config`
  mandatory in prod) and interleave>1, `cp_utils.py` requires
  `supports_mtp_with_cp_non_trivial_interleave_size`, which the sparse impl does not set.
  Keep `--enforce-eager` (DCP all_gather inside a captured graph is fragile).
- Prefill/mixed decode-steps under CP are **not** handled yet (C asserts pure-decode). 1M is
  a decode regime; prefill-CP is a later add.
- 512k on `main` remains the production fallback throughout.

## Validation — DSA sparse corrupts SILENTLY (`/health` stays 200)

Functional tests are mandatory; never gate on `/health`. In order:
1. **LSE-recombine unit check** (post C, no DCP): split one decode's KV into two artificial
   subsets, attend each, merge via `merge_two_sparse_mla_subsets_with_sink`; assert merged ==
   single-pass to fp32 tol. This is what confirms the C reshape (`lse.transpose(1,2)…`) and
   base/layout are right.
2. **Global top-k SET-EQUALITY** (the load-bearing test): for a fixed prompt, log the chosen
   top-2048 **global** slot ids from the replicated path vs the DCP path; assert set-equal per
   query token. The naive local-top-k bug compiles, runs, and is wrong — only this catches it.
   (Instrument the indexer to dump ids; compare offline.)
3. **Cross-rank determinism**: all 8 ranks agree bit-for-bit on the id set each step.
4. **Short-ctx A/B** (`validate_dcp.py`, temp 0): DCP vs replicated, token-for-token match at
   8k–32k.
5. **Needle-in-haystack** (`validate_dcp.py`): unique fact at varied depths, 128k→512k→1M;
   exact retrieval.
6. **>512k coherence soak**: multi-turn past 512k and at ~1M; no "word salad" degradation.
7. **MTP-under-CP**: interleave=1, `num_speculative_tokens=3` active, acceptance + output
   match the non-spec DCP run.

Go-live only after 2, 5, 6 pass at 1M.

## Adopt vs build

Build. No CUDA, fp8-KV, sparse-MLA context-parallel KV exists today. vllm-ascend PR #4702 is
NPU-only; vLLM RFC #30055 is a design blueprint; DeepSeek-V4's 1M uses native KV compression
(GLM-5.2 is V3.2-class DSA, no compressors). **Track RFC #30055 / the vllm-ascend line** — a
CUDA backend there could flip B from build to adopt. SGLang's "Lightning TopK" radix-select
is the reference algorithm for B.

## State on this branch

- `kernels/flashmla_sparse.py` — **C done** (`can_return_lse_for_decode`,
  `_forward_fp8_kv_decode_with_lse`, `forward_mqa` returns LSE when `dcp_world_size>1`).
- `glm52-dcp-patches.sh` — **A done** (confirm-gated `CONFIRM_GLM52_DCP=YES`, anchor-guarded).
- `validate_dcp.py` — coherence + needle harness (runs against any endpoint, incl. the live
  512k one as a baseline).
- **Open**: D-lite (M1 indexer-replicated), then B + D (M2), then G (launcher), then the
  validation gates on the cluster (Shelly recovery ready for wedges).
