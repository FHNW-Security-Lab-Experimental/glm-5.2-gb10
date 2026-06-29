# PERF#2-for-prefill — candidate-union prefill (STEP C rewrite) implementation spec

**Status: IMPLEMENTED + BOOT-TESTED 2026-06-29 → NO-GO. Production keeps item-C prefill.**
Correct (needle PASS at 32k/131k/256k) and scales sublinearly, BUT **+23% slower at 131k**
(353 s vs item-C 287 s, after the torch.topk-union fix which only saved ~6%) and **wedges at
512k**. Root cause is fundamental, not the union kernel: the candidate all-gather moves
`N × (query-tokens) × topk` per layer — for *prefill* there are thousands of query rows per
chunk (vs 1–4 for decode), so the candidate exchange dominates at low/moderate context, and
the per-chunk local-logits working set (`context/8` wide) collides with the reserved KV pool at
512k. The candidate-union pattern transfers to **decode** (few rows → the +12% in production)
but **not to prefill** (many rows). Fails the "not slower at small context" bar → not promoted.
All code on branch `glm52-prefill-port` (production `main` + tag `glm52-perf2-prod-2026-06-28`
untouched) for a future revisit (e.g. a true distributed top-k that avoids the full candidate
all-gather, or context-routing that only uses it for the 256k–~450k band where it beats item-C).

## Goal
Replace item-C's **slow** prefill (gather local shard-K → `all_gatherv` the FULL shard-K →
reassemble GLOBAL K → `fp8_fp4_mqa_logits` + `top_k_per_row_prefill` over the FULL global K)
with the proven PERF#2 **candidate-union**: each rank runs the prefill logits + top-k over its
**LOCAL** shard-K (8× narrower), takes LOCAL top-2048 carrying global within-request key
positions, **fixed-size** all-gathers only those candidates `[chunk_rows, N×2048]`, unions,
re-selects the exact global top-2048, owned-masks. Deletes the O(prompt²/chunk) `all_gatherv`
+ the full-global-width logits/top-k. Expected ~1.5–3× @256k, >5× @512k–1M, kills the ~930k wedge.

## Key source facts (live `sparse_attn_indexer.py` / `indexer.py`)
- `cu_seqlen_ks/ke` are **per query-token** (length `output_query_len`), built by
  `_build_prefill_chunk_metadata_kernel`. `cu_seqlen_ks[t] = cu_seq_lens[q]` (request q's GLOBAL
  K-row start); `cu_seqlen_ke[t] = cu_seqlen_ks[t] + causal_count_t` (compress_ratio==1).
- `top_k_per_row_prefill(logits, ks, ke, out, ...)` returns **within-request** positions
  (0..causal−1) — proven because the existing owned-mask interprets them as global within-request
  positions via `_ti % N == r` (owner=p%N). NOT global-K-row indices.
- `fp8_fp4_mqa_logits(q, K, weights, ks, ke)` → `logits[chunk_rows, K_width]`; ks/ke bound each
  query token to its request's key range.
- Already present (item-C, reuse): `dcp_local_cu_seq_lens` ([num_reqs+1] local key prefix sum),
  `dcp_local_total_seq_lens`, `dcp_local_seq_lens_allranks`, the local gather into
  `k_quant_local_full`, and `_own(gt,rk) = gt//N + clamp((gt%N)−rk, 0, 1)`.
- Interleave-1: owner(p)=p%N, local(p)=p//N. The lp-th owned key of a request on rank r is at
  global within-request pos `lp*N + r`.

## The two load-bearing pieces

### A. LOCAL per-token `ks/ke` (derive in the layer, per chunk — no kernel change)
```
causal = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks                 # [rows] global causal count
q      = searchsorted(chunk.cu_seq_lens, chunk.cu_seqlen_ks, right=False)  # [rows] request id
                                                                 # (cu_seqlen_ks[t] == cu_seq_lens[q])
local_ks = chunk.dcp_local_cu_seq_lens[q]                        # [rows] local K start
owned    = (causal // N) + clamp((causal % N) - r, 0, 1)         # [rows] owned causal count = _own(causal, r)
local_ke = local_ks + owned
```
`local_ks/ke` are int32, index the LOCAL K buffer `_kq_loc` (per-request-concat-local order).

### B. local → global mapping
`top_k_per_row_prefill` over local logits returns LOCAL within-request `lp` (0..owned−1).
Global within-request pos `gp = lp*N + r`. (Invalid lanes: kernel emits -1 → keep -1.) Then the
union picks the global top-2048 by strict `(score desc, gp asc)`, identical to decode.

## Prefill-loop edits (`sparse_attn_indexer.py`, gate all on `_glm52c_N > 1`; N==1 byte-identical)
Replace item-C's gather+all_gatherv+reassembly block + the global logits/top-k with:
```
# (gather: keep LOCAL shard-K into _kq_loc; reuse across skip_kv_gather sub-chunks)
if not chunk.skip_kv_gather:
    if N<=1: ops.cp_gather_indexer_k_quant_cache(kv_cache, k_quant, k_scale, block_table, cu_seq_lens)   # stock
    else:    ops.cp_gather_indexer_k_quant_cache(kv_cache, _kq_loc, _ks_loc, block_table, dcp_local_cu_seq_lens)  # LOCAL only; NO all_gatherv, NO reassembly
# every chunk, N>1:
#   build local_ks/ke (A); local logits over (_kq_loc,_ks_loc) with local_ks/ke;
#   top_k_per_row_prefill(local_logits, local_ks, local_ke, _loc_idx, ...)  -> lp
#   valid=_loc_idx>=0; gp = where(valid, _loc_idx*N + r, -1); sc = gather(local_logits, gp-relative)... 
#       (gather the score at each selected lp from local_logits row, like decode r_r1 step 2)
#   _allsc = all_gather(sc, dim=1); _allpos = all_gather(gp, dim=1)        # FIXED-SIZE [rows, N*topk]
#   two-stable-sort (score desc, gp asc) -> sel; topk_indices = gather(_allpos, sel) (where score>-inf else -1)
#   owned-mask (UNCHANGED): topk_indices = where(topk_indices%N==r & >=0, topk_indices//N, -1)
```
The score-gather mirrors decode r_r1: `topk_indices_buffer.new_empty` workspace for `_loc_idx`;
gather `local_logits` at `_loc_idx` (safe-index invalid→0 then mask to -inf); `*N+r` for gp.

`fp8_fp4_mqa_logits` + `top_k_per_row_prefill` now run over the LOCAL K — **8× narrower**, and
the FULL-K `all_gatherv` + `index_select` reassembly are **deleted**.

### skip_kv_gather (CORRECTION from item-C)
- The **K gather** is inside `if not skip_kv_gather` (reuse `_kq_loc` across sub-chunks via the
  persistent `k_quant_local_full` buffer — do NOT clear it between sub-chunks).
- The **candidate exchange** (local logits → top-k → all_gather → union → owned-mask) runs **every
  chunk** (each sub-chunk has its own query tokens). The `all_gather` is **fixed-size**
  `[chunk_rows, N*topk]` and `chunk_rows` is rank-uniform (queries are replicated, only KV is
  sharded) → no `all_gatherv`, no size divergence, NOT the PERF#1 deadlock class.

## `indexer.py` edits
None beyond item-C's existing `dcp_local_*` fields (the local ks/ke derive in the layer from
them + `cu_seqlen_ks/ke` + `cu_seq_lens`). Keep the `compress_ratio==1` assert.

## Validation (gate every step on set-equality + long needle, NEVER /health)
1. **Offline (extended `validate_perf2_prefill.py`):** add a localization unit-test — for random
   per-token `causal` and per-request layouts, assert `local_ke−local_ks == _own(causal,r)` and
   that gathering the LOCAL K rows `[local_ks:local_ke]` + mapping `lp→lp*N+r` reproduces exactly
   the owned subset of the global causal range. Plus the existing candidate-union set-equality.
2. **Adversarial review** of the written patch (text): kernel arg order/shapes; the `searchsorted`
   request-id; off-by-one in `owned`; fixed-size all_gather symmetry; skip_kv_gather; anchors; N==1.
3. **In-container (boot-test):** capture the candidate-union top-k SET vs an N=1 reference for a
   >2048-token prompt; then needle 128k → 512k → 1M coherence + prefill timing vs item-B.

## Rollback
Separate from production: this rewrites STEP C on `glm52-prefill-port` only. Production `main`
(perf2 decode + item-C prefill) and the deployed `c45be5db` are untouched until validated +
explicitly promoted. Item-B + item-C reassembly remain the proven path.
