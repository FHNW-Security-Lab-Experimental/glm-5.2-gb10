# Item B — sparse-aware global top-k under DCP (design)

The wall for fp8 + 1M. Items A, C, E are done: under `--decode-context-parallel-size 8` the
engine boots, KV shards (4.75M-token pool), and the LSE recombine runs. But output is corrupted
because the DSA top-k is computed **per-shard**, so each rank attends a different key set and the
recombine stitches incompatible softmaxes. This documents the fix, grounded in the live
`ab66606` source.

## Exact corruption mechanism (source-level)

Decode indexer path, `vllm/model_executor/layers/sparse_attn_indexer.py`:
```
logits = fp8_fp4_paged_mqa_logits(q, kv_cache, weights, seq_lens,
                                  decode_metadata.block_table, schedule_metadata, ...)
# logits: [num_rows, max_seq] — q·k_index scores
ops.top_k_per_row_decode(logits, next_n, seq_lens, topk_indices, ...)  # -> top-2048 per query
topk_indices_buffer[...] = topk_indices
```
Under DCP, `decode_metadata.block_table` and `seq_lens` are the **per-rank sharded** ones
(`prepare_dcp_local_seq_lens`, interleaved `slot % cp_world_size`). So `logits` cover only the
rank's 1/8 of keys → `top_k_per_row_decode` returns a per-rank **local** top-2048.

`FlashMLASparseImpl.forward_mqa` then reads `topk_indices_buffer` and attends those keys. Each
rank attends its own local-top-2048 (a different key set per rank); `cp_lse_ag_out_rs` combines
them as if they were partitions of one softmax — they are not → garbage (seen at 17 tokens).

For correctness, every rank must attend `(global top-2048) ∩ (its shard)`. Two coupled fixes:
- **B1**: compute the global top-2048 (agreed across ranks).
- **B2**: each rank attends only the owned subset of those globals, mapped to local slots.

## Candidate approaches

### Approach 1 — logits all-gather (recommended; no position bookkeeping)
In the indexer decode path, under DCP:
1. Compute local `logits` [num_rows, local_seq] as today.
2. `all_gather` logits across the DCP group and reassemble to global order. Because ownership is
   interleaved (`pos % cp == rank`), global logits column `p` = rank `p%cp`'s local column
   `p//cp`. A single `all_gather` + an interleave gather (or a strided scatter into a
   `[num_rows, global_seq]` buffer) reconstructs it. Decode `num_rows` is tiny (≤ batch × next_n,
   e.g. ≤ 4 with MTP), so even at 1M this is ~16 MB/rank gathered — fine over RDMA.
3. `top_k_per_row_decode` over the **global** logits → global top-2048 positions, identical on
   all ranks (deterministic).
4. Write the global top-2048 to `topk_indices_buffer`.
Then in `forward_mqa` (B2): mask `topk_indices` to owned positions (`idx % cp_world_size ==
dcp_rank`, else the -1 sentinel the kernels already skip), so each rank attends only its shard's
share of the global set → partial attention → existing LSE recombine.
- **Pro:** top-k positions are already global; B2 is a one-line modulo mask; reuses the stock
  top-k kernel. **Con:** reassembling interleaved logits into global order must exactly match the
  kernel's `seq_lens`/block layout; the all-gather grows with seq (acceptable at decode widths).

### Approach 2 — local top-k gather + re-top-k (memory-light, more bookkeeping)
Each rank keeps its local top-2048 `(score, GLOBAL position)`; `all_gather` → `[num_rows,
8×2048]`; re-`top_k_per_row` → exact global top-2048 (exact, since the global winners are a
subset of the per-shard top-2048 unions). Then B2 mask as above.
- **Pro:** gather is fixed-size (8×2048), independent of seq. **Con:** must convert local→global
  positions before the gather and carry scores; two-pass.

### Approach 3 — dense-MLA-under-DCP milestone (correctness first, not sparse)
Skip the indexer under DCP and run **dense** MLA over the sharded KV — the *stock vLLM dense MLA
DCP path, already correct upstream*. Gives a correct 1M immediately, but decode is O(seq) not
O(2048): slow at 1M (likely single-digit tok/s or worse). Useful only to *prove* end-to-end 1M
coherence and de-risk the recombine, then move to Approach 1/2 for speed. (Mirrors the GLM-5.1
dense carve-out, but for CP.)

**Recommendation:** Approach 1 for the real fix; optionally Approach 3 first as a one-boot
correctness checkpoint (confirms LSE recombine + memory end-to-end at 1M before investing in the
distributed top-k).

## B2 — the owned-mask + slot mapping (common to 1 & 2)

`topk_indices` from the indexer are per-request token positions. Ownership (interleave=1):
`owner(p) = p % cp_world_size`. In `forward_mqa` under DCP, before
`triton_convert_req_index_to_global_index`, set `topk_indices[idx % cp_world_size != dcp_rank] =
-1`. Then `convert` maps the surviving (owned) positions through the **local** (sharded)
block_table to local cache slots. **Open item:** confirm `convert` + the sharded block_table
yield the correct local slot for an owned global position — this is the part to verify first
(unit test below), as it's where an off-by-`cp` indexing bug would hide.

## Why this needs a dev harness, not blind boots

Each full-cluster DCP boot is ~15 min + production downtime, and a wrong distributed top-k fails
*silently* (coherent-looking but wrong). Develop against fast checks instead:

1. **LSE-recombine unit test** (`validate_lse_recombine.py`, runs on the live 512k container, no
   DCP, no downtime): split one decode's top-k keys into N disjoint subsets, attend each via the
   sparse kernel, merge with `sparse_mla_kernels.merge_*`/the cp_lse formula, assert ==
   single-pass to fp32 tol. Locks in that the LSE base/layout is correct (the foundation) and
   gives a regression target.
2. **Global-top-k set-equality** (instrument the indexer to dump selected positions): assert the
   DCP path's chosen set == the replicated path's set, per query. This is THE test that catches
   a wrong distributed top-k (which otherwise looks plausible).
3. Only after 1 & 2 pass offline: one DCP boot → needle → coherence soak at 512k → 1M.

## Harness #1 result (2026-06-28) — FOUNDATION PROVEN

`validate_lse_recombine.py`, run in the live production container (no downtime):
```
max|out_merged - out_full| = 4.0e-03  (tol 2e-2) ok
max|lse_merged - lse_full| = 9.5e-07  (tol 1e-3) ok
LSE-RECOMBINE: PASS
```
The sparse kernel's `(out, lse)` compose *exactly* via the LSE-weighted merge DCP uses, so the
recombine math + LSE base/layout (items C/E) are correct. **The DCP corruption is therefore
item B alone** (each rank attends a different, non-disjoint key set instead of an owned partition
of one global top-2048).

**No-downtime GPU dev loop unlocked:** production saturates the GB10 unified memory, so a naive
test process OOMs. Running it with
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6
CUDA_MODULE_LOADING=LAZY` lets a small test share the node alongside the live engine:
`sudo docker exec -i -e PYTORCH_CUDA_ALLOC_CONF=... -e CUDA_MODULE_LOADING=LAZY vllm-glm52 python3 - < validate_lse_recombine.py`

## Status / next steps

1. ✅ Harness #1 (LSE recombine) — PASS; foundation proven, no-downtime loop works.
2. Implement Approach 1 (indexer logits all-gather + B2 owned-mask) behind the DCP patch.
3. Harness test 2 (top-k set-equality) — instrument the indexer at a DCP boot.
4. Boot → needle → coherence soak → 1M.
