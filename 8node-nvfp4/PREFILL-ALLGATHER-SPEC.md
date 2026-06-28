# Prefill indexer DCP all-gather — implementation spec (Approach A, source-anchored)

I now have complete, source-confirmed grounding. I have all the facts to write the precise prefill all-gather implementation spec. Here is the deliverable.

---

# PREFILL DSA INDEXER UNDER DCP — Approach A implementation spec (Step C, glm52-dcp-patches.sh)

Verified live against `vllm-glm52` (vLLM dist `/usr/local/lib/python3.12/dist-packages/vllm`, @ ab66606). The decode item-B patch (R1/R2 + E1/E2/E3 in `glm52-dcp-patches.sh`) is the proven template; this mirrors it for prefill. **Two corrections to the plan as written are flagged inline (CORRECTION-1, CORRECTION-2) — they are load-bearing.**

## 0. Confirmed facts (the ones the spec hangs on)

- **Root cause confirmed.** `build_prefill_chunk_metadata` (`indexer.py:646`) builds `cu_seq_lens`/`total_seq_lens`/`cu_seqlen_ks`/`cu_seqlen_ke` from GLOBAL `compressed_seq_lens` (`indexer.py:508,540-552,670`), but `block_table=block_table[start_idx:end_idx]` (`indexer.py:717`) is the per-rank /N shard. The shard sizing is real: `MultiGroupBlockTable` (`block_table.py:245-247`) sizes rows `cdiv(max_model_len, block_size·total_cp_world_size)`, and `_compute_slot_mapping_kernel` (`block_table.py:357-380`) only writes a token's slot on its owner rank. So `cp_gather_indexer_k_quant_cache(kv_cache, k_quant, k_scale, chunk.block_table(shard), chunk.cu_seq_lens(global))` (`sparse_attn_indexer.py:197-203`) walks global-length key runs through a /N block_table → OOB → IMA at `fp8_fp4_mqa_logits` (`sparse_attn_indexer.py:234`).

- **Write-time interleave (definitive, from the slot-mapping kernel).** With `CP_KV_CACHE_INTERLEAVE_SIZE=1`: `is_local ⟺ (pos % virtual_block_size) % N == rank`, and `virtual_block_size = block_size·N`, so **owner(pos) = pos % N**, and `local_block_offset = (pos % (block_size·N)) // N`, i.e. **local(pos) = pos // N** within the request. This is exactly the decode item-B mapping. The write is at raw `positions` granularity.

- **`compress_ratio = 1` for GLM-5.2** (`indexer.py:330-333`: default 1; only set >1 for DeepseekV4). So compressed key index == token index. **Write the local-len math against `self.compress_ratio` generically — do NOT hardcode 1.**

- **`cu_seqlen_ks/ke` stay GLOBAL.** They index into the (reassembled) global K row space (`indexer.py:763-770`: `ks = seq_start`, `ke = seq_start + (start_pos+1+offset)//COMPRESS_RATIO`). They are correct as-is once K is full global order. **Do not localize them** (this is the §4 gotcha; it is right).

- **No kernel edits.** `fp8_fp4_mqa_logits` + `top_k_per_row_prefill` are correct given full global-order K and global ks/ke.

- **Workspace.** `get_simultaneous(*(shape,dtype))` returns byte-sliced views from one buffer (`workspace.py:92-117`); each call re-slices the same growable arena. The global K workspace today is `_gather_workspace_shapes(total_seq_lens=GLOBAL, ...)` → `(T,128) fp8 + (T,4) uint8`, where `T ≤ max_prefill_buffer_size = max_model_len*40` (`indexer.py:219-231`).

- **`enforce-eager` is on in production** (CLAUDE.md) — the per-chunk all_gather + Python reassembly is acceptable; no CUDA-graph capture concern for prefill.

---

## (4) Getting N / r (do this first; it's referenced everywhere)

In `sparse_attn_indexer.py`, inside the `if has_prefill:` block, before the chunk loop. Use the **identical** guarded accessor the decode R2 patch uses, so prefill and decode read DCP from one source:

```python
# GLM52_DCP_ITEMC  DCP group for the prefill indexer K all-gather.
try:
    from vllm.distributed.parallel_state import get_dcp_group as _glm52_gdg
    _glm52c_dcp = _glm52_gdg()
    _glm52c_N = _glm52c_dcp.world_size
    _glm52c_r = _glm52c_dcp.rank_in_group
except Exception:
    _glm52c_dcp = None; _glm52c_N = 1; _glm52c_r = 0
```

Guard everything new behind `if _glm52c_N > 1:` so the non-DCP production path is byte-identical (the prefill block must be unchanged when N==1).

For the per-chunk **local seq-len** math we need the chunk's per-request COMPRESSED seq lengths. The cleanest source is to **pass them through the chunk metadata** rather than recompute in the layer (the layer does not have `compressed_seq_lens`). So most of the work lands in `indexer.py`.

---

## (2) `indexer.py` — `build_prefill_chunk_metadata` changes

Add a per-rank-LOCAL `cu_seq_lens` (for the local gather) and a LOCAL `total_seq_lens` (for the local dst buffer), **alongside** the existing global ones. Add a field to carry them, plus the per-(req) local lengths for all ranks (needed for ragged reassembly).

### 2a. Add fields to `DeepseekV32IndexerPrefillChunkMetadata` (anchor `indexer.py:169-180`)

```python
    skip_kv_gather: bool = False
    # GLM52_DCP_ITEMC  per-rank-LOCAL gather metadata (None when DCP disabled).
    dcp_local_cu_seq_lens: "torch.Tensor | None" = None   # int32 [num_reqs+1], this rank's owned compressed-key prefix sum
    dcp_local_total_seq_lens: int = 0                       # sum of this rank's owned compressed keys for the chunk
    dcp_local_seq_lens_allranks: "torch.Tensor | None" = None  # int32 [N, num_reqs], per-req owned count on EVERY rank (CPU is fine)
```

### 2b. In `build_prefill_chunk_metadata`, compute the local cu_seq_lens

The function already has `compressed_seq_lens` (global, device) and `compressed_seq_lens_cpu` (global, CPU). Add the DCP-local derivation, gated on N>1. Insert after the existing global `cu_seq_lens` cumsum (`indexer.py:689-691`):

```python
    # GLM52_DCP_ITEMC  build per-rank-local cu_seq_lens for the sharded gather.
    dcp_local_cu_seq_lens = None
    dcp_local_total_seq_lens = 0
    dcp_local_seq_lens_allranks = None
    try:
        from vllm.distributed.parallel_state import get_dcp_group as _gdg
        _N = _gdg().world_size; _r = _gdg().rank_in_group
    except Exception:
        _N = 1; _r = 0
    if _N > 1:
        _I = 1  # cp_kv_cache_interleave_size; assert==1 enforced in builder __init__
        # per-request GLOBAL compressed lengths for this chunk (device + cpu)
        _g = compressed_seq_lens[start_idx:end_idx]                       # device int
        _g_cpu = compressed_seq_lens_cpu[start_idx:end_idx]               # cpu int
        # round-robin owned count for THIS rank (cp_utils formula, owner=p%N):
        def _local_len(gt, rank):
            rounds = gt // (_N * _I)
            rem = torch.clamp((gt % (_N * _I)) - rank * _I, min=0, max=_I)
            return rounds * _I + rem
        _local_g = _local_len(_g, _r)                                     # device int [num_reqs]
        dcp_local_cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
        dcp_local_cu_seq_lens[:1] = 0
        torch.cumsum(_local_g, dim=0, out=dcp_local_cu_seq_lens[1:])
        dcp_local_total_seq_lens = int(sum(
            int(_local_len(_g_cpu, rr).sum().item()) for rr in range(_N)  # need only THIS rank's, but…
        )) if False else int(_local_len(_g_cpu, _r).sum().item())
        # per-(rank,req) owned counts on ALL ranks for ragged reassembly (CPU; tiny: N×num_reqs):
        dcp_local_seq_lens_allranks = torch.stack(
            [_local_len(_g_cpu, rr) for rr in range(_N)], dim=0
        ).to(torch.int32)   # [N, num_reqs] on CPU
```

Then add these to the returned dataclass (`indexer.py:716-720`):

```python
        dcp_local_cu_seq_lens=dcp_local_cu_seq_lens,
        dcp_local_total_seq_lens=dcp_local_total_seq_lens,
        dcp_local_seq_lens_allranks=dcp_local_seq_lens_allranks,
```

### 2c. `__init__` guard (anchor the scheduler buffer alloc, `indexer.py:323-326`)

Add the interleave assertion next to the existing buffers (mirrors decode E2's `assert _I == 1`):

```python
        # GLM52_DCP_ITEMC  prefill K all-gather requires interleave==1 (owner=p%N math).
        if get_total_cp_world_size() > 1:
            assert self.vllm_config.parallel_config.cp_kv_cache_interleave_size == 1, \
                "GLM52_DCP item C requires cp_kv_cache_interleave_size==1"
```

> **CORRECTION-1 (vs the plan's "compute local cu_seq_lens in the layer"):** the plan suggested calling `prepare_dcp_local_seq_lens` from the layer. That kernel writes a fixed `[max_num_reqs]` persistent buffer and operates on the *global decode 2D seq_lens*, not on the per-chunk compressed slice. For prefill you must derive locals from the chunk's `compressed_seq_lens[start:end]` (compressed units!) inside `build_prefill_chunk_metadata`, where that tensor exists. Doing it in the layer would require re-plumbing `compressed_seq_lens` into the layer and re-slicing per chunk — strictly worse. Keep the local-len math in the builder.

### 2d. workspace sizing

`get_max_prefill_buffer_size` already bounds the GLOBAL chunk to `max_model_len*40` rows, so **the existing global K workspace (buffer #2) is reusable unchanged**. You only add a *smaller* local staging buffer (#1), sized to a padded local max. No change to `get_max_prefill_buffer_size` is required.

---

## (1) `sparse_attn_indexer.py` — prefill loop edits

Replace the per-chunk gather + cast block (`sparse_attn_indexer.py:191-222`). Keep the workspace fetch (`k_quant_full, k_scale_full = workspace_manager.get_simultaneous(values_spec, scales_spec)`) — that's buffer #2 (GLOBAL). Add a **second** `get_simultaneous` call for the local staging buffer #1.

### 1a. Allocate the local gather staging buffer (once, before the loop)

```python
        # GLM52_DCP_ITEMC  local staging buffer for the per-rank sharded gather.
        # Sized to the padded local max (ceil(global_total/N)) so all-gatherv has a
        # common dst capacity; only [:local_total] rows are valid per chunk.
        if _glm52c_N > 1:
            _glm52c_local_cap = (total_seq_lens + _glm52c_N - 1) // _glm52c_N
            lvals_spec, lscales_spec = _gather_workspace_shapes(
                _glm52c_local_cap, head_dim, fp8_dtype, use_fp4_cache
            )
            k_quant_local_full, k_scale_local_full = workspace_manager.get_simultaneous(
                lvals_spec, lscales_spec,
            )
```

### 1b. Per-chunk: local gather → all_gatherv → reassemble to GLOBAL order

Replace the `if not chunk.skip_kv_gather:` gather block. **The global buffers `k_quant`/`k_scale` (sliced to `chunk.total_seq_lens`) remain the inputs to the logits kernel** — we just fill them via gather+allgather+reassembly instead of the direct C++ gather:

```python
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                if _glm52c_N <= 1:
                    # unchanged production path
                    ops.cp_gather_indexer_k_quant_cache(
                        kv_cache, k_quant, k_scale,
                        chunk.block_table, chunk.cu_seq_lens,
                    )
                else:
                    # GLM52_DCP_ITEMC  (1) gather THIS rank's shard with LOCAL cu_seq_lens
                    _lt = chunk.dcp_local_total_seq_lens
                    _kq_loc = k_quant_local_full[: _lt]
                    _ks_loc = k_scale_local_full[: _lt]
                    if _lt > 0:
                        ops.cp_gather_indexer_k_quant_cache(
                            kv_cache, _kq_loc, _ks_loc,
                            chunk.block_table,            # per-rank shard (matches local cu_seq_lens)
                            chunk.dcp_local_cu_seq_lens,  # LOCAL prefix sum (NOT chunk.cu_seq_lens)
                        )
                    # (2) ragged all_gatherv across the DCP group (per-rank local_total varies by ±1/req)
                    _sizes = [int(chunk.dcp_local_seq_lens_allranks[rr].sum().item())
                              for rr in range(_glm52c_N)]
                    _kq_all = _glm52c_dcp.all_gatherv(_kq_loc, dim=0, sizes=_sizes)   # [sum_sizes, 128] fp8
                    _ks_all = _glm52c_dcp.all_gatherv(_ks_loc, dim=0, sizes=_sizes)   # [sum_sizes, 4]  uint8
                    # (3) reassemble shard-concat order -> GLOBAL per-request-concat key order.
                    _idx = _glm52c_build_reassembly_index(
                        chunk.dcp_local_seq_lens_allranks,   # [N, num_reqs] CPU
                        _glm52c_N, device=k_quant.device,
                    )                                        # int64 [chunk.total_seq_lens]
                    torch.index_select(_kq_all, 0, _idx, out=k_quant)
                    torch.index_select(_ks_all, 0, _idx, out=k_scale)
            # else: skip_kv_gather -> reuse k_quant/k_scale from the prior chunk (see CORRECTION-2)
```

`cu_seqlen_ks/ke` and the logits + top-k calls (`sparse_attn_indexer.py:234-255`) are **unchanged** — they already use the GLOBAL `chunk.cu_seqlen_ks/ke` and now operate over a genuinely global-order K.

### 1c. The reassembly index builder (module-level helper, near `_gather_workspace_shapes`)

This is the ragged shard→global permutation. **`all_gatherv` concatenates rank-major** (`[rank0's _sizes[0] rows | rank1's _sizes[1] rows | …]`), and within each rank's block the rows are per-request-concat in ascending owned-key order. We want global row `g` (request `q`, global key `p`) ← rank `(p%N)`'s block, that rank's request-`q` offset, plus `(p//N)`:

```python
def _glm52c_build_reassembly_index(local_lens_allranks, N, device):
    # local_lens_allranks: int32 [N, num_reqs] (CPU), owned compressed-key count per (rank,req).
    # Returns int64 [total] permutation s.t. K_global[g] = K_all[idx[g]], with
    # global order = per-request-concat, keys ascending; owner(p)=p%N, local(p)=p//N.
    L = local_lens_allranks  # [N, R]
    Rn = L.shape[1]
    # base offset of each rank's block in the all_gatherv output:
    per_rank_total = L.sum(dim=1)                  # [N]
    rank_base = torch.zeros(N, dtype=torch.int64)
    rank_base[1:] = torch.cumsum(per_rank_total[:-1], dim=0)
    # within-rank per-request offset:
    req_off = torch.zeros((N, Rn), dtype=torch.int64)
    req_off[:, 1:] = torch.cumsum(L[:, :-1].to(torch.int64), dim=1)
    idx_parts = []
    for q in range(Rn):
        # global length of request q = sum over ranks of owned counts
        glen = int(L[:, q].sum().item())
        if glen == 0:
            continue
        p = torch.arange(glen, dtype=torch.int64)  # global key 0..glen-1
        owner = p % N
        local = p // N
        src = rank_base[owner] + req_off[owner, q] + local
        idx_parts.append(src)
    return torch.cat(idx_parts).to(device)
```

This is the CUDA-graph-friendly `index_select` form of `reorg_kvcache` (`mla_attention.py:1879-1949`), specialized to interleave=1. It is the same algorithm `validate_dcp_reassembly.py` already validates for decode — reuse that harness with the prefill row layout.

> **CORRECTION-2 (skip_kv_gather — the rank-divergence hazard the plan flagged but under-specified):** `build` sets `skip_kv_gather = (query_slice.start > 0)` (`indexer.py:551,705`) for chunked-query sub-chunks, which then **reuse** the prior chunk's K from the *shared* `k_quant_full`/`k_scale_full` buffers. Under DCP this is safe *only because* the reassembled global K is written into those same global buffers (1b), so the reuse path reads valid global K with no collective — and since `skip_kv_gather` is derived from rank-uniform global metadata, all 8 ranks take the identical branch every chunk (no NCCL divergence). **Invariant to preserve:** do NOT clear/resize `k_quant_full`/`k_scale_full` between sub-chunks, and do NOT issue the all_gatherv on a `skip_kv_gather` chunk. The `if not chunk.skip_kv_gather:` guard already enforces this — just don't move the collective outside it.

---

## (3) Prefill ATTENTION (`flashmla_sparse.py`) — owned-mask: where, and why nothing new is needed there

**The prefill attention needs an owned-mask, but it is applied in `sparse_attn_indexer.py` (the indexer output), NOT in `flashmla_sparse.py`.** Concretely:

- The indexer must emit **owned-local** top-k positions, identical to decode R2. After `top_k_per_row_prefill` writes GLOBAL positions into `topk_indices_buffer[token_start:token_end, :topk_tokens]`, add the owned-mask **inside the prefill loop, per chunk** (the decode R2 transform):

```python
            # GLM52_DCP_ITEMC  prefill topk are GLOBAL positions; keep owned -> local p//N else -1.
            if _glm52c_N > 1:
                _ti = topk_indices  # the chunk slice already written by top_k_per_row_prefill
                _owned = (_ti % _glm52c_N == _glm52c_r) & (_ti >= 0)
                _loc = torch.div(_ti, _glm52c_N, rounding_mode="floor")
                _ti.copy_(torch.where(_owned, _loc, torch.full_like(_ti, -1)))
```

  Place this immediately after the `ops.top_k_per_row_prefill(...)` call (`sparse_attn_indexer.py:246-255`), writing back into the same `topk_indices` view.

- Why `flashmla_sparse.py` is **unchanged**: in `_forward_fp8_kv_separate_prefill_decode`, the chunk path calls `triton_convert_req_index_to_global_index(req_id_per_token, attn_metadata.block_table, topk_indices, …)` (`flashmla_sparse.py:675-685`). `attn_metadata.block_table` is the **per-rank shard**, and the `-1` sentinel + `return_valid_counts` already make that kernel drop unowned/invalid indices. So feeding it owned-local `p//N` (else -1) is exactly what it expects — the same contract decode relies on. The per-chunk BF16 prefill kernel (`_bf16_flash_mla_kernel`, `flashmla_sparse.py:904-943`) already returns the per-shard base-e LSE (`result[2]`), and `forward_mqa` already threads `return_lse=True` under DCP (`flashmla_sparse.py:975-987`) into `mla_attention.py`'s `cp_lse_ag_out_rs`/`dcp_a2a_lse_reduce`. **Item C/E covers prefill chunks already** — `lse_full[chunk.tokens_slice] = chunk_lse` (`flashmla_sparse.py:777-786`). No prefill-attention edit is needed.

  Caveat to verify, not patch: confirm the chunk path's workspace-offset mapping (`prefill_workspace_starts`, `cp_gather_and_upconvert_fp8_kv_cache`, `flashmla_sparse.py:762-771`) gathers the **owned** main-KV for the chunk consistently with the owned-local topk. It should — both index the same shard block_table — but assert it in the soak (a mismatch is silent garbage, not a crash).

---

## (5) The single most likely bug + a check

**Most likely bug: ragged `all_gatherv` size/order mismatch in the reassembly index — silent garbage, not a crash.** Two coupled ways it goes wrong:

1. **Ragged sizes.** `local_total_r` differs by ±1 per request whenever `L % N != 0`. If you use the equal-shape `all_gather` (`parallel_state.py:636`) instead of `all_gatherv` (`parallel_state.py:657`, which takes explicit `sizes=`), or if `_sizes` is computed from the wrong (global, or this-rank-only) lengths, the gathered blocks are misaligned and `rank_base` is wrong → every request after the first reads the wrong rank's keys. **You MUST use `all_gatherv` with `sizes` = `[local_total_r for r in 0..N-1]` derived from `dcp_local_seq_lens_allranks`.** (This is the one mandatory fix from the verdict — do not `all_gather` the ragged tensors directly.)

2. **Order.** Causality holds **iff** the reassembled global K is in ascending per-request key order (key 0..L-1), because `ke = seq_start + (pos+1)//ratio` assumes that. `owner=p%N / local=p//N` produces exactly that order; a rank-major (block) reassembly would silently mask the wrong keys.

**The check (decisive, offline + online):**

- **Offline:** extend `validate_dcp_reassembly.py` with the prefill row layout — synthesize per-(rank,req) ragged owned counts for `L % N ∈ {0,…,7}`, build `_kq_all` with known sentinel values encoding `(req, global_key)`, run `_glm52c_build_reassembly_index` + `index_select`, and assert the result equals the single-node per-request-concat global K row order **and** that `k_scale` carries the identical permutation. This is the §7-equivalent gate; keep it green for any reshape change.

- **Online (the real proof):** on a 2-node toy DCP boot, capture `k_quant` (the reassembled global buffer) for a small `>2048`-token prompt and `torch.equal` it against an `N=1` reference gather of the same prompt's indexer cache, **before** the logits kernel. If those K buffers match row-for-row (values **and** scales), the IMA is gone *and* the top-k is provably over the true global key set. Then the top-k SET-EQUALITY vs N=1 (validation #3 in the plan) is the go-live gate; never gate on `/health` (item-A/C alone keep `/health` 200 while selecting wrong keys).

**Secondary risk to assert (the verdict's "new vs template"):** the **scale** all_gatherv. The main-attn template asserts `DCP not support scaled kvcache` and never gathers a scale; the indexer path carries `k_scale (T,4) uint8` consumed by `fp8_fp4_mqa_logits` (`k_scale.view(torch.float32).squeeze(-1)`, `sparse_attn_indexer.py:222`). You must `all_gatherv` + `index_select` `k_scale` with the **identical** `_sizes` and `_idx` as `k_quant` (done in 1b). An asymmetry here is silent wrong logits, not a crash — the offline check above must compare scales too.

---

## Implementation checklist for `glm52-dcp-patches.sh` Step C

1. `indexer.py`: anchor `skip_kv_gather: bool = False` → add 3 dataclass fields (2a); anchor the `__init__` scheduler-buffer block → add interleave assert (2c); anchor `build_prefill_chunk_metadata`'s global `cu_seq_lens` cumsum + the `DeepseekV32IndexerPrefillChunkMetadata(...)` return → insert local-len math + pass fields (2b). Sentinel `# GLM52_DCP_ITEMC`, anchor-guarded, abort if anchors absent (mirror item-B's FATAL-on-mismatch).
2. `sparse_attn_indexer.py`: add module-level `_glm52c_build_reassembly_index` (1c); inside `if has_prefill:` add the N/r accessor (4) + local staging buffer (1a); replace the gather block (1b); add the prefill owned-mask after `top_k_per_row_prefill` (3). Gate all on `_glm52c_N > 1`; N==1 path byte-identical.
3. `py_compile` both files (item-B already does this); abort on failure.
4. Gates: `validate_dcp_reassembly.py` (extended for prefill, incl. scales) → 2-node K-equality capture → DCP=8 top-k SET-EQUALITY vs N=1 → needle 128k→512k→1M coherence soak.

**Files (absolute):**
- Live source read: container `vllm-glm52` at `/usr/local/lib/python3.12/dist-packages/vllm/{model_executor/layers/sparse_attn_indexer.py, v1/attention/backends/mla/indexer.py, model_executor/layers/attention/mla_attention.py, v1/worker/cp_utils.py, v1/worker/block_table.py, distributed/parallel_state.py, _custom_ops.py}`
- Patch target: `/home/blacksheeep/SourceCode/dgx-sparks-llm/remote/glm52-gb10/glm52-dcp-patches.sh` (add Step C after item B, line ~263)
- Kernel (no edit needed, confirmed): `/home/blacksheeep/SourceCode/dgx-sparks-llm/remote/glm52-gb10/kernels/flashmla_sparse.py`
- Validation: `/home/blacksheeep/SourceCode/dgx-sparks-llm/remote/glm52-gb10/validate_dcp_reassembly.py` (extend for prefill+scales)
- Spec context already on disk: `/home/blacksheeep/SourceCode/dgx-sparks-llm/remote/glm52-gb10/ITEM-B-SPEC.md` (lines 64-90 are the prior design sketch this supersedes)