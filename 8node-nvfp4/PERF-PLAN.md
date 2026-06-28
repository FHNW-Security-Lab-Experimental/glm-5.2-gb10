# DCP 1M throughput optimization plan (prioritized, adversarially verified)

I have everything needed. The existing offline harness `validate_dcp_reassembly.py` already encodes exactly the offline tests Approach 2 and the prefill index-cache need (set-equality of the owned-union, reassembly identity, and prefill ragged reassembly), and `validate_dcp.py` is the boot needle. Here is the prioritized plan.

---

# GLM-5.2-NVFP4 DCP=8 1M throughput — prioritized optimization plan

**Live config verified in-container (`vllm-glm52`):** DCP=8, TP=8, `--max-model-len 1048576`, `--max-num-batched-tokens 4096`, `--max-num-seqs 4`, MTP `k=3` FLASHMLA_SPARSE, `--enforce-eager`, `GPU_MEMORY_UTILIZATION` **launcher default 0.72** (the input that said 0.78 was reading the in-container util; the repo default and the TP refusal guard at `start_glm52_8node.sh:423-427` say <=0.75). 78 layers, `first_k_dense_replace=3` → ~75 sparse-indexer layers/forward, `index_topk=2048`, `compress_ratio=1` (indexer KV uncompressed → full-context K).

**Correction to the inputs, adopted from the verdict:** both kernel changes are *correct and strict wins*, but each ships with a silent-corruption trap. Approach 2 must (a) feed the LOCAL top_k `dcp_local_seq_lens`, not global `seq_lens`, and (b) tie-break the re-top_k deterministically on global position. The prefill cache must build the index in **int64 on the K device**. These are not optional polish — they are the difference between "exact" and "off by one boundary column, silently."

The cluster is **live serving dcp-1m**. Nothing below is applied hot. Each change is gated through the offline harness (`validate_dcp_reassembly.py`, which already contains set-equality + reassembly-identity tests) and a boot needle (`validate_dcp.py`) on the next maintenance cutover, never on `/health`.

---

## Ranked changes (value ÷ risk, highest first)

### 1. PREFILL index/sizes cache across layers — **DO FIRST** (free, large, low risk)

**What:** The reassembly index and the all_gatherv `_sizes` depend only on `chunk.dcp_local_seq_lens_allranks`, which `build_prefill_chunk_metadata` already computes **once per forward**. They are bit-identical across all ~75 sparse layers, yet are rebuilt every layer with a `for q in range(R): int(L[:,q].sum().item())` host-sync loop plus N more `.item()` syncs for `_sizes`.

**Change (file:line):**
- `V/v1/attention/backends/mla/indexer.py:169-184` (dataclass `DeepseekV32IndexerPrefillChunkMetadata`): add two fields next to the existing `dcp_local_*`:
  ```python
  dcp_reasm_idx: "torch.Tensor | None" = None      # int64, on K device
  dcp_gatherv_sizes: "list[int] | None" = None      # host ints
  ```
- `V/v1/attention/backends/mla/indexer.py:727` (right after `dcp_local_seq_lens_allranks = torch.stack(...)`): build both once, in the `if _N > 1:` block, from the CPU `_own(_gc_cpu, rr)` values that are *already materialized here* — so the `.item()` syncs happen **once per forward, not per layer**:
  ```python
  L = dcp_local_seq_lens_allranks.to(torch.int64)          # [N, R] on CPU side already
  prt = L.sum(dim=1)                                        # per-rank totals
  rb = torch.zeros(_N, dtype=torch.int64); rb[1:] = torch.cumsum(prt[:-1], 0)
  ro = torch.zeros((_N, R), dtype=torch.int64); ro[:, 1:] = torch.cumsum(L[:, :-1], 1)
  parts = []
  for q in range(R):
      glen = int(L[:, q].sum().item())
      if glen == 0: continue
      p = torch.arange(glen, dtype=torch.int64); ow = p % _N; lo = p // _N
      parts.append(rb[ow] + ro[ow, q] + lo)
  dcp_reasm_idx = (torch.cat(parts) if parts else torch.zeros(0, dtype=torch.int64)).to(device)  # device=block_table.device, int64 — REQUIRED
  dcp_gatherv_sizes = [int(prt[rr].item()) for rr in range(_N)]
  ```
  and pass both into the dataclass constructor at `indexer.py:782`.
- `V/model_executor/layers/sparse_attn_indexer.py:208-220` (the live `def _glm52c_reasm_idx`) — **delete**; `:245-246` (`_sizes` list-comp) — **delete**. In the per-chunk body (`:247-253`) replace with `_sizes = chunk.dcp_gatherv_sizes` and `_ridx = chunk.dcp_reasm_idx`.

**Ship in repo:** `remote/glm52-gb10/glm52-dcp-patches.sh` — add the two dataclass fields in the item-C `patch_indexer` block; build `reasm_idx`+`sizes` after the `dcp_local_seq_lens_allranks` anchor (~`:286+`); drop the per-layer `_glm52c_reasm_idx`/`_sizes` from `patch_sai` (anchors at `:348`, `:390`). Keep the `# GLM52_DCP_ITEMC` sentinel style.

**Expected effect:** eliminates ~75× redundant Python index builds and the per-layer `(R+N)` CUDA→host syncs. At 1M with heavy sub-chunking (the 256 MB logits cap forces ~64 query rows/sub-chunk → many sub-chunks/layer), that is hundreds-to-thousands of full pipeline drains/forward removed. Plausible **1.3–1.6× prefill wall-clock** at 400k–1M; bounded below by the irreducible `fp8_fp4_mqa_logits` + `top_k_per_row_prefill` compute. **Decode unaffected.**

**Risk: low.** The cached index is provably bit-identical to the per-layer one (same inputs, same arithmetic). The only trap is **int64 + correct device** — an int32 or wrong-device index silently mis-selects K rows. Build with `.to(torch.int64)` and `.to(device)` where `device = block_table.device`.

**Validation gate:** `python remote/glm52-gb10/validate_dcp_reassembly.py` (its `test_prefill_reassembly` / `build_reassembly_index` already assert the ragged shard→global identity for N=8 across multiple `glens`). Then on the next cutover, `validate_dcp.py` boot needle + a known long prefill set-equality vs the pre-change build. Never `/health`.

---

### 2. DECODE Approach 2 (local top-2048 union) — **DO SECOND** (big decode win, medium risk, needs the two correctness fixes)

**What:** Replace the all_gather of FULL padded logits + global `top_k_per_row_decode` over `Lmax*N≈1,048,576` with: per-rank LOCAL top-2048 over the shard (~131k), convert to global positions, all_gather only `[rows, N*topk]=16,384`, re-top_k over 16,384 → exact global top-2048. Mathematically exact: the column partition by `p%N` is disjoint and complete, so the global top-2048 ⊆ union of per-rank top-2048.

**Change (file:line):** `V/model_executor/layers/sparse_attn_indexer.py:402-427` (the `all_gather`+view/transpose/reshape block) and `:430-460` (the family-120-forced global `top_k_per_row_decode`) → replace with the Approach-2 sketch from the input, **with the two required fixes**:

```python
if _glm52_N > 1:
    rows = logits.shape[0]
    # FIX (a): LOCAL top_k MUST use dcp_local_seq_lens (the per-row owned counts the
    # logits kernel already used as _glm52_lsl), NOT global seq_lens — global would
    # over-scan past this rank's owned cols into garbage.
    _loc_lsl = decode_metadata.dcp_local_seq_lens \
        if decode_metadata.dcp_local_seq_lens is not None else seq_lens
    _loc_idx = topk_indices_buffer.new_empty((num_padded_tokens, topk_tokens))
    ops.top_k_per_row_decode(logits, next_n, _loc_lsl, _loc_idx,
                             rows, logits.stride(0), logits.stride(1), topk_tokens)
    _valid = _loc_idx >= 0
    _safe  = torch.where(_valid, _loc_idx, torch.zeros_like(_loc_idx)).long()
    _loc_score = torch.gather(logits, 1, _safe)                       # kernel returns idx, not value
    _loc_score = torch.where(_valid, _loc_score, _loc_score.new_full((), float("-inf")))
    _glob_pos  = torch.where(_valid, _loc_idx.long() * _glm52_N + _glm52_r,
                             torch.full_like(_loc_idx.long(), -1))
    _all_score = _glm52_dcp.all_gather(_loc_score.contiguous(), dim=1)        # [rows, N*topk]=16384
    _all_pos   = _glm52_dcp.all_gather(_glob_pos.contiguous(), dim=1)
    # FIX (b): deterministic tie-break on GLOBAL position. The old full-width kernel breaks
    # score ties by global column order; torch.topk breaks them by buffer index = lowest RANK.
    # fp8_ds_mla KV gives few distinct logit values at long ctx, so boundary ties are REAL.
    # Perturb by -eps*global_pos (eps below fp32 ULP of the score range) so the SET is identical.
    _eps = 1e-6  # tune below score-range ULP; or do a lexicographic (score, -pos) sort
    _rank_key = torch.where(_all_score == float("-inf"), _all_score,
                            _all_score - _eps * _all_pos.to(_all_score.dtype))
    _k = min(topk_tokens, _all_score.shape[1])
    _v, _sel = torch.topk(_rank_key, k=_k, dim=1)
    topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
    topk_indices.fill_(-1)
    _g = torch.gather(_all_pos, 1, _sel)
    _g = torch.where(_v == float("-inf"), torch.full_like(_g, -1), _g)
    topk_indices[:, :_k] = _g.to(topk_indices.dtype)
    # owned-mask UNCHANGED (this is r_r2 / item-B owned writeback)
    _owned = (topk_indices % _glm52_N == _glm52_r) & (topk_indices >= 0)
    _loc   = torch.div(topk_indices, _glm52_N, rounding_mode="floor")
    topk_indices.copy_(torch.where(_owned, _loc, torch.full_like(topk_indices, -1)))
else:
    # existing single-rank persistent_topk / top_k_per_row_decode path (unchanged)
```

Caveat on FIX (b): `-eps*global_pos` with `pos` up to 1M can underflow fp32 precision against large scores; a lexicographic `(score, -pos)` sort (stable secondary key) is the safer form and is what I'd ship — the harness will tell you if the eps form is exact. Keep the LOCAL top_k workspace separate from `topk_indices_buffer` so the in-place writes don't clobber.

**Ship in repo:** `remote/glm52-gb10/glm52-dcp-patches.sh:189-239` (the `r_r1` R1 anchor that injects the all_gather/reassembly) — replace with the Approach-2 body; `:240-251` (`r_r2` owned-mask) — **keep unchanged**.

**Expected effect:** global top_k input 1,048,576 → 16,384 (64×); eliminates the per-step `.view(rows,N,Lmax).transpose(1,2).reshape(...).contiguous()` ~1M-element fresh buffer + non-coalesced transpose; all_gather payload ~16 MB → ~256 KB. The removed work is the only decode stage that scales with **full** context rather than shard context. If stage (iii) is 25–40% of per-step decode latency (plausible: a 1M-wide kernel + 1M transpose vs. an 8×-narrower local logits kernel and sparse top-2048 attention), expect roughly **+15–30% decode (~17–18 → ~21–24 tok/s)**; even at 15% it is a clean measurable gain at zero correctness cost. The added 131k-wide local top_k is over the same shard width the logits kernel already scanned — strictly net-negative work.

**Risk: medium.** Not bandwidth or shape — purely the **cross-rank tie-break** (most-likely-bug per the verdict) and feeding the **wrong seq_lens** to the local top_k. Both are addressed above. Also guard the genuine <2048-valid-cols short case (handled: `-inf`→`-1`, owned-mask drops).

**Validation gate:** `validate_dcp_reassembly.py` `test_owned_union` already asserts the owned-union set-equality for N=8; extend it (or add a sibling) to compare Approach-2's produced top-2048 **set** against a single-rank reference `top_k_per_row_decode` over the same synthetic global logits, **with the identical tie-break applied to both**. Must be bit-identical on construct-tie inputs (inject exact-equal logit values at the boundary). Then boot needle `validate_dcp.py`. Gate go-live on **set-equality**, never `/health` (sparse corrupts silently).

---

### 3. MTP `k=3 → k=5` A/B — **DO THIRD** (only per-stream decode knob, config-only, must measure)

**What:** `NUM_SPECULATIVE_TOKENS=3 → 5` (`start_glm52_8node.sh:109`, emitted into `--speculative-config` at `:369`). The in-checkpoint nextn head is single (`num_nextn_predict_layers=1`), so k>1 reuses it autoregressively; marginal acceptance of tokens 4–5 decays at long context.

**Expected effect:** honestly **+0% to +10%**, and it can **regress** if tokens 4–5 are mostly rejected (extra draft+verify for nothing). This is the only knob that touches single-stream decode tok/s without a kernel change.

**Risk: low/medium** (config-only, instant rollback). **Must A/B**: measure accepted-tokens/step and tok/s at k=3 vs k=5 on a real long-context decode; keep whichever wins. Run **after** Approach 2 lands so you're tuning the optimized decode path.

**Validation gate:** `validate_dcp.py` needle for coherence (sparse + speculative both corrupt silently); accept only if tok/s strictly improves.

---

### 4. PREFILL `--max-num-batched-tokens 4096 → 8192` — **DO FOURTH** (config-only prefill win, soak for OOM)

**What:** `MAX_NUM_BATCHED_TOKENS` (`start_glm52_8node.sh:93`). Chunked prefill is on by default, so this is the prefill **chunk size** — 8192 halves the chunk *count*, halving per-layer collective + (now-cached, post-#1) reassembly overhead. The "4096 warns with 4 slots" is the `max_num_batched_tokens > max_num_seqs*max_model_len` warning, which `4*1048576 ≫ 8192` does NOT trigger — it's safe.

**Expected effect:** roughly **1.3–1.8× faster prefill** vs 4096 standalone, but **the gain is mostly subsumed by change #1** (both attack per-layer chunk overhead). Treat this as incremental after the cache lands, not additive at face value.

**Risk: medium** — peak activation/logits workspace grows with chunk tokens; must boot-soak against the documented **GB10 unified-memory OOM-wedge** (`gb10-tp-unified-memory-oom`: high util/buffers can ping-only-wedge the whole cluster, needs power-cycle). Try 8192 first; 12288/16384 only if boot is stable and prefill latency still matters (diminishing — chunk count already halved by #1).

**Validation gate:** clean boot + a 400k+ prefill without OOM-wedge, then `validate_dcp.py` needle.

---

### 5. `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB 256 → 512` — **OPTIONAL, AFTER #1, soak required**

**What:** `start_glm52_8node.sh:288`. The 256 MB cap (`max_logits_elems=64M`) forces, at N=1M, ~64 query rows/sub-chunk → many sub-chunks. Raising it directly cuts sub-chunk count (the per-layer launch/collective multiplier), which is *more* leverage than #4 because it attacks the sub-chunk count rather than the chunk count.

**Expected effect:** ~1.1–1.3× prefill on top of #1.

**Risk: medium** — transient logits buffer is `M*N*4` bytes; 512 MB ≈ doubled workspace. Same OOM-wedge soak as #4. Land #1 first (it makes the per-sub-chunk reassembly cheap, so the remaining win from fewer sub-chunks is the collective/launch latency only). Trial 512 before touching #4; do not stack 512 *and* 16384 batched-tokens without a full soak.

---

## SKIP (do not change for throughput)

- **CUDA graphs (`PIECEWISE`/`FULL`) — keep `--enforce-eager`.** The DCP decode path issues a runtime `all_gather` + dynamic-shape python reassembly + `torch.where` owned-mask every step; collectives and dynamic shapes inside a captured graph are exactly the fragility `1M-SPARSE-CP-KV-PLAN.md` constraint F warns about. Decode is BW-bound, so the kernel-launch win is small. **High risk, low reward.**
- **`GPU_MEMORY_UTILIZATION` ↑ — keep 0.72.** Decode is memory-**bandwidth**-bound, not KV-pool-bound; a bigger pool gives **zero** tok/s, only capacity. 0.72 is also under the TP OOM-wedge ceiling the launcher hard-refuses above (`:423-427`). Do not raise for throughput.
- **`--max-num-seqs 4 → 6-8` — aggregate only.** Raises concurrent-stream aggregate throughput (pool holds ~4.1× full-1M), **zero** single-stream effect, and slightly raises per-step decode work. Raise only if the real workload is concurrent; keep 4 for pure single-stream 1M.
- **Fusing the two prefill `all_gatherv` into one — do NOT** (per verdict): `k_quant` (fp8 head_dim bytes) and `k_scale` (4 bytes) have different per-row widths; fusing them is not safe without separate validation, and the launch-latency saving is marginal next to #1.

---

## RECOMMENDED IMPLEMENTATION ORDER (lowest-risk-highest-value first)

1. **#1 Prefill index/sizes cache** — free, ~1.3–1.6× prefill, provably bit-identical. Gate: `validate_dcp_reassembly.py` (prefill identity) + boot needle. Ship first; it's pure overhead removal with no memory cost.
2. **#2 Decode Approach 2** (with the dcp_local_seq_lens fix + deterministic tie-break) — ~+15–30% decode, the structural decode win. Gate: extend `validate_dcp_reassembly.py` set-equality vs single-rank reference with identical tie-break, then boot needle. Highest absolute value; medium risk fully mitigated by the two fixes.
3. **#3 MTP k=5 A/B** — config flip on the now-optimized decode path; keep only if tok/s strictly improves. Trivial rollback.
4. **#4 max-num-batched-tokens → 8192**, then **#5 MAX_LOGITS_MB → 512** — both behind a GB10 OOM-wedge soak; expect each to be largely subsumed by #1, so treat as incremental and stop when prefill latency is acceptable.

**Net target:** #1+#2 alone should move decode ~17–18 → ~21–24 tok/s and prefill ~1.3–1.6× with **zero** correctness cost (both gated on set-equality / reassembly-identity, never `/health`). #3–#5 are measure-and-keep tuning on top. The two kernel edits ship via the anchor-guarded `remote/glm52-gb10/glm52-dcp-patches.sh` (item-B `r_r1` for #2; item-C `patch_indexer`/`patch_sai` for #1) and apply in-container at the next cutover — never hot on the live dcp-1m engine.

**Key file:line index:** decode patch site `V/model_executor/layers/sparse_attn_indexer.py:402-427` + `:430-460` (local-seq-lens feed at `:383`); prefill cache build `V/v1/attention/backends/mla/indexer.py:169-184` (dataclass) + `:727`/`:782` (build), delete `V/model_executor/layers/sparse_attn_indexer.py:208-220` + `:245-246`; repo patches `remote/glm52-gb10/glm52-dcp-patches.sh:189-251` (B) + `:273-409` (C); launcher knobs `remote/glm52-gb10/start_glm52_8node.sh:93,109,288`; validators `remote/glm52-gb10/validate_dcp_reassembly.py` (offline) + `remote/glm52-gb10/validate_dcp.py` (boot needle).