# Item B — Approach 1 implementation spec (source-anchored, vLLM @ ab66606)

Verified against the live container by the `item-b-dcp-mapping` workflow + adversarial re-derivation.
Mapping CONFIRMED (interleave=1): `owner(p)=p%N`, `local(p)=p//N`; indexer local-logits column `c`
on rank `r` == global position `c*N + r`. Both algorithmic cores validated offline:
`validate_dcp_reassembly.py` (reassembly + owned-mask, ALL PASS) and `validate_lse_recombine.py`
(LSE recombine, PASS).

## Critical correction the workflow surfaced
The indexer is **DCP-unaware**: `decode_metadata.seq_lens` is GLOBAL 2D `(B, next_n)` (MTP next_n=4
on GB10) and `decode_metadata.block_table` is the global-coordinate table, but the per-rank cache is
sized `cdiv(max_model_len, block_size·N)`. So feeding GLOBAL seq_lens to `fp8_fp4_paged_mqa_logits`
**over-reads** beyond the ~S/N owned slots. Item B is therefore TWO coupled edits:
1. **Localize** the seq_lens fed to the logits kernel (expand-then-localize on the 2D tensor —
   naive pre-expansion localize is wrong by up to next_n-1 columns/row).
2. **Globalize the top-k**: all_gather local logits → reassemble to global column order → top-k over
   GLOBAL logits with GLOBAL seq_lens → write owned-local `p//N` (else -1) into topk_indices_buffer.

Localize formula (matches cp_utils): `rounds = g//(N*I); rem = clamp(g%(N*I) - r*I, 0, I);
local = rounds*I + rem` (I = cp_kv_cache_interleave_size = 1).

## Edits (all via glm52 runtime patch, in-container; OPS flashmla_sparse.py unchanged)

| # | File | Anchor | Change |
|---|------|--------|--------|
| 1 | `vllm/v1/attention/backends/mla/indexer.py` | `DeepSeekV32IndexerDecodeMetadata` dataclass | add `dcp_local_seq_lens` (+ `dcp_local_schedule_metadata`) fields |
| 2 | indexer.py | `__init__` (scheduler buffer alloc) | alloc `dcp_scheduler_metadata_buffer`; `assert cp_interleave==1` |
| 3 | indexer.py | `build()` after final GLOBAL 2D `seq_lens` | compute `dcp_local_seq_lens` (expand-then-localize) + local schedule; pass both into the dataclass |
| 4 | `vllm/model_executor/layers/sparse_attn_indexer.py` | module top | `from vllm.distributed.parallel_state import get_dcp_group` |
| 5 | sparse_attn_indexer.py decode block | the `fp8_fp4_paged_mqa_logits(...)` call | feed LOCAL seq_lens + matching schedule_metadata |
| 6 | sparse_attn_indexer.py | after `logits = ...`, before top-k | all_gather + pad-to-`Lmax=(max_model_len+N-1)//N` (−inf) + reassemble `view(num_rows,N,Lmax).transpose(1,2).reshape(num_rows,Lmax*N)` |
| 7 | sparse_attn_indexer.py | the `top_k_per_row_decode` call | run over GLOBAL logits with GLOBAL seq_lens |
| 8 | sparse_attn_indexer.py | after top-k, before unpack | owned→`p//N` else -1 write-back (`(p%N==r)&(p>=0)`) |

Prefill path (sparse_attn_indexer.py L176-256): **no change** — it gathers full per-chunk KV across
shards before its top-k (verify `chunk.skip_kv_gather==False` under DCP).

## Watch items
- **§7 most-likely bug:** reassembly column order / unequal Lmax across ranks → silent garbage.
  Caught by `validate_dcp_reassembly.py` (offline) — keep it green for any reshape change.
- **schedule_metadata (§4):** must match the LOCAL seq_lens for the logits kernel; the GB10 torch/triton
  fallback (`sm12x_deep_gemm_fallbacks.py`) may ignore it — verify; if ignored, no-op for correctness.
- **Perf caveat (not a blocker):** §6 adds one DCP all_gather of `[B·next_n, Lmax]` fp32 per decode
  step on the hot path (~max_model_len/N cols). Measure decode tok/s; if it dominates the ~22 tok/s
  budget at the 512k tier, fall back to M1 D-lite (replicated indexer) there and reserve Approach 1
  for the true-1M tier.

## Boot #3 result (2026-06-28) — DECODE item B WORKS; prefill indexer is the next bug

- ✅ **Decode item B is correct.** With Step A+B, DCP=8+fp8+MTP serves and the coherence test that
  was `"OKOKOK…"` garbage in boot #2 returns `"17 multiplied by 23 equals 391, DONE."` KV pool
  4,800,512 tokens (sharded). The patch applied cleanly on all 8 nodes (E1/E2/E3 + R1/R2 + py_compile).
- ❌ **Prefill indexer crashes at scale.** The 16k needle's PREFILL hit
  `RuntimeError: Triton Error [CUDA]: an illegal memory access` at `sparse_attn_indexer.py:233`
  (the prefill block L176-256, NOT the decode path we patched). This is exactly the §5 caveat
  ("verify `chunk.skip_kv_gather==False` under DCP"): the prefill indexer's KV gather /
  `fp8_fp4_mqa_logits` is not DCP-safe at scale — it reads the per-rank sharded indexer cache with
  global seq_lens/indices → out-of-bounds. Short prompts (tiny prefill) survive; ~16k does not.
- **Next fix:** make the prefill indexer DCP-aware (L176-256). Either confirm/repair the cross-shard
  KV gather (`cp_gather_indexer_k_quant_cache`) so prefill logits are genuinely over the full key
  set, or apply the same localize+all_gather+global-top-k treatment to the prefill `top_k_per_row_prefill`
  path. Read L176-256 + `build_prefill_chunk_metadata` (`skip_kv_gather`, `cu_seq_lens`) first.

## Prefill indexer under DCP — design for the next iteration

Root cause (boot #3, fully traced): `build_prefill_chunk_metadata` (indexer.py L646+) is **DCP-unaware**
— `cu_seq_lens`/`total_seq_lens`/`cu_seqlen_ks/ke` come from the GLOBAL compressed_seq_lens, but the
block_table is the per-rank shard (sized `/N`). So `cp_gather_indexer_k_quant_cache` (a **compiled
C++ op**, `_C_cache_ops`) + `fp8_fp4_mqa_logits` (L233) read global-many keys from a shard → IMA.
Short prompts (tiny prefill) survive; ~16k does not. The decode path (now correct) does NOT share
this code.

This is a substantial parallel to decode item B (a focused next sub-project, not a quick patch):
- **Approach A (recommended): all-gather the indexer K shards for the chunk.** Before the prefill
  logits, NCCL all_gather each rank's sharded indexer-K into a full per-chunk workspace (mirrors the
  main attention's prefill KV all-gather, `mla_attention.py:1527-1536` "additional kvcache allgather
  across the DCP group"). Then the existing gather/`fp8_fp4_mqa_logits`/`top_k_per_row_prefill` run
  over FULL K with GLOBAL cu_seqlens → correct global top-k. The indexer K is the small tensor
  (kv_lora-free index dim), so the gather is affordable even at 1M. Then apply the **owned-mask**
  (p%N==r → p//N else -1) to the prefill `topk_indices` (the decode R2 analog) so the sharded
  main-KV prefill attention (item C chunk path) attends only owned keys and the LSE recombine stitches.
- **Approach B:** localize cu_seq_lens/block_table per shard + all_gather the prefill *logits* — but
  prefill logits are `[chunk_tokens, full_seq]`, far too large to all_gather. Rejected.

Open sub-items to confirm when implementing A: (1) the C++ `cp_gather` writes into a workspace sized
`total_seq_lens` — after the K all-gather, `total_seq_lens`/`cu_seq_lens` must describe the FULL
(gathered) extent, not the shard; (2) the prefill attention (`_forward_fp8_kv_separate_prefill_decode`
chunk path) must also owned-mask its topk + the per-shard LSE recombine (item C/E) must cover the
prefill chunk under CP; (3) `skip_kv_gather` semantics under DCP. Validate with a needle that forces a
real prefill (>2048 tokens) once landed.

## Validation gates (in order)
1. `validate_dcp_reassembly.py` (offline) — PASS ✅
2. `validate_lse_recombine.py` (live container, no downtime) — PASS ✅
3. DCP=8 boot → top-k SET-EQUALITY vs N=1 reference (instrument or logits-capture) — the decisive test
4. needle 128k→512k→1M → coherence soak → go-live
