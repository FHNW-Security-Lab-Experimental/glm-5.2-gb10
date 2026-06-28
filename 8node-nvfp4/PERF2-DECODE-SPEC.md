# PERF#2 — DCP decode local-top-2048 all-gather (Approach 2)

**Status: GO — boot-tested live 2026-06-28, +12% decode (5.8→6.5 tok/s @ 4k–90k),
correctness identical (needle 32k/131k, deterministic A/B). See PERF-RESULTS.md.** Design + 4-lens adversarial
verification (math brute-forced 4000 cases); blocker + tie-break major folded into the fixes
below. Implemented in `glm52-dcp-patches.sh` (now the production DCP patch; the original item-B
decode is preserved as `glm52-dcp-patches-itemb.sh`). Pre-boot gates all green:
- offline correctness gate `validate_perf2_decode.py` — **ALL PASS** (set-equality + order +
  rank-invariance vs brute-force global top-K, all edge cases);
- adversarial review of the written patch — codegen/control-flow/collective-safety/MTP/dtype
  all clean; **fixed-size all-gather** (`[rows,8×2048]`) so it cannot hit PERF#1's
  size-mismatch deadlock;
- anchors verified against the **live container source** (`a_r3`/`a_r2` match verbatim).

## What it does

Today (item-B "Approach 1"), every sparse-indexer layer per decode step all-gathers the
**full per-key indexer logits** across the 8 DCP ranks (~`[rows, total_kv_len]`, up to
`[rows, 1,048,576]` ≈ 16 MB/layer at 1M), reassembles them to global key order, runs a
global top-2048, then owned-masks. That all-gather scales with context length and is the
decode bottleneck under DCP.

**Approach 2:** each rank computes its **local** top-2048 over its own KV shard, then
all-gathers only those candidates — a **fixed `[rows, 8×2048]` ≈ 256 KB/layer, independent
of context length**. Union the 8×2048 candidates, take the global top-2048 from the union,
owned-mask exactly as today.

Expected: materially faster decode (the workflow estimate was ~+15–30%; the all-gather
volume drop is ~16 MB → 256 KB per layer at 1M, ×~75 layers per step).

## Why it is EXACT (not an approximation)

Ownership is interleave-1: global positions partition disjointly by `owner(p)=p%N`; rank `r`
scores exactly its owned columns, so its logit for any owned `p` is bit-identical to the
global kernel's. The global top-2048 set `G` is downward-closed in score, so for any rank
`r`, `G ∩ owned(r)` has ≤2048 elements and each is within rank `r`'s **own** top-2048.
Therefore `G ⊆ ∪_r local-top-2048(r)`, and the top-2048 of the union **equals** `G`.
Edge case (rank owns <2048 keys, e.g. early decode): the kernel emits `-1`/`-inf` in spare
lanes, which never win. Brute-forced in `scratchpad/refute_perf2.py` (K≤2048, N∈{2,4,8},
dense ties) — all structural cases PASS.

## The fixes (folded in from the adversarial pass)

### FIX 1 — BLOCKER: guard/remove the stock decode top-k under DCP
The stock decode top-k (`ops.top_k_per_row_decode` over `logits`, container
`sparse_attn_indexer.py` ~L337–366, the post-A4 reroute block) runs **between** the item-B
R1 and R2 anchors and is covered by neither. Approach 2 deletes the reassembly, so `logits`
stays the **local** `[rows, Lloc]` shard. If the stock call still fires it runs
`top_k_per_row_decode` over the **local** logits with the **global** `seq_lens` and
**overwrites** `topk_indices` with over-scan garbage → silent corruption (the item-B
word-salad failure mode).
**Required:** add a THIRD anchored edit that wraps the stock block in `if _glm52_N <= 1:`
(single-rank only); the global top-k lives entirely inside the `_glm52_N > 1` Approach-2
block. The new anchor must match the **post-A4** text (`glm52-sparse-patches.sh` A4 runs
before `glm52-dcp-patches.sh`) — verify ordering.

### FIX 2 — MAJOR: ship only the lexicographic (score, −pos) tie-break
Use a strict total order `(score desc, global_pos asc)` for the union re-top-k so every rank
picks the **bit-identical set**. Implement as an explicit two-key stable sort/argsort —
**delete the `score − eps*pos` perturbation path entirely**: at `pos~1M` and large scores,
`eps*pos` underflows the fp32 ULP of the score and silently re-ties.

### FIX 3 — MAJOR: validate SET-equality + rank-invariance, not "bit-identical to Approach 1"
The "bit-identical to the stock global top-k" claim is unverifiable on this box (the stock
kernel's tie-break is undocumented, in-container CUDA). Gate on:
1. SET-equality of `{global positions}` between Approach-2's union path and a
   **replicated-reassembly Approach-1 reference**, on non-tied inputs.
2. **Rank-invariance** on tied inputs: every rank computes the identical set under
   `(score,−pos)`.
3. (Optional, to *claim* bit-identity) empirically pin the stock kernel's tie-break
   in-container with exact-equal logits straddling the 2048 boundary.
This is a comparator test (`validate_dcp_reassembly.py`), **never `/health`**.

## Concrete edits (`glm52-dcp-patches.sh`)

- **E1/E2/E3 (indexer.py, a1/a2/a3)** — UNCHANGED. Approach 2 reuses the local
  `dcp_local_seq_lens` + `dcp_local_schedule_metadata` fields they add.
- **R1 body (`r_r1`)** — KEEP the local `_glm52_lsl`/`_glm52_lsched` selection + local
  `fp8_fp4_paged_mqa_logits` call (lines ~190–212) and the `get_dcp_group`/`_glm52_N`/
  `_glm52_r` block. REPLACE the reassembly tail (the pad → all_gather → view → transpose →
  reshape → `topk_indices = ...`) with, inside `if _glm52_N > 1:`:
  1. `_loc_lsl = dcp_local_seq_lens` (fallback `seq_lens`)
  2. `_loc_idx = topk_indices_buffer.new_empty((num_padded_tokens, topk_tokens))` — SEPARATE workspace (do not clobber the canonical buffer)
  3. `ops.top_k_per_row_decode(logits, next_n, _loc_lsl, _loc_idx, logits.shape[0], logits.stride(0), logits.stride(1), topk_tokens)` — LOCAL pass
  4. `_valid = _loc_idx >= 0; _safe = where(_valid,_loc_idx,0).long(); _loc_score = gather(logits,1,_safe); _loc_score = where(_valid,_loc_score,-inf)`
  5. `_glob_pos = where(_valid, _loc_idx.long()*_glm52_N + _glm52_r, -1)`
  6. `_all_score = _glm52_dcp.all_gather(_loc_score.contiguous(), dim=1)`; `_all_pos = _glm52_dcp.all_gather(_glob_pos.contiguous(), dim=1)` — each `[rows, N*topk]=[rows,16384]`
  7. deterministic re-top-k on `(score desc, global_pos asc)` → `_v, _sel` (two-key sort/argsort; **no eps**)
  8. `topk_indices = topk_indices_buffer[:num_padded_tokens,:topk_tokens]; topk_indices.fill_(-1); _g = gather(_all_pos,1,_sel); _g = where(_v==-inf,-1,_g); topk_indices[:, :_k] = _g`
- **NEW third anchor** — FIX 1: guard the stock decode top-k block under `if _glm52_N <= 1:`.
- **R2 body (`r_r2`)** — UNCHANGED byte-for-byte (`(topk_indices % N == r) & (>=0) → p//N else -1`).
- **`validate_dcp_reassembly.py`** — FIX 3 gate: union path vs replicated-reassembly reference, identical `(score,−pos)` order, injected boundary ties + a rank-owns-<K case.

## Open items to confirm at implementation (source reads)
- Exact location of the stock global `top_k_per_row_decode` relative to the R1/R2 anchors in
  the **post-A4** `sparse_attn_indexer.py` (decide third-anchor text).
- `top_k_per_row_decode` arg order on the GB10 image (match the A4 call site).
- `next_n` (MTP draft width) local vs global in the local pass.
- indexer logits dtype (fp32 assumed) for the ULP/tie-break analysis.

## Why this is safer than PERF#1 (which deadlocked)
PERF#1 hung the 256k prefill in a **variable-size** all-gatherv (size mismatch across ranks
→ stuck NCCL collective at 96% GPU). PERF#2's all-gather is **fixed-size** (`[rows,16384]`,
context-independent) on every rank — no per-rank size divergence, the failure mode that
killed #1 cannot occur here. It still touches a distributed collective, so test on the
hardened launcher with the same instrumentation (capture + crash/hang poller) and the
needle + A/B decode-equivalence gate before promoting.

## Rollback (now that perf2 is production)
perf2 is folded into the production `glm52-dcp-patches.sh` (combined `c45be5db`, deployed to
all 8 nodes). The original item-B decode is preserved as `glm52-dcp-patches-itemb.sh` (combined
`glm52-sparse-patches-dcp-itemb.sh`, sha 979cb445, staged on every node). To roll back: relaunch
with `PATCH_SCRIPT=~/vllm-glm52/runtime/glm52-sparse-patches-dcp-itemb.sh`. `~/glm-triton-dcp`
kernels are shared and unchanged.
