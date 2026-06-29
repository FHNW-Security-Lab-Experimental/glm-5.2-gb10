# PERF optimization results (DCP 1M prefill + decode tok/s)

Outcomes of the `PERF-PLAN.md` items. **PERF#2 is now the production DCP decode path**
(`glm52-dcp-patches.sh`, combined patch `c45be5db`, staged on all 8 nodes); the original
item-B decode is preserved as the one-file rollback `glm52-dcp-patches-itemb.sh` (combined
`glm52-sparse-patches-dcp-itemb.sh`, sha `979cb445`, also on every node). `~/glm-triton-dcp`
kernels are unchanged.

## Launcher hardening (DONE — committed `glm52-1m-perf`)

Two boot failures during back-to-back config swaps on 2026-06-28 were traced to the
**launcher**, not engine config, and fixed in `start_glm52_8node.sh`:
- **Cleanup race** — `docker rm -f` returns before the driver frees the dead container's
  unified memory; loading the new 465 GB model while ~70 GB was still held made
  `WorkerProc initialization failed`. Fix: `create_container` waits (≤90 s) for this node's
  used memory to fall below ~16 GB before starting the new container.
- **Log rotation** — the old glob `glm52-rank*.log` re-rotated already-stamped backups every
  boot, growing the name until `mv` failed "File name too long" and (under `set -e`) aborted
  the launch, losing the crash log. Fix: rotate only the live `rank<N>.log`, tolerant `mv`,
  prune to newest 12.

These made the rest of the testing reliable (clean teardown clears a wedged collective; crash
logs survive). Note: container logs are **UTC**, host is **CEST (+2)** — reconcile timestamps.

## PERF#1 — prefill reassembly-index cache: **NO-GO** (reproducible hang)

**Idea:** build the prefill all-gatherv reassembly index + per-rank gatherv sizes ONCE per
step instead of per-layer (×~75). Offline reassembly math validated; appeared bit-identical.

**Result:** reproducibly **deadlocks the multi-chunk prefill**. Measured 2026-06-28 at 256k
(`max_num_seqs=4`, util 0.78, hardened launcher, clean boot):
- Small/single-chunk requests: fine (healthy, coherent `391 DONE`).
- 256k needle (≈64 prefill chunks of 4096): **hang**. GPU pinned at **96%** for 20+ min,
  `vllm:prompt_tokens_total` frozen, zero scheduler progress, a queued tiny request times out.
- Per-rank evidence: **workers (rank1/2) went silent right after JIT-compiling
  `_fp8_paged_mqa_logits_rowwise_kernel`** (the DSA indexer logits kernel); the **head
  EngineCore spun logging "No available shared memory broadcast block found in 60 seconds"**
  (workers stopped consuming work). Classic stuck cross-rank collective in the indexer
  all-gather path — exactly the code PERF#1 modified.

**Root cause (hypothesis):** the cached **variable-size** gatherv sizes / reassembly index do
not match the per-chunk reality across ranks on a multi-chunk prefill → `all_gatherv` size
mismatch → NCCL deadlock. The offline validator checked reassembly arithmetic but not
cross-rank gatherv-size agreement under real chunked sharding. The baseline (working
`dcp-1m`) completed the same 256k in **754 s**, so 256k is not inherently a hang.

**This explains the earlier "crash":** the first #1 boot did not crash — it hung the first
long request the same way; health flapped and the log was lost to the (now-fixed) rotation bug.

**If revisited:** build the index/sizes **per prefill chunk** (not per request/step), and add
a cross-rank gatherv-size-agreement assertion to `validate_dcp_reassembly.py` before any boot.
Code is on branch `glm52-1m-perf` (separate patch file; working patch untouched). Given the
gain was modest and long DCP prefill is impractical regardless, prefer PERF#2 first.

## PERF#2 — decode local-top-2048 all-gather: **GO — +12% decode, validated live**

Implemented in `glm52-dcp-patches.sh` (production; item-B rollback = `glm52-dcp-patches-itemb.sh`),
all gates green, **boot-tested on 8×GB10 at dcp-1m 2026-06-28**. Spec:
**[`PERF2-DECODE-SPEC.md`](PERF2-DECODE-SPEC.md)**.

**Correctness:** offline gate `validate_perf2_decode.py` ALL PASS (union-top-k == brute-force
global top-K set, identical order, rank-invariant); live A/B vs working at temp=0 — math +
needle (32k AND 131k) **identical**; free-form generation differs only as a paraphrase that
**also varies run-to-run on perf2 itself** (GPU attention float-nondeterminism, present in
both configs — not a regression).

**Decode speedup (single-stream, eager, GB10, 2026-06-28):**

Real-token decode (counting `completion_tokens`; MTP packs ~2.7–3.0 tokens/SSE-chunk, so an
earlier chunk-rate measurement under-read by ~2.8× — the cluster was always ~16–18 tok/s):

| context | working dcp-1m | perf2 | speedup |
|---|---|---|---|
| 4k   | 15.7 tok/s | **17.7** | **+12.7%** |
| 90k  | 16.5 tok/s | (~18.6)  | ~+13% |
| 256k (1M proxy) | 16.9 tok/s | **19.0** | **+12.4%** |

Flat ~+12% from 4k→256k (perf2's local top-k grows 512→32k over that span with no erosion of
the advantage → expected to hold at true 1M; true-1M prefill ~30–40 min/run made a direct 1M
measurement impractical). The win is because item-B always pads to `Lmax=131072` and runs the global
top-k over the **full 1M-wide** logits *every decode step regardless of actual prompt length*;
perf2 replaces that fixed tax with a sharded local top-k + a 16k union. At true 1M the win
should hold or grow (item-B stays 1M-wide; perf2 stays ~16k). No hang — the all-gather is
**fixed-size** (`[rows,8×2048]`), so PERF#1's variable-gatherv deadlock cannot occur.

**Promoted to production (2026-06-28):** perf2's STEP B is folded into `glm52-dcp-patches.sh`
(item-B preserved as `glm52-dcp-patches-itemb.sh`); the production combined patch was rebuilt
(`c45be5db`) and deployed to all 8 nodes, so `start_glm52_config.sh {dcp-1m|dcp-512k}` now uses
the perf2 decode (dcp-512k inherits the gain). Rollback = repoint `PATCH_SCRIPT` at
`glm52-sparse-patches-dcp-itemb.sh` (staged on every node) and relaunch. Bigger decode wins
beyond this would need the per-step compute (MoE/MLA/MTP in eager) addressed — cudagraph (hard
on GB10) or MTP-k tuning (#3).

## PERF#2-for-prefill — candidate-union prefill: **NO-GO** (boot-tested 2026-06-29)

Ported the decode candidate-union to prefill (each rank runs the prefill logits + top-k over
its LOCAL shard-K, fixed-size all-gathers candidates, unions → exact global top-2048). Full
de-risking done (offline gate incl. localization formula GREEN; dry-run + py_compile;
adversarial review GO-WITH-FIXES, fixes folded). Boot-tested at dcp-1m:
- **Correct** — needle PASS at 32k / 131k / 256k; tiny-context guards held.
- **Scales sublinearly** (1.61× time for 1.96× tokens) — the right direction vs item-C's superlinear.
- **But +23% slower at 131k** (353 s vs item-C 287 s) and **wedges at 512k**.

**Root cause (fundamental, not the union kernel):** the candidate all-gather moves
`N × query-tokens × topk` per layer. Prefill has thousands of query rows per chunk (decode has
1–4), so the candidate exchange is hundreds of GB of fabric traffic and dominates at
low/moderate context; the per-chunk local-logits buffer (`context/8` wide) grows and collides
with the reserved 1M KV pool at 512k (OOM-wedge). The `torch.topk`-union fix (vs the
double-argsort) only saved ~6% (374 → 353 s) — confirming the bottleneck is the *exchange*, not
the union. **The candidate-union transfers to decode (+12%, in production) but not to prefill.**

Fails the "not slower at small context" bar → **production keeps item-C prefill.** Code on
branch `glm52-prefill-port` for a future revisit (a true distributed top-k that avoids the full
candidate all-gather; or context-routing to use it only for the 256k–~450k band where it beats
item-C). The biggest *practical* long-context lever remains **prod-steering** (route <512k to
the non-DCP `prod` path) from the roadmap — independent of this.

## dcp-1m faster (research 2026-06-29) — see `DCP1M-FASTER-RESEARCH.md`

Deep code-grounded survey of how to speed up dcp-1m (keep 1M + 4 slots). Headlines:
- **The 4 slots ARE the win.** Live batch curve (8k, decode-isolated, `concurrent_decode.py`):
  **N=1 18.9 → N=2 28.8 (1.52×) → N=4 40.3 tok/s aggregate (2.13×)** — the engine amortizes the
  61% MoE-weight read across concurrent streams. Already live for <256k; for concurrent ≥256k raise
  the proxy `VLLM_PROXY_LONG_CONTEXT_TOKENS` (Amdahl-capped ~1.8× at true 1M by the 39% per-row indexer scan).
- **Decode is bandwidth-bound:** MoE weights ~61% / indexer full-shard-K scan ~39% / selected-MLA+all-gather <0.5%.
- **Kernel-verified correction:** `index_topk` 2048→1024 is NOT a bandwidth lever (the indexer scans the full
  shard-K *before* top-k, `sm12x_mqa.py`) → ~+2–5% (compute) with full multi-fact 1M-recall risk → cheap-maybe.
- Real per-token byte lever = **`num_experts_per_tok` 8→6** (+8–13% single-stream, recall-gated, off-distribution).
- Prefill: **`max-num-batched-tokens` 4096→8192/16384** (cheap, cuts per-chunk all-gatherv rounds) > the effort-L
  distributed-topk ring. NEW-dead: all-gather score quant (0.03% of bytes), sub-FP8 indexer-K (needs new sm_121 FP4 kernel).

## Not started (legacy notes; superseded by the dcp-1m research above for the DCP path)
- #3 MTP k=3→5 A/B · #4 max-num-batched-tokens 4096→8192 · #5 MAX_LOGITS_MB.
