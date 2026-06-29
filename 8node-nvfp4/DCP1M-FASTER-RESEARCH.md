# Making dcp-1m faster (true 1M + 4 slots) — research findings (2026-06-29)

Deep, code-grounded investigation of how to speed up the **dcp-1m** profile (DCP=8, `max_model_len=1,048,576`,
`MAX_NUM_SEQS=4`, `--enforce-eager`, util 0.78, PERF#2 decode) **without dropping below 1M context or 4 slots**.
Method: 7 lever families investigated against the actual patch/kernel code, then synthesis + an adversarial
critic that re-read the kernels. One live measurement (the aggregate batch curve) was run; the rest are
analysis + concrete experiments to run later. Single-stream baseline this session: ~18.7–18.8 tok/s warm @4k.

## TL;DR

- **The 4 slots ARE the speedup.** Live batch curve (8k ctx, decode-isolated): **N=1 18.9 → N=2 28.8 (1.52×)
  → N=4 40.3 tok/s aggregate (2.13×).** The engine amortizes the dominant MoE-weight read across concurrent
  streams. This is **already live for <256k traffic** (the proxy only serializes ≥256k). Single-stream tok/s
  badly undersells dcp-1m for a multi-user cluster.
- **Single-stream decode is near the bandwidth wall.** Per-step decode bytes ≈ **MoE expert weights 61% /
  lightning-indexer full-shard-K scan 39% / selected-MLA + PERF#2 all-gather <0.5%.** Latency-only levers are
  capped (proven: `NCCL_PROTO=LL128` made decode ~6% *slower*). No single decode lever beats ~+15% single-stream.
- **`index_topk` is NOT a bandwidth lever** (kernel-verified correction — see below).

## Decode bandwidth model (per rank, per step, at 1M)

The MQA logits kernel (`kernels/sm12x_mqa.py`) scans the **entire** shard-K (`seq_len_kv = k_fp8.shape[0]`,
loops the full range) to score every key **before** top-k selection. So:

| term | share | scales with | amortizes across concurrent rows? |
|---|---|---|---|
| MARLIN MoE active-expert weight reads (~8 experts × 3 GEMMs × ~75 layers) | ~61% | experts/token | **yes** (read once per step regardless of rows) |
| lightning-indexer full-shard-K scan (compress_ratio=1 → every key scored) | ~39% | **context length** (per row) | **no** (each stream scans its own shard) |
| selected-key MLA attend (`topk × 656 B`) + PERF#2 all-gather `[rows, 8×topk]` | <0.5% | index_topk | n/a |

Consequences: (1) the only way past the single-stream wall is **amortizing the 61% MoE read across rows** =
the aggregate axis. (2) Aggregate is Amdahl-capped at true 1M to **~1.8×** because the 39% indexer-scan term is
per-row and does **not** amortize (at short context it scans little → the measured 2.13× at 8k is the optimistic
end). (3) `index_topk` only touches the <0.5% terms.

## ⚠️ Kernel-verified correction: `index_topk` 2048→1024 is not a bandwidth win

A prior survey listed `index_topk` halving as "the largest measurable BW win (halves the all-gather)". **Wrong.**
The indexer scans the full shard-K *before* selecting top-k, so `index_topk` is independent of the 39%
indexer-scan term; halving it shrinks only the all-gather (~0.03%) + selected-MLA read (~0.4%) = **<0.5% of
decode bytes.** Its real decode gain is **~+2–5% (mostly compute, not bytes)** while it carries the **full
multi-fact 1M-recall risk** (halving the receptive field is exactly what 1M exists to avoid). **Demoted to
cheap-maybe**, not a headline lever. (`compress_ratio` must stay 1 — asserted in `glm52-dcp-patches.sh:400`.)

## Ranked levers (keep 1M + 4 slots)

### #1 — Use the 4 slots concurrently (aggregate). LOSSLESS, no reboot. **CONFIRMED 2.13× @ N=4.**
The dcp-1m KV pool holds ~4.12× a full-1M context, but `tools/vllm_keepalive_proxy.py` takes an exclusive lock
for any prompt ≥`VLLM_PROXY_LONG_CONTEXT_TOKENS` (=256000) → only **one** ≥256k stream runs end-to-end. For
<256k traffic the 4 slots already run concurrently (the 2.13× is live). To extend the aggregate win to
concurrent **long-context** (≥256k) users, raise `VLLM_PROXY_LONG_CONTEXT_TOKENS` (e.g. 1.1M) /
`VLLM_PROXY_SERIALIZE_LONG_CONTEXT=0` and restart the proxy service (no engine reboot). **Production-facing**
(public proxy) + needs OOM-watch (4 concurrent long streams). Gate: aggregate ≥1.6× at N=4 + monotonic 1→4 +
min free unified mem >~8 GB/node through the overlap (the all-8-node OOM-wedge ceiling) + a 16k/131k needle on
one concurrent stream still coherent. Probe: `concurrent_decode.py` (engine-direct batch curve).

### #2 — `num_experts_per_tok` 8→6 (single-stream + aggregate). The only verified per-token *byte* lever.
Hits the 61% MoE-weight term directly (~−15% of total decode bytes), MARLIN-unchanged (just gathers fewer
grouped tiles). **+8–13% single-stream** (best-grounded decode number). **Off-distribution** (model trained at
top-8) → recall-risky. Experiment: 5-min pre-check logging `num_experts_per_tok` in-container (confirm the
override reaches the router; if `config_glm_moe_dsa` ignores the alias the lever is void), then
`HF_OVERRIDES='{"qk_rope_head_dim":64,"num_experts_per_tok":6}'`, keep 4 slots / 1M, separate boot. Gate
(RECALL + coherence, never /health): decode ≥+6% AND needle recall == baseline (zero misses 16k+131k, 3 depths)
AND a ≥5-item reasoning/coherence check intact AND accepted-tokens/step not down >0.2. Any miss → revert. Try 6
before 4. **Do NOT stack with any other sparse/mixture change in one A/B** (unattributable recall regression).

### #3 — `max-num-batched-tokens` 4096→8192/16384 (prefill). Cheapest attack on the real long-context pain.
dcp-1m prefill is slow (131k≈287s, 256k≈754s, 1M≈30–40min, superlinear): the item-C indexer all-gatherv +
reassembly fires **per-chunk-per-layer**, chunk count = prefill_tokens / chunk_size. Bigger chunks → ~linearly
fewer collective rounds, on the existing proven path, zero new kernel. Trades unified-memory headroom (bigger
prefill activation) for fewer rounds. Gate: OOM headroom at util 0.78 (drop to 0.74 if it captures-OOMs) +
prefill faster at 131k/256k + sparse top-k set-equality unchanged.

## Parked / maybe (with trigger)

- **MTP k-sweep {2,4,5}** (`NUM_SPECULATIVE_TOKENS`) — config-only, lossless (verify is exact). Trigger:
  piggyback on any boot; expectation is k=3 stays optimal (single nextn head reused autoregressively, tokens
  4–5 decay at long context). Cheap insurance, not a headline.
- **MTP cheaper-draft attention/MoE** — make the *draft* forward cheaper (smaller draft-only `index_topk`, or
  draft-only `num_experts_per_tok`) while the *verify* keeps full settings → output bit-exact, each speculative
  token cheaper. Needs a per-layer draft-vs-target discriminator patch (vLLM resolves one global value). +2–4%.
  Trigger: only if #2 is rejected at 1M for recall. (The draft also routes through MARLIN MoE, so a draft-only
  experts cut targets the dominant byte term on rejected drafts — better than draft-only index_topk.)
- **Ring/streaming distributed top-k for prefill** — effort L; the only structural prefill lever. Replaces
  item-C's full-K all_gatherv + reassembly with a ring that shrinks peak working set N×topk → topk (fixes the
  512k wedge that killed the candidate-union) and removes the full-K gatherv. Two GB10 unknowns gate it:
  (1) does the vLLM DCP GroupCoordinator expose usable p2p send/recv (else it degrades to N−1 serial rounds);
  (2) do N−1 hops × ~75 layers × num_chunks fit the latency budget. Trigger: only if true-1M prefill latency
  becomes the binding complaint. Mandatory offline SET-equality unit test (streaming-top-k correctness trap)
  before any boot; boot gate: SET-equality vs N=1 + needle through 512k + **not slower than item-C at 131k**
  (the bar PERF#2-prefill failed) + faster at ≥256k.

## Confirmed dead (do not relitigate)

- **`index_topk` as a *bandwidth* lever** (above — it's ~+2–5% compute, full recall risk; cheap-maybe at best).
- **Quantize the PERF#2 all-gather scores** (fp32→fp16/fp8) — the all-gather is ~0.03% of decode bytes
  (~7 MiB/step/node); saves sub-0.1% AND collapses near-ties at the 2048 boundary → silent recall erosion.
- **Sub-FP8 (FP4/INT4) indexer-K read** — indexer K is already FP8 e4m3; lower needs a net-new sm_121 FP4
  nibble-unpack Triton kernel (none exists; the FlashInfer/CUTLASS FP4 class IMAs on sm_121). Best case <+6%,
  dominated by the free index_topk path anyway.
- From the prior survey: cudagraph-on-DCP, prefix-caching-on-DCP (sharded KV → 0 hits), Expert-Parallel MoE
  (forces broken FP4 grouped-GEMM + wedged all 8 nodes), two TP=4 replicas (~116 GiB/node = OOM),
  `NCCL_PROTO=LL128` (regression), PERF#1 prefill reassembly-index cache (multi-chunk all_gatherv deadlock),
  PERF#2-for-prefill candidate-union (+23% @131k, wedges @512k).

## Honest bottom line

For a **multi-user** dcp-1m deployment the throughput headroom is the **aggregate axis** — already 2.13× at
4 slots for <256k, extendable to long-context concurrency via the proxy threshold (#1). **Single-stream** has
~+15% left at most (`num_experts_per_tok`, recall-gated). **Prefill** is the largest absolute latency; the
cheap attack is `max-num-batched-tokens` (#3), the structural one (distributed-topk ring) is effort-L and
GB10-uncertain. None of these requires dropping 1M or the 4 slots.

## Files
`kernels/sm12x_mqa.py` (full-shard-K scan before top-k — the index_topk correction), `kernels/flashmla_sparse.py`
(`self.topk_tokens = hf_config.index_topk`; 656 B/key fp8_ds_mla), `glm52-dcp-patches.sh` (PERF#2 all-gather
L254-255, topk whitelist {512,1024,2048} L297/L330, `assert compress_ratio==1` L400, item-C prefill L485-511),
`start_glm52_8node.sh` (HF_OVERRIDES L127, NUM_SPECULATIVE_TOKENS L109, util>0.78 OOM guard L449-454),
`start_glm52_config.sh` (dcp-1m profile), `tools/vllm_keepalive_proxy.py` (LONG_CONTEXT serialization),
`concurrent_decode.py` (the aggregate batch-curve probe). See also `PERF-RESULTS.md`, `CONFIGS.md`.
