# The Marlin NVFP4-MoE collective hang — root cause + fix (2026-07-01)

The **second, harder** cyber-model wedge cause (the first was the production/cyber memory conflict — see
`INCIDENT-2026-07-01.md`). This one is a genuine runtime-kernel deadlock, found by live forensics and fixed with
a single env var at **zero feature cost**.

## Symptom
Under real OpenCode agentic load the model froze: GPUs ~100% on all 8 nodes, `/health`=200, `num_requests_running`=0,
`generation_tokens_total` frozen, trivial requests hang — a hung TP=8 collective, **no crash, no error log**.

## Root cause (proven, not guessed): `moe_wna16_marlin_gemm` deadlock
Method: on a wedge, `capture_wedge_forensics.sh` py-spy-dumps all 8 ranks from the host **twice, 8 s apart**;
a rank whose stack is byte-identical across both = truly stuck (not merely mid-forward). Result:
```
2 of 8 ranks STUCK in  moe_wna16_marlin_gemm (vllm/_custom_ops.py:2469)   ← the MARLIN NVFP4 MoE kernel
                       _fused_marlin_moe -> modelopt.py:1660 -> deepseek_v2.py:373 (MoE layer)
6 of 8 ranks STUCK in  a downstream dense GEMM  ← parked waiting on the MoE collective the 2 never complete
ablation_patch: in ZERO of the 8 stuck stacks  ← the ablation is NOT involved; this is pure runtime
```
**Mechanism:** with TP=8 the NVFP4 experts are sharded and top-k gating routes a different token→expert
distribution to each rank each step. Under `--max-num-seqs 4` + MTP (draft+verify) on CUDA's **default 8 hardware
connections**, and with `--enforce-eager` (no CUDA-graph ordering), the MoE GEMM and the following all-reduce
launch on separate HW queues that **race on Marlin's shared workspace buffer**. On the 2 ranks whose routing hits
the pathological shape/timing, the kernel never signals completion; the other 6 correctly park in the all-reduce.
Matches upstream **vLLM #41725** (same GB10/sm_121 + NVFP4-MoE + TP; compute-layer stall; NCCL toggles
ineffective; removing Marlin made it *worse*). Note all three NVFP4-MoE backends are GB10-broken: FlashInfer-CUTLASS
illegal-memory-accesses, VLLM_CUTLASS is garbage, MARLIN deadlocks — MARLIN was the least-bad and this is its bug.

## The fix: `CUDA_DEVICE_MAX_CONNECTIONS=1`
Collapses the 8 HW queues to 1, forcing `MoE GEMM → all-reduce → MTP` into strict program order (the ordering
CUDA graphs would give, which we can't use on enforce-eager / DCP). Pure env; the launcher already forwards it to
all 8 nodes (`launch.sh` line ~280 local + ~489 remote). **Baked into `05_launch_cyber_model.sh` as the default**,
so `start_cyber.sh`, direct `05`, and the watchdog's restart all inherit it. **Keeps MTP + 4 slots + 1M + eager.**

## Validation (before/after, same load)
| | baseline (no fix) | `CUDA_DEVICE_MAX_CONNECTIONS=1` |
|---|---|---|
| 4-concurrent realistic stress | **permanent deadlock** in ~80s (2-snapshot: all 8 ranks stuck in Marlin/GEMM; needed a restart) | **22 min clean** — gen counter climbed 7,234→118,467 (~5k tok/min), never froze, probes all 200 |

Re-validate any time: `LABEL=x CONC=4 DURATION=1800 ~/cyber-watchdog/stress_repro.sh` (declares a wedge only on a
sustained 3-min freeze confirmed by 2-snapshot — no false positives from prefill saturation).

## If it ever recurs (fallback ladder, all keep features)
1. `--disable-custom-all-reduce` (force NCCL all-reduce instead of vLLM's multi-stream custom all-reduce).
2. `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180` — self-heal instead of wedge.
3. **EMULATION MoE backend** — one-line oracle edit (`_glm52_unsafe |= {MARLIN}` in `glm52-sparse-patches.sh`
   STEP C) physically removes `moe_wna16_marlin_gemm`. Strongest stability fix, but measure the decode-speed cost
   (dequant NVFP4→bf16) vs the 22.5 tok/s baseline first. Do NOT global-`--moe-backend emulation` (kills the MTP head).

The **active-probe watchdog** (`tools/`) remains the last-resort backstop; it also auto-runs the forensics capture
before any restart, so a novel hang is always diagnosable.
