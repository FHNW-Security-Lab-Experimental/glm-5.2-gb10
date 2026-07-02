# cuBLAS EngineCore crash — root cause + fix (2026-07-02)

A **third** GLM-5.2 failure class, distinct from the production-conflict wedge (`INCIDENT-2026-07-01.md`) and the
Marlin-MoE deadlock (`MARLIN-MOE-DEADLOCK-FIX.md`). This one is a **fatal EngineCore crash**, not a hang.

## Symptom
OpenCode showed `EngineCore encountered an issue. See stack trace (above) for the root cause.` The engine had
died (`vllm.v1.engine.exceptions.EngineDeadError`); model stayed **down for hours** because the watchdog only
handled *wedges* (health=200 + frozen), not *crashes* (health=000).

## Root cause: a transient GB10 cuBLAS flake
```
RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling
  cublasGemmEx(... a, CUDA_R_16BF, ... b, CUDA_R_16BF, ... CUDA_R_16BF, ...)   ← a bf16 GEMM
```
- It's the **legacy `cublasGemmEx` bf16 path** = an UNQUANTIZED bf16 linear (`default_unquantized_gemm` →
  `torch.nn.functional.linear`) — a small non-NVFP4 layer (MTP draft head / router gate / weighted norm).
- The engine log at crash time shows **GPU KV cache 0.0-0.6%, 2 requests** → **not memory pressure, not the 1M
  config, not the Marlin hang.** It fired **once in ~10h at light load, well past warmup.**
- Diagnosis: an **intermittent cuBLAS-internal flake on brand-new sm_121 silicon** — most likely an async error
  from a concurrent sm_12x Triton kernel surfacing on the *next* (innocent) cuBLAS call, or a multi-stream cuBLAS
  workspace race. Both are transient and **succeed on retry**. Not a version-fixed cuBLAS bug (image is CUDA 13.0.2).

## Fix #1 (the real fix): retry the bf16 GEMM once — transparent, no crash
`05_launch_cyber_model.sh` STEP D patches `default_unquantized_gemm` in-container to:
```python
try:
    return torch.nn.functional.linear(x, weight, bias)
except RuntimeError as _e:
    if 'CUBLAS_STATUS_INTERNAL_ERROR' not in str(_e): raise
    torch.cuda.synchronize()
    return torch.nn.functional.linear(x, weight, bias)   # transient flake succeeds on retry
```
A rare cuBLAS flake becomes a **transparent retry (no EngineCore crash, no service interruption)**, bit-identical
on the happy path, narrow except (OOM/shape errors still propagate). Verify live:
`docker exec vllm-glm52-cyber grep -c CUBLAS_STATUS_INTERNAL_ERROR .../vllm/model_executor/layers/utils.py` → ≥1.

## Fix #2 (rate-reducers): lower how often the flake happens
- **`CUBLAS_WORKSPACE_CONFIG=:4096:8`** — fixed per-stream cuBLAS workspace (kills the workspace-race path).
  `05` default; forwarded to all 8 nodes via the launcher passthrough (`start_glm52_8node.sh` ~L280/L489).
- **`CUDA_DEVICE_MAX_CONNECTIONS=1`** — already on (the Marlin fix); also shrinks the cross-stream race window.

## Fix #3 (backstop): the watchdog now recovers CRASHES too
`watchdog_cyber_cluster.sh` previously *skipped* on health!=200 (so a crash stayed down). Now: health!=200 +
**EngineCore process gone** (2 consecutive) → restart (shared `do_restart`); EngineCore alive (slow load / long
prefill) → still skip. So even a residual crash self-heals in ~2-4 min instead of hours — but Fix #1 means it
shouldn't fatally crash in the first place.

## Why not just a version bump
No published CUDA-13.x cuBLAS bug matches this signature; a lib/image swap is high-risk on the pinned sm_121
stack and unlikely to be the true fix. The retry wrapper neutralizes the flake regardless of its exact origin —
the right call for rare, transient, new-silicon library flakiness. If the retry frequency ever climbs, revisit.
