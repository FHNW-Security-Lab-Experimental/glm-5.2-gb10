#!/usr/bin/env bash
# start_glm52_config.sh — one entry point for the three GLM-5.2 serving configs on the 8x GB10
# cluster. Sets the right env per profile and execs the shared launcher (~/vllm-glm52/runtime/
# launch.sh). All profiles: TP=8, fp8_ds_mla KV, in-checkpoint MTP, --enforce-eager, util 0.78.
#
#   prod      non-DCP, 512k, KV replicated per rank. FASTEST (~22.5 tok/s). Production default.
#   dcp-512k  DCP=8, 512k, KV sharded. ~20-21 tok/s. Same context as prod, lower per-stream mem.
#   dcp-1m    DCP=8, 1M (1048576), KV sharded. ~17-18 tok/s decode. TRUE 1M context.
#
# Decode tok/s measured 2026-06-28 (single stream, warm): prod ~22.5 | dcp-512k ~20-21 | dcp-1m ~17-18.
# CAVEAT (dcp-*): prefill of very long prompts is slow (per-chunk-per-layer indexer K all-gather);
#   ~131k prompts are practical, ~400k+ prefill is impractically slow today (optimization target).
#   Decode speed above is unaffected. See CONFIGS.md.
#
# DCP profiles require the staged DCP artifacts on ALL 8 nodes:
#   ~/glm-triton-dcp/                              (kernels: copy of ~/glm-triton + LSE-returning flashmla_sparse.py)
#   ~/vllm-glm52/runtime/glm52-sparse-patches-dcp.sh   (combined base+DCP patch; regen via build-dcp-patch.sh)
# Watchdog: sparks-glm52-watchdog restarts via launch.sh with PRODUCTION env, so it would clobber a
#   DCP run. This script STOPS the watchdog for dcp-* and re-ENABLES it for prod.
set -euo pipefail
PROFILE="${1:-}"
RT=/home/blacksheeep/vllm-glm52/runtime
export CONFIRM_GLM52=YES
# ENFORCE_EAGER env-overridable for the cudagraph-on-prod A/B (default 1 = eager, unchanged).
# Set ENFORCE_EAGER=0 to let the launcher capture CUDA graphs (b12x-probed FULL/PIECEWISE).
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.78}"

case "$PROFILE" in
  prod|non-dcp-512k)
    export KERNELS_DIR=/home/blacksheeep/glm-triton
    export PATCH_SCRIPT=$RT/glm52-sparse-patches.sh
    export MAX_MODEL_LEN=524288
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    # PROD_EXTRA_ARGS lets the prefix-caching A/B inject --enable-prefix-caching /
    # --no-enable-prefix-caching without further edits (default empty = unchanged).
    export EXTRA_VLLM_ARGS="${PROD_EXTRA_ARGS:-}"
    WATCHDOG=enable
    ;;
  dcp-512k)
    export KERNELS_DIR=/home/blacksheeep/glm-triton-dcp
    export PATCH_SCRIPT=$RT/glm52-sparse-patches-dcp.sh
    export MAX_MODEL_LEN=524288
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
    export EXTRA_VLLM_ARGS="--decode-context-parallel-size 8"
    WATCHDOG=disable
    ;;
  dcp-1m)
    export KERNELS_DIR=/home/blacksheeep/glm-triton-dcp
    export PATCH_SCRIPT=$RT/glm52-sparse-patches-dcp.sh
    export MAX_MODEL_LEN=1048576
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    export EXTRA_VLLM_ARGS="--decode-context-parallel-size 8"
    WATCHDOG=disable
    ;;
  *)
    echo "usage: $0 {prod|dcp-512k|dcp-1m}" >&2
    exit 2
    ;;
esac

echo "[start_glm52_config] profile=$PROFILE  max_model_len=$MAX_MODEL_LEN  max_num_seqs=$MAX_NUM_SEQS  dcp=${EXTRA_VLLM_ARGS:-off}"

# Watchdog: only the PRODUCTION (non-DCP) config may be auto-restarted by the watchdog
# (it relaunches with production env). DCP runs are managed manually.
if [[ "$WATCHDOG" == "disable" ]]; then
  sudo systemctl stop sparks-glm52-watchdog.timer 2>/dev/null || true
  echo "  watchdog: stopped (DCP run is manually managed)"
else
  echo "  watchdog: will be left for manual re-enable after health (sudo systemctl start sparks-glm52-watchdog.timer)"
fi

# Drop page cache on all 8 first — the 465 GB mmap (and the larger 1M prefill workspaces)
# need the unified-memory headroom.
for h in 192.168.88.{101..108}; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 blacksheeep@"$h" 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null' 2>/dev/null || true
done

exec "$RT/launch.sh"
