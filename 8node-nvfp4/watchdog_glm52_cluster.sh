#!/usr/bin/env bash
# watchdog_glm52_cluster.sh — restart GLM-5.2 if its engine dies.
#
# Runs on the HEAD (single-shot; driven by sparks-glm52-watchdog.timer every 60s).
# GLM-5.2's NVFP4 MoE (FlashInfer-CUTLASS) can intermittently hit a CUDA illegal
# memory access (cudaMemsetAsync in cutlass_fused_moe) that kills EngineCore. The
# container stays Up (entrypoint is `sleep infinity`) but `vllm serve` exits, so
# /health goes dead. This restarts the cluster ONLY when the engine is truly gone —
# never during a slow long-context prefill (which can look frozen but the serve
# process is alive). Manual + watchdog restarts share restart.lock (flock).
set -uo pipefail

HEALTH="${GLM52_HEALTH_URL:-http://127.0.0.1:8000/health}"
CONTAINER="${GLM52_CONTAINER:-vllm-glm52}"
LAUNCH="${GLM52_LAUNCH:-/home/blacksheeep/vllm-glm52/runtime/launch.sh}"
LOCK="${GLM52_LOCK:-/home/blacksheeep/vllm-glm52/restart.lock}"
LOG="${GLM52_WATCHDOG_LOG:-/home/blacksheeep/vllm-glm52/logs/watchdog.log}"
STATE="${GLM52_STATE:-/home/blacksheeep/vllm-glm52/.watchdog_fail}"

# Production launch env — keep in sync with the documented 512k config.
LAUNCH_ENV=(
  CONFIRM_GLM52=YES
  ENFORCE_EAGER="${GLM52_ENFORCE_EAGER:-1}"
  MAX_MODEL_LEN="${GLM52_MAX_MODEL_LEN:-524288}"
  MAX_NUM_SEQS="${GLM52_MAX_NUM_SEQS:-4}"
  GPU_MEMORY_UTILIZATION="${GLM52_GPU_UTIL:-0.78}"
)

mkdir -p "$(dirname "$LOG")"
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" >> "$LOG"; }

code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 "$HEALTH" 2>/dev/null || true)"
if [ "$code" = "200" ]; then
  rm -f "$STATE" 2>/dev/null || true
  exit 0
fi

# /health not healthy. Is the engine process actually gone? (slow prefill => alive)
serve_alive=0
if sudo docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
  if sudo docker exec "$CONTAINER" pgrep -f 'vllm serve' >/dev/null 2>&1; then
    serve_alive=1
  fi
fi
if [ "$serve_alive" = "1" ]; then
  log "health=$code but 'vllm serve' still running (loading or slow prefill) — NOT restarting"
  rm -f "$STATE" 2>/dev/null || true
  exit 0
fi

# Require TWO consecutive dead checks before restarting (matches the Kimi watchdog
# discipline; avoids a transient probe failure triggering a restart).
prev="$(cat "$STATE" 2>/dev/null || echo 0)"
if [ "$prev" -lt 1 ]; then
  echo 1 > "$STATE"
  log "health=$code and engine process gone (strike 1/2) — will restart next check if still dead"
  exit 0
fi
rm -f "$STATE" 2>/dev/null || true

log "health=$code and engine process gone (confirmed) — restarting GLM-5.2"
exec 9>"$LOCK"
if ! flock -n 9; then
  log "another restart holds restart.lock — skipping"
  exit 0
fi
( cd /home/blacksheeep/vllm-glm52/runtime && env "${LAUNCH_ENV[@]}" bash "$LAUNCH" ) >> "$LOG" 2>&1
log "restart launched (rc=$?)"
