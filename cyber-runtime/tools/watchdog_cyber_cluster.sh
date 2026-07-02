#!/usr/bin/env bash
# Active-probe watchdog for glm-5.2-nvfp4-cyber (the layer-targeted ablation model).
#
# Catches the GB10 collective/unified-memory WEDGE that the health/counter watchdogs miss: on a wedge the
# engine keeps /health=200 and num_requests_running=0 (looks idle) while the GPUs spin and generation is
# frozen — the stall detectors skip it because running==0. This one ACTIVELY PROBES: it fires a trivial
# generation and, if that TIMES OUT *while the generation counter is frozen* (i.e. not merely busy serving
# other requests), counts a wedge. After N consecutive wedges it relaunches the cyber model at a SAFE util,
# flock-guarded so it can't collide with a manual restart.
#
# Deployed on the head as ~/cyber-watchdog/watchdog_cyber_cluster.sh, run every ~3 min by
# sparks-cyber-watchdog.timer (systemctl --user). Runs as blacksheeep (has ssh keys + passwordless sudo docker).
set -uo pipefail

HEALTH="http://127.0.0.1:8000/health"
METRICS="http://127.0.0.1:8000/metrics"
GEN="http://127.0.0.1:8000/v1/chat/completions"
MODEL="${CYBER_MODEL:-glm-5.2-nvfp4-cyber}"
CONTAINER="${CYBER_CONTAINER:-vllm-glm52-cyber}"
LAUNCH="${CYBER_LAUNCH:-/home/blacksheeep/glm52-cyber-ablation/scripts/05_launch_cyber_model.sh}"
PROFILE="${CYBER_PROFILE:-dcp-1m}"   # standing prod config = 1M + 4 slots (validated 2026-07-02); a restart keeps 1M
UTIL="${CYBER_UTIL:-0.82}"                     # 0.82 is the LOAD floor (0.72 OOMs the load); the wedge was a
                                               # production conflict, not util — see below.
# 200s so a COLD-START JIT (~3 min Triton compile on the first request, "not a wedge") completes within the
# probe and resets — a true wedge hangs indefinitely, so it still times out and counts.
PROBE_TIMEOUT="${CYBER_PROBE_TIMEOUT:-200}"
NEED="${CYBER_WEDGE_CONSECUTIVE:-2}"           # consecutive wedge detections before restarting
STATE="/var/tmp/sparks-cyber-watchdog"; mkdir -p "$STATE"
WEDGE_FILE="$STATE/wedge-count"; LOCK="$STATE/restart.lock"

log() { logger -t sparks-cyber-watchdog "$*" 2>/dev/null || true; echo "$(date -Is) $*"; }
gen_counter() { curl -s --max-time 5 "$METRICS" 2>/dev/null | awk '/^vllm:generation_tokens_total/{print int($2)}' | head -1; }

# 1. Health gate: if the API is down (or restarting), this is NOT the wedge case — leave it (a death/restart
#    is either the death-watchdog's job or an in-progress relaunch). Reset the wedge counter and exit.
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 "$HEALTH" 2>/dev/null || true)"
if [ "$code" != "200" ]; then echo 0 > "$WEDGE_FILE"; log "health=$code (not the wedge case) — skip"; exit 0; fi

# 2. Active probe: a trivial generation. Note the generation counter before/after.
gen1="$(gen_counter)"
pc="$(curl -s -o /dev/null -w '%{http_code}' --max-time "$PROBE_TIMEOUT" "$GEN" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"say OK\"}],\"max_tokens\":8,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" 2>/dev/null || true)"
if [ "$pc" = "200" ]; then echo 0 > "$WEDGE_FILE"; log "probe ok — healthy"; exit 0; fi

# 3. Probe failed. Busy (counter moving) vs wedged (counter frozen)?
gen2="$(gen_counter)"
if [ -n "${gen1:-}" ] && [ -n "${gen2:-}" ] && [ "$gen2" -gt "$gen1" ]; then
  echo 0 > "$WEDGE_FILE"; log "probe=$pc but generation moving ($gen1->$gen2) = busy, not wedged"; exit 0
fi

# 4. Probe failed AND generation frozen = WEDGE. Count consecutive; restart at NEED.
n=$(( $(cat "$WEDGE_FILE" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$WEDGE_FILE"
log "WEDGE suspected: probe=$pc, gen frozen (${gen1:-?}==${gen2:-?}), count=$n/$NEED"
if [ "$n" -ge "$NEED" ]; then
  (
    flock -n 9 || { log "restart lock held (manual/other restart in progress) — skip"; exit 0; }
    echo 0 > "$WEDGE_FILE"
    log "WEDGE CONFIRMED — relaunching $CONTAINER at util=$UTIL"
    # FORENSICS FIRST: dump where every rank is stuck BEFORE we kill anything (the whole point — find the root
    # cause of the hang). ~60s, timeout-bounded so it can never block the recovery.
    if [ -x "$HOME/cyber-watchdog/capture_wedge_forensics.sh" ]; then
      log "capturing wedge forensics (py-spy all 8 ranks) -> ~/cyber-watchdog/wedge-forensics/"
      timeout 150 "$HOME/cyber-watchdog/capture_wedge_forensics.sh" "wedge-$(date +%Y%m%d-%H%M%S)" 2>&1 | tail -12 | while IFS= read -r l; do log "  $l"; done || log "forensics capture failed/timed out (continuing to restart)"
    fi
    # stop PRODUCTION + disable its watchdog first — vllm-glm52 (62G/node) can't co-exist with cyber and was
    # the true wedge cause (the glm52 watchdog kept resurrecting it).
    sudo systemctl disable --now sparks-glm52-watchdog.timer >/dev/null 2>&1 || true
    sudo docker rm -f "$CONTAINER" vllm-glm52 >/dev/null 2>&1 || true
    for h in 192.168.88.{102..108}; do ssh -o ConnectTimeout=8 "$h" "sudo docker rm -f $CONTAINER vllm-glm52 >/dev/null 2>&1" 2>/dev/null || true; done
    # WAIT for GPU/unified memory to be reclaimed before reloading — a fresh NVFP4 load on stale memory OOMs
    # (rank 0 dies mid-load, NVRM Out of memory). Poll the head's unified memory down to <20% (or ~100s max).
    for i in $(seq 1 20); do
      used="$(free -m | awk '/Mem:/{print int($3*100/$2)}')"
      log "waiting for memory reclaim: head=${used:-?}%"
      [ "${used:-100}" -lt 20 ] && break
      sleep 5
    done
    sleep 5
    CONFIRM_CYBER=YES GPU_MEMORY_UTILIZATION="$UTIL" bash "$LAUNCH" "$PROFILE" > /home/blacksheeep/glm52-cyber-launch.log 2>&1
    # wait for health, then WARM the Triton JIT so the next watchdog probe isn't a false positive
    for i in $(seq 1 130); do [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 "$HEALTH" 2>/dev/null)" = "200" ] && break; sleep 15; done
    curl -s -o /dev/null --max-time 240 "$GEN" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":8,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" 2>/dev/null || true
    log "relaunch complete + JIT warmed"
  ) 9>"$LOCK"
fi
