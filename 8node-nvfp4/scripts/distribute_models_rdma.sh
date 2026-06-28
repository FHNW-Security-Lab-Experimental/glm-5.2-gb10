#!/usr/bin/env bash
# Distribute one or more model dirs from the head (~/models/<name>) to the other
# nodes over the 200G RDMA fabric (10.0.0.x), in parallel.
#
# Run ON the head (spark-a175). The RDMA IPs are only reachable node-to-node, so
# this must run on a cluster node, not a workstation. Bulk model transfer is the
# one sanctioned use of the RDMA net for non-inference traffic (cf. transfer-cellB.sh).
#
#   bash ~/distribute_models_rdma.sh GLM-5.2-NVFP4 MiniMax-M3-NVFP4
#
# Override targets:  TARGETS="10.0.0.12 10.0.0.13" bash ~/distribute_models_rdma.sh <model...>
set -uo pipefail

MODELS=("$@")
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=(GLM-5.2-NVFP4 MiniMax-M3-NVFP4)

# Worker RDMA IPs (head is 10.0.0.11). Default = all 7 workers (102-108).
TARGETS="${TARGETS:-10.0.0.12 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17 10.0.0.18}"
SRC_BASE="${SRC_BASE:-$HOME/models}"
DST_BASE="${DST_BASE:-/home/blacksheeep/models}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15)
LOG_DIR="$HOME/models/logs"; mkdir -p "$LOG_DIR"

log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

# sanity: sources must exist on the head
for m in "${MODELS[@]}"; do
  [[ -d "$SRC_BASE/$m" ]] || { log "ERROR: source missing: $SRC_BASE/$m"; exit 1; }
done
log "distributing: ${MODELS[*]}"
log "targets: $TARGETS"

push_node() {
  local ip="$1"; local lg="$LOG_DIR/distribute-$ip.log"
  : > "$lg"
  for m in "${MODELS[@]}"; do
    log "[$ip] rsync $m -> start" | tee -a "$lg"
    ssh "${SSH_OPTS[@]}" "blacksheeep@$ip" "mkdir -p '$DST_BASE/$m'" 2>>"$lg"
    # -aH preserve links/perms, --inplace+--partial resumable, numeric-ids; retry on drop
    local tries=0
    until rsync -aH --numeric-ids --partial --inplace --info=progress2 \
            -e "ssh ${SSH_OPTS[*]}" \
            "$SRC_BASE/$m/" "blacksheeep@$ip:$DST_BASE/$m/" >>"$lg" 2>&1; do
      tries=$((tries+1)); [[ $tries -ge 10 ]] && { log "[$ip] $m FAILED after $tries tries"; return 1; }
      log "[$ip] $m rsync dropped, retry $tries" | tee -a "$lg"; sleep 10
    done
    log "[$ip] $m done ($(ssh "${SSH_OPTS[@]}" "blacksheeep@$ip" "du -sh '$DST_BASE/$m' 2>/dev/null | cut -f1"))" | tee -a "$lg"
  done
}

for ip in $TARGETS; do push_node "$ip" & done
wait
log "ALL DISTRIBUTION DONE"
# Verify sizes match the head
for m in "${MODELS[@]}"; do
  src=$(du -sh "$SRC_BASE/$m" 2>/dev/null | cut -f1)
  log "verify $m  head=$src"
  for ip in $TARGETS; do
    printf '   %s -> %s\n' "$ip" "$(ssh "${SSH_OPTS[@]}" "blacksheeep@$ip" "du -sh '$DST_BASE/$m' 2>/dev/null | cut -f1")"
  done
done
