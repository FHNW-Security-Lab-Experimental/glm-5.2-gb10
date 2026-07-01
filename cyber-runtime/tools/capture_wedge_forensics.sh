#!/usr/bin/env bash
# capture_wedge_forensics.sh — snapshot WHERE every TP=8 rank is stuck during a wedge (a hung collective).
#
# Run BEFORE restarting a wedged model (the watchdog calls it on WEDGE CONFIRMED; run by hand the moment gen
# freezes). It py-spy-dumps every rank's vLLM worker + EngineCore from the HOST (host root bypasses the
# container's seccomp/ptrace_scope; the in-container py-spy is blocked without CAP_SYS_PTRACE). Python-only
# stacks (native unwinding is broken on aarch64/GB10, but the Python stack names the stuck vLLM op).
#
# Triage: 7 ranks will be parked in an NCCL collective (all_reduce/broadcast/all_gather = waiting). The ONE rank
# whose top frame is NOT a collective — down in sparse-MLA / MTP / a Triton kernel — is the culprit that hung.
#
#   ./capture_wedge_forensics.sh [label]
set -uo pipefail
PYSPY="${PYSPY:-$HOME/pyspy-venv/bin/py-spy}"
TS="${1:-$(date +%Y%m%d-%H%M%S)}"
OUT="${WEDGE_OUT:-$HOME/cyber-watchdog/wedge-forensics}/$TS"
NODES=(192.168.88.101 192.168.88.102 192.168.88.103 192.168.88.104 192.168.88.105 192.168.88.106 192.168.88.107 192.168.88.108)
mkdir -p "$OUT"

# per-node dump (base64 so no quoting survives ssh): dump the worker + EngineCore via host py-spy on their host PIDs
read -r -d '' NODE_SH <<'EOS' || true
PS="$HOME/pyspy-venv/bin/py-spy"
for pat in "VLLM::Worker_TP" "VLLM::EngineCore"; do
  for pid in $(ps -eo pid,args | grep "$pat" | grep -v grep | awk '{print $1}'); do
    st=$(cut -d' ' -f3 /proc/$pid/stat 2>/dev/null)
    echo "### $pat pid=$pid state=$st"
    sudo "$PS" dump --pid "$pid" 2>&1
    echo
  done
done
EOS
B64=$(printf '%s' "$NODE_SH" | base64 -w0)

echo "[forensics] py-spy stacks, all 8 ranks -> $OUT"
for i in "${!NODES[@]}"; do
  h="${NODES[$i]}"
  ( if [ "$h" = "192.168.88.101" ]; then echo "$B64" | base64 -d | bash
    else ssh -o ConnectTimeout=8 "$h" "echo $B64 | base64 -d | bash"; fi ) > "$OUT/rank${i}.txt" 2>&1 &
done
wait

echo "[forensics] GPU state..."
for i in "${!NODES[@]}"; do
  h="${NODES[$i]}"
  ( if [ "$h" = "192.168.88.101" ]; then nvidia-smi; else ssh -o ConnectTimeout=8 "$h" nvidia-smi; fi ) > "$OUT/gpu_rank${i}.txt" 2>&1 &
done
wait

echo "[forensics] done -> $OUT"
echo "=== TRIAGE: ranks parked in a collective (waiting) vs the culprit ==="
for f in "$OUT"/rank*.txt; do
  r=$(basename "$f" .txt)
  if grep -qiE "all_reduce|allreduce|broadcast|all_gather|_c10d|work.wait|ProcessGroupNCCL" "$f" 2>/dev/null; then
    echo "  $r: WAITING (in collective)"
  else
    top=$(grep -m1 -E "\(vllm/|\.py:" "$f" 2>/dev/null | sed 's/^ *//')
    echo "  $r: *** SUSPECT (not in collective) *** top: ${top:-<no python frame>}"
  fi
done
