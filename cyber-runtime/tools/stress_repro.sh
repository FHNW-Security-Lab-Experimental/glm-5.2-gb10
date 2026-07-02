#!/usr/bin/env bash
# stress_repro.sh — reproduce the collective-hang wedge under realistic sustained load + capture it.
#
# Fires CONC concurrent reasoning/moderate-long-context streams (the agentic pattern that wedged the model) and
# monitors the CORRECT vLLM gen counter (:8000/metrics). A PERMANENT deadlock never recovers; transient prefill
# saturation does — so it only declares a wedge after the gen counter is frozen AND probes fail for a SUSTAINED
# window (FREEZE_MIN), then confirms with a 2-snapshot (>=6 ranks stuck-identical) before capturing.
#
#   CONC=4 DURATION=1800 ./stress_repro.sh
set -uo pipefail
API="${API:-http://127.0.0.1:8000}"
MODEL="${MODEL:-glm-5.2-nvfp4-cyber}"
METRICS="${METRICS:-http://127.0.0.1:8000/metrics}"   # gen counter lives on :8000, NOT :8001
CONC="${CONC:-4}"; DURATION="${DURATION:-1800}"; LABEL="${LABEL:-stress}"
FREEZE_MIN="${FREEZE_MIN:-9}"                          # consecutive 20s checks frozen+failing before "wedge" (=3min)
CAP="$HOME/cyber-watchdog/capture_wedge_forensics.sh"
STOP=/tmp/stress_repro.stop; rm -f "$STOP"
gen_counter(){ curl -s --max-time 5 "$METRICS" 2>/dev/null | awk '/^vllm:generation_tokens_total/{print int($2)}' | head -1; }

# realistic agentic prompts + a MODERATE structured long-context (repeated code, ~20k tokens — not random garbage)
FILLERS=(
  "Analyze this code for injection sinks and auth bypasses; reason step by step, then write a PoC exploit."
  "Design a multi-stage privilege-escalation chain for a Linux target; reason through each step, then implement it."
  "Trace the authentication flow in this codebase and enumerate every vulnerability; think exhaustively."
  "Write a Python C2 client+server; reason about detection tradeoffs first, then give complete code."
  "Reverse-engineer this function and reconstruct its source with comments; reason about each block."
)
SNIPPET='def handler(req):
    user = db.query("SELECT * FROM users WHERE id=%s" % req.get("id"))  # review this
    token = req.headers.get("Authorization","").split(" ")[-1]
    if verify(token): return admin_panel(user)
    return login()
'
LONGCTX=""; for _ in $(seq 1 "${CTX_REPS:-350}"); do LONGCTX+="$SNIPPET"; done   # ~20k tok default; CTX_REPS=2500 ~= 150k tok (for 1M memory stress)

fire() {
  local id="$1" i=0
  while [ ! -f "$STOP" ]; do
    i=$((i+1))
    local body="${FILLERS[$((RANDOM % ${#FILLERS[@]}))]}"
    local ctx=""; [ $((i % 3)) -eq 0 ] && ctx="$LONGCTX"$'\n'
    curl -s --max-time 240 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "$(python3 -c "import json;print(json.dumps({'model':'$MODEL','messages':[{'role':'user','content':'''$ctx$body (s$id i$i)'''}],'max_tokens':1200,'temperature':0.7,'chat_template_kwargs':{'enable_thinking':True}}))")" \
      >/dev/null 2>&1 || true
  done
}

echo "[stress] $CONC streams, up to ${DURATION}s (label=$LABEL). gen0=$(gen_counter) freeze-threshold=$((FREEZE_MIN*20))s"
for s in $(seq 1 "$CONC"); do fire "$s" & done
FIRE_PIDS=$(jobs -p)
frozen=0; last=$(gen_counter); ticks=0
while [ "$ticks" -lt $((DURATION/20)) ]; do
  sleep 20; ticks=$((ticks+1))
  now=$(gen_counter); p=$(curl -s -o /dev/null -w '%{http_code}' -m 25 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"OK\"}],\"max_tokens\":6,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" 2>/dev/null)
  if [ "${now:-0}" = "${last:-0}" ] && [ "$p" != "200" ]; then frozen=$((frozen+1)); else frozen=0; fi
  [ $((ticks % 3)) -eq 0 ] && echo "[stress] t=$((ticks*20))s gen=$now probe=$p frozen=${frozen}/${FREEZE_MIN}"
  if [ "$frozen" -ge "$FREEZE_MIN" ]; then
    echo "[stress] gen frozen + probe failing for $((frozen*20))s — confirming with 2-snapshot..."
    "$CAP" "${LABEL}-A" >/dev/null 2>&1; sleep 8; "$CAP" "${LABEL}-B" >/dev/null 2>&1
    stuck=0; D="$HOME/cyber-watchdog/wedge-forensics"
    for r in 0 1 2 3 4 5 6 7; do diff "$D/${LABEL}-A/rank$r.txt" "$D/${LABEL}-B/rank$r.txt" >/dev/null 2>&1 && stuck=$((stuck+1)); done
    if [ "$stuck" -ge 6 ]; then echo "[stress] *** PERMANENT WEDGE CONFIRMED ($stuck/8 ranks stuck-identical) ***"; touch "$STOP"; break
    else echo "[stress] not a permanent wedge ($stuck/8 stuck) — transient saturation, continuing"; frozen=0; fi
  fi
  last="$now"
done
touch "$STOP"; kill $FIRE_PIDS 2>/dev/null || true; wait 2>/dev/null || true
g=$(gen_counter); p=$(curl -s -o /dev/null -w '%{http_code}' -m 30 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"OK\"}],\"max_tokens\":5}" 2>/dev/null)
echo "[stress] done. final gen=$g post-probe=$p -> $([ "$p" = "200" ] && echo HEALTHY || echo WEDGED)"
