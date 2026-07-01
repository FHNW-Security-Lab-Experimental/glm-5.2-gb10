# Wedge / freeze prevention — glm-5.2-nvfp4-cyber

**Incident (2026-07-01):** the cyber model froze under OpenCode agentic load — GPUs pinned at ~100% on all 8
nodes, `/health`=200, `num_requests_running`=0, `generation_tokens_total` frozen, a direct trivial request
timed out. Not a crash (EngineCore + workers alive, no error logs) — a **collective hang**.

## ACTUAL root cause: PRODUCTION/CYBER memory conflict (not util)
**The real culprit — found after a long chase — was `sparks-glm52-watchdog`.** It targets the *production*
`vllm-glm52` and restarts it whenever it's down. Production and cyber are a **profile swap** (same 465 GB
weights, ~62 GB/node each) — they **cannot co-exist** (2×62 GB > 121 GB). The production watchdog kept
resurrecting `vllm-glm52` (especially auto-starting it after any reboot), so the cyber model wedged/OOM'd
because 62 GB/node was already taken. This reproduced at **both 0.72 and 0.82**, on **stale and freshly-rebooted**
heads — proving it was the conflict, not util. (Each GB10 node has 121 GB *unified* CPU+GPU memory; an alloc
failing on one rank mid-collective hangs the TP=8 all-reduce → the "GPU spinning, health=200, 0 requests, gen
frozen" wedge.) The CLAUDE.md rule is exactly this: *only the watchdog matching the running model may be active.*

**A second, opposite trap:** lowering `gpu-memory-utilization` to "fix" the wedge **starves the LOAD** — at
0.72/0.78 weights (58 G) + profiling peak + KV overflow the budget and rank 0 dies (`NVRM: Out of memory`).
**0.82 is REQUIRED to load.** Lower util does NOT prevent the wedge and DOES break loading. Wrong lever.

## Prevention (the actual fix)
1. **Before running cyber, STOP production + disable its watchdog** (this is THE fix):
   ```bash
   sudo systemctl disable --now sparks-glm52-watchdog.timer      # stop it resurrecting production
   sudo docker rm -f vllm-glm52; for h in 192.168.88.{102..108}; do ssh $h "sudo docker rm -f vllm-glm52"; done
   systemctl is-active sparks-cyber-watchdog.timer               # the CYBER watchdog should be the active one
   ```
   `start_cyber.sh` should be run only after this; a fresh reboot re-enables production's watchdog, so re-disable it.
2. **Keep `gpu-memory-utilization = 0.82`** — it's the LOAD floor. Do not lower it (starves the load, OOMs rank 0).
3. If cyber wedges under **heavy concurrent serving** (0.82 leaves ~12 GB/node for the activation peak), the
   lever is **reducing the peak, NOT util**:
   - **`MAX_NUM_SEQS=2`** (from 4): `MAX_NUM_SEQS=2 CONFIRM_CYBER=YES ~/glm52-cyber-ablation/scripts/05_launch_cyber_model.sh prod`
   - **Reduce `MAX_MODEL_LEN`** 524288→262144 (smaller KV pool → more activation headroom; costs max context).
   - **Disable MTP** — removes the speculative draft's memory + the FLASHMLA_SPARSE path (costs ~30% decode
     speed). MTP is in the launcher's speculative-config; a no-MTP variant would drop `--speculative-config`.
   - **Proxy admission control** — lower the keepalive proxy's long-context serialization threshold (≥256k→≥128k)
     and/or cap concurrent engine requests, to bound the peak upstream.

## Backstop (only if prevention still lets one through): `sparks-cyber-watchdog`
Active-probe watchdog (`tools/watchdog_cyber_cluster.sh` + `tools/systemd/sparks-cyber-watchdog.{service,timer}`,
every 2 min on the head). The existing death/stall watchdogs **miss this wedge** (they treat `running==0` as
idle). This one fires a trivial generation probe; if it **times out while the generation counter is frozen**
(a wedge, vs merely busy = counter moving), after **2 consecutive** detections it relaunches the cyber model —
**stopping production + disabling `sparks-glm52-watchdog` first**, at **util 0.82** (the load floor; flock-guarded;
warms the JIT after). Probe timeout is **200 s** so a cold-start Triton JIT (~3 min, "not a wedge") completes and
resets instead of false-triggering. It is a **backstop, not the fix** — the real prevention is stopping production
+ disabling its watchdog (see *ACTUAL root cause* above); **no util change prevents the wedge** (and lowering util
OOMs the load).

## Relaunch hygiene (a second OOM mode — load-time, not serve-time)
Cycling relaunches too fast **OOMs the fresh load**: killing a container does not instantly reclaim its
GPU/unified memory, so a new NVFP4 load started seconds later runs out of memory — rank 0 dies mid-load
(`dmesg`: `NVRM: Out of memory [NV_ERR_NO_MEMORY]`), workers hang waiting, health never reaches 200. Seen
2026-07-01 when rapidly superseding relaunches during the incident recovery.
**Rule:** between stopping and relaunching, **wait for memory reclaim** (per-node unified memory back to
~2-5%) — don't rapid-cycle. `sparks-cyber-watchdog`'s restart now polls the head's memory down to <20%
before reloading; do the same manually:
```bash
sudo docker rm -f vllm-glm52-cyber; for h in 192.168.88.{102..108}; do ssh $h "sudo docker rm -f vllm-glm52-cyber"; done
until [ "$(free -m | awk '/Mem:/{print int($3*100/$2)}')" -lt 20 ]; do sleep 5; done   # wait for reclaim
CONFIRM_CYBER=YES ~/glm52-cyber-ablation/scripts/05_launch_cyber_model.sh prod          # 05 defaults to util 0.72
```

## Verify
```bash
systemctl is-active sparks-cyber-watchdog.timer                 # active
python3 -c "import json;print(json.load(open('/home/blacksheeep/models/cyber/ablate_config.json')))"  # config
grep 'util=' ~/glm52-cyber-launch.log | tail -1                 # util=0.72
bash ~/cyber-watchdog/watchdog_cyber_cluster.sh                 # -> "probe ok — healthy"
```
