# cyber-runtime — serving + operations for `glm-5.2-nvfp4-cyber` on GB10

Runtime and operational knowledge for serving the **cyber-de-censored** GLM-5.2-NVFP4 variant on the 8-node
DGX-Spark/GB10 cluster (authorized FHNW Security Lab red-team use). This is the **runtime** side only — the
ablation itself (how the refusal directions are computed, the hook, the research) lives in the companion repo
**[`FHNW-Security-Lab-Experimental/glm52-cyber-ablation`](https://github.com/FHNW-Security-Lab-Experimental/glm52-cyber-ablation)**
(`run_ablation_pipeline.sh` to reproduce, `start_cyber.sh` to serve).

The cyber model is the **same `nvidia/GLM-5.2-NVFP4` weights** as production served under a different name with a
runtime ablation hook — i.e. it uses the exact base runtime documented in `../8node-nvfp4/` (image
`vllm-node-tf5-glm52:base`, the sm_121 sparse-MLA patches, MTP, `fp8_ds_mla` KV, NCCL 2.30.4 + IB passthrough).

## Contents
| file | what |
|---|---|
| `RUNTIME_BACKUP.md` | the **exact** live serve command + env + artifact sha256s + restore procedure (the full runtime backup) |
| **`MARLIN-MOE-DEADLOCK-FIX.md`** | **the collective-hang root cause (`moe_wna16_marlin_gemm` deadlock) + the fix `CUDA_DEVICE_MAX_CONNECTIONS=1` — validated** |
| `INCIDENT-2026-07-01.md` | postmortem: the freeze/OOMs were a **production/cyber memory conflict**, not util; full timeline + fix |
| `WEDGE-PREVENTION.md` | GB10 unified-memory wedge/OOM operations — the real cause + the correct levers (not util) |
| `tools/capture_wedge_forensics.sh` | py-spy-dumps all 8 ranks on a wedge (host-side; the rank not in a collective = the culprit) |
| `tools/stress_repro.sh` | reproduce the collective hang under load + confirm a fix (sustained-freeze + 2-snapshot detection) |
| `runtime/vllm-serve-command.txt` | the captured `vllm serve` argv |
| `runtime/ablate_config.json` | the in-container ablation config (mode/alpha/n_dirs/layer band) read by the hook |
| `runtime/glm52-sparse-patches-cyber.sh` | the combined in-container patch = base sparse patch + ablation arming |
| `tools/watchdog_cyber_cluster.sh` | **active-probe** wedge watchdog (the death/stall watchdogs miss `running==0` wedges) |
| `tools/systemd/sparks-cyber-watchdog.{service,timer}` | the systemd units (every 2 min, on the head) |

## The one operational rule (learned the hard way — see INCIDENT)
**Production (`vllm-glm52`) and cyber (`vllm-glm52-cyber`) are the same 465 GB weights (~62 GB/node) — they
cannot co-exist on 121 GB unified nodes.** `sparks-glm52-watchdog` resurrects *production* and auto-starts it on
reboot; if it is active while you run cyber, cyber OOMs (looks like a wedge or a load failure). **Only the watchdog
matching the running model may be active.** Before serving cyber:
```bash
sudo systemctl disable --now sparks-glm52-watchdog.timer                 # stop production's watchdog
sudo docker rm -f vllm-glm52; for h in 192.168.88.{102..108}; do ssh $h "sudo docker rm -f vllm-glm52"; done
```
`start_cyber.sh` (in the ablation repo) does this automatically. **Keep `gpu-memory-utilization = 0.82`** — it is
the *load floor*; lowering it OOMs the load (do not "fix" a wedge by lowering util). If cyber wedges under heavy
concurrent serving, reduce `MAX_NUM_SEQS` / `MAX_MODEL_LEN` — not util. Full detail in `WEDGE-PREVENTION.md`.

## Install the active-probe watchdog (backstop)
```bash
mkdir -p ~/cyber-watchdog && cp tools/watchdog_cyber_cluster.sh ~/cyber-watchdog/
sudo cp tools/systemd/sparks-cyber-watchdog.* /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now sparks-cyber-watchdog.timer
systemctl is-active sparks-cyber-watchdog.timer     # -> active
```
It health-gates, fires a trivial generation probe, and only relaunches (at util 0.82, stopping production first)
when the probe times out **while the generation counter is frozen** — a true wedge, not a busy long prefill.
