# RUNTIME_BACKUP.md

Definitive runtime backup and restore document for the deployed cyber model `glm-5.2-nvfp4-cyber`.

Observed live over SSH (read-only) on 2026-07-01. Head = `spark-a175` / `192.168.88.101` (RDMA `10.0.0.11`). All secrets redacted.

---

## 1. Deployed state at a glance

| Property | Value |
|---|---|
| Served model name | `glm-5.2-nvfp4-cyber` |
| Base weights | `/models/GLM-5.2-NVFP4` (full `nvidia/GLM-5.2-NVFP4`, NVFP4, no prune, 433 GB) |
| Container | `vllm-glm52-cyber` |
| Image | `vllm-node-tf5-glm52:base` (id `137ffa76b425`, `sha256:137ffa76b4254dda2a9a234f721c985edd3f956c04bff909ade65d8cb30b35c8`) |
| Topology | TP=8, PP=1, 8 nodes, `mp` executor, master over RDMA `10.0.0.11:29555` |
| Context | 512k (`--max-model-len 524288`), non-DCP path (~22.5 tok/s) |
| KV cache | `fp8_ds_mla`, block-size 256 |
| Speculative | MTP k=3 (`method=mtp`, `attention_backend=FLASHMLA_SPARSE`) |
| Ablation band (live) | `mode=subspace alpha=2.0 n_dirs=12 layer_lo=25 layer_hi=70` (matches `AGENT_ROBUST_CHECKPOINT.md`) |
| Served direction file | `/models/cyber/cyber_direction.pt` (but `mode=subspace` uses `R` from the subspace — see §2) |
| Health | `curl http://127.0.0.1:8000/health` → `200`; `/v1/models` → single id `glm-5.2-nvfp4-cyber` (`root=/models/GLM-5.2-NVFP4`, `max_model_len=524288`) |

Runtime versions inside the container: **torch** `2.11.0+cu130`, **transformers** `5.12.1`, **vllm** `20260613.dev242+gab6660699.d20260627` (vLLM commit `ab66606`, built 2026-06-27).

Nodes (head runs rank 0 locally and cannot ssh itself):

| Mgmt IP | Hostname | RDMA | Role |
|---|---|---|---|
| 192.168.88.101 | spark-a175 | 10.0.0.11 | HEAD (rank 0, local) |
| 192.168.88.102 | — | 10.0.0.12 | worker |
| 192.168.88.103 | — | 10.0.0.13 | worker |
| 192.168.88.104 | — | 10.0.0.14 | worker |
| 192.168.88.105 | — | 10.0.0.15 | worker (confirmed live, node-rank 4) |
| 192.168.88.106 | — | 10.0.0.16 | worker |
| 192.168.88.107 | — | 10.0.0.17 | worker |
| 192.168.88.108 | — | 10.0.0.18 | worker |

---

## 2. Exact runtime

### 2.1 Live `vllm serve` command (rank 0, head, PID 74)

```
/usr/bin/python3 /usr/local/bin/vllm serve /models/GLM-5.2-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --served-model-name glm-5.2-nvfp4-cyber \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --nnodes 8 --node-rank 0 \
  --master-addr 10.0.0.11 --master-port 29555 \
  --distributed-executor-backend mp \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len 524288 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.82 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --enforce-eager \
  --speculative-config {"method":"mtp","num_speculative_tokens":3,"attention_backend":"FLASHMLA_SPARSE"} \
  --hf-overrides {"qk_rope_head_dim":64}
```

Workers run the identical command with `--node-rank N` (worker .105 confirmed running `--node-rank 4`).

Key facts: TP=8 / PP=1 over 8 nodes; master over RDMA `10.0.0.11:29555`, `mp` executor. 512k context (non-DCP, fastest path). 4 seqs, 4096 batched tokens, gpu-mem-util `0.82`. KV cache `fp8_ds_mla`, block-size 256. MTP speculative k=3. `--enforce-eager` on. Parsers `glm45` (reasoning) / `glm47` (tool-call). `--hf-overrides qk_rope_head_dim=64` is the head_dim alias-collision fix (5.2 NVFP4 dies at load with "704 exceeds 576" without it).

### 2.2 Live `ablate_config.json` — `/home/blacksheeep/models/cyber/ablate_config.json`

```json
{"enabled": true, "direction_file": "/models/cyber/cyber_direction.pt", "alpha": 2.0, "layer": "auto", "class_name": "DeepseekV2DecoderLayer", "mode": "subspace", "n_dirs": 12, "gate_k": 1.0, "layer_lo": 25, "layer_hi": 70}
```

sha256 = `30c9152e77da3e0aa63811bb91ae6c3f5b75f09645c89e5cfd3c9cdb8d1fd0b0`

**Verified byte-identical on all 8 nodes** — head (.101) and every worker (.102–.108) returned the identical sha256 `30c9152…d1fd0b0`.

Notes on the live config vs. the context brief:
- `layer` is `"auto"` (not a fixed layer index).
- `direction_file` is `cyber_direction.pt`, not `cyber_subspace.pt`. In `mode=subspace` the hook math uses only the matrix `R` from the subspace; `direction_file` is `torch.load`ed to populate `directions`/`best_layers` but those are not used for the subspace math. So the served ablation is driven by the subspace file, and `cyber_direction.pt` is a valid loadable container that does not change the deployed behavior.
- Band `25-70` matches `AGENT_ROBUST_CHECKPOINT.md`. Repo `README.md` band `30-65` is **stale** — the deployed band is `25-70`.

### 2.3 How the ablation hook is armed — `~/vllm-glm52/runtime/glm52-sparse-patches-cyber.sh`

This combined patch is generated at launch by `05_launch_cyber_model.sh` (= base prod patch `glm52-sparse-patches.sh` + a cyber arming tail). It is idempotent, `set -euo pipefail`, and runs once inside every container before `vllm serve`. It performs the standard GLM-5.2 GB10/sm_121 fixes and then arms the hook:

- **STEP A — `glm52-sm12x-sparse`** (validated vs vLLM `ab66606`): creates the `deepseek_v4_ops/__init__.py` package marker for the runtime-mounted sm12x Triton kernels; anchored-rewrites `vllm/utils/deep_gemm.py` so on capability-family 120 (GB10) the three DeepGEMM-only ops (`fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits`, `tf32_hc_prenorm_gemm`) short-circuit to the sm12x Triton fallbacks before the `_missing()` gate; patches `sparse_attn_indexer.py` so `SparseAttnIndexer.__init__` no longer requires `has_deep_gemm()` on fam-120. **A4** reroutes fam-120 decode top-k off `persistent_topk` to `top_k_per_row_decode` (keeps `>~397k max_model_len` from crashing). Sentinel-guarded, FATAL on any missing anchor, `py_compile`-checked.
- **STEP B — `glm52-b12x-sparse`:** `pip install --no-deps b12x==0.23.0`, probes the real decode imports, and writes `/workspace/glm52-b12x.env` with `GLM52_B12X_MLA` + `GLM52_CUDAGRAPH_MODE` (FULL if import OK, PIECEWISE if not) for the launcher to source. fused_indexer score-mode patch intentionally omitted.
- **STEP C — `glm52-nvfp4-moe-backend`:** patches `oracle/nvfp4.py` to strip FlashInfer-CUTLASS + `VLLM_CUTLASS` NVFP4 MoE backends on fam-120 so auto-select lands on **MARLIN** (FlashInfer/CUTLASS SM120 paths CUDA-IMA or produce garbage). Leaves the in-checkpoint MTP MoE oracle untouched.

**Ablation-arming tail (verbatim — the part that arms the hook):**

```bash
# ---- cyber-ablation arming: inject ensure_hooks into DeepseekV2Model.forward (reliable, in-place) ----
SP=$(python3 -c "import site;print(site.getsitepackages()[0])")
cp /models/cyber/ablation_patch.py "$SP/ablation_patch.py"
python3 - <<'PYEOF'
import os
V=os.popen("python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))'").read().strip()
F=os.path.join(V,"model_executor/models/deepseek_v2.py")
src=open(F).read()
if "_ab.ensure_hooks" in src:
    print("[cyber-ablation] ensure_hooks already injected"); raise SystemExit
cls=src.index("class DeepseekV2Model")
i=src.index("for idx, layer in enumerate(", cls)
ls=src.rfind("\n",0,i)+1
indent=src[ls:i]
inj=("%stry:\n%s    import ablation_patch as _ab; _ab.ensure_hooks(self)\n%sexcept Exception:\n%s    pass\n"
     % (indent,indent,indent,indent))
open(F,"w").write(src[:ls]+inj+src[ls:])
print("[cyber-ablation] injected ensure_hooks into DeepseekV2Model.forward (%s)" % F)
PYEOF
```

The tail copies `/models/cyber/ablation_patch.py` into site-packages, then in-place injects `try: import ablation_patch as _ab; _ab.ensure_hooks(self)` into vLLM's `deepseek_v2.py` `DeepseekV2Model.forward`, just before the `for idx, layer in enumerate(...)` loop. Idempotent via the `_ab.ensure_hooks` sentinel. **The live in-container module is named `ablation_patch.py`** (staged from `/models/cyber/ablation_patch.py`); the repo source is `scripts/04_ablation_patch.py`.

Successful arming shows in container logs as `[cyber-ablation] subspace loaded: R(...)` and `hooked N DeepseekV2DecoderLayer ... mode=subspace`. Setting `alpha=0` gives the pristine baseline (reversible, live).

---

## 3. Artifact inventory

Head direction dir: `/home/blacksheeep/glm52-cyber-ablation/direction/`. Repo dir: `remote/glm52-cyber-ablation/direction/`.

### 3.1 Direction / subspace files (the load-bearing backup targets)

| File | Size (bytes) | sha256 | In git? | Head↔repo | Role |
|---|---|---|---|---|---|
| `cyber_subspace.pt` | 434421 | `1baa22029c957e16d687b45d95799e8dae48a7b49cf5dc5be77324d92fc4650f` | ✅ committed | ✅ identical | **Served / load-bearing** — SVD `R` matrix the deployed `mode=subspace` hook uses (layer 43, d=6144) |
| `cyber_subspace_prompt.pt` | 434421 | `1baa22029c957e16d687b45d95799e8dae48a7b49cf5dc5be77324d92fc4650f` | ❌ not in repo | — | Backup, byte-identical to served `cyber_subspace.pt` (content covered by the committed served copy) |
| `cyber_subspace_reason.pt` | 433061 | `872398f93d3cf0cdf993b9d9a9e6136aab82b0f28484b4079aa7967349308d6a` | ✅ committed | ✅ identical | Reasoning-capture experiment variant (not served) |
| `cyber_direction.pt` | 5823083 | `88bde5aac77e6c2d02eb1e685e2bb0231f72d574aa9b77253115ab14ae90399a` | ✅ committed (stale) | ⚠️ **DRIFT** | Single-direction container; named as `direction_file` but not used for the deployed subspace math |

**`cyber_direction.pt` drift:** git blob is an older revision — sha `1495273…168c4`, 3,883,607 bytes (Jun 30 10:52) vs. head/served sha `88bde5aa…0399a`, 5,823,083 bytes (Jun 30 13:58). This is **not load-bearing** for the deployed `mode=subspace` config (git's copy loads fine as a `directions`/`best_layers` container). Only matters if you switch to a single-direction mode (`full`/`general`/`cyber_only`) — then restore the served `88bde5aa…` copy or regenerate. The served subspace files match git exactly, so the live-served artifact is fully backed up.

### 3.2 Captured activations (regenerable, NOT committed)

First-pass prompt-position acts — `/home/blacksheeep/glm52-cyber-ablation/direction/`, root-owned (~3.2 GB total):

| File | Size (bytes) | sha256 |
|---|---|---|
| `acts_cyber_harmful.pt` | 906729937 | `9fd61548…` |
| `acts_harmless.pt` | 981489709 | `a9fe882a…` |
| `acts_noncyber_harmful.pt` | 1320786701 | `c3d8aadd…` |

Reasoning-position acts — `/home/blacksheeep/models/cyber/reason/` (~384 MB total, root-owned):

| File | Size (bytes) | sha256 |
|---|---|---|
| `acts_cyber_harmful.pt` | 134207889 | `a975d6ab43a80c0e9ec213caf3591840e8636b80254e222547815a8194111be5` |
| `acts_harmless.pt` | 134207405 | `ffe7be40fdb8a61bf15bdba3ebac392ffef510e7891b776a88ce4596cbf2a453` |
| `acts_noncyber_harmful.pt` | 134208205 | `2da82db97b49ecbecdb049d242aa587816a2fe6e22d7dd372188167ac9475542` |

(Report B saw the reason dir populated on the head; Report D noted it as cleared during its pass. Either way these acts are regenerable via `13_capture_reasoning.py` / `14_launch_capture_reason.sh` and are not committed.)

### 3.3 Scripts (all committed) — `remote/glm52-cyber-ablation/scripts/`

`00_fetch_datasets.sh`, `01_consolidate_buckets.py`, `02_extract_activations.py`, `02_launch_capture.sh`, `03_compute_direction.py`, `04_ablation_patch.py`, `05_launch_cyber_model.sh`, `06_evaluate_refusal.py`, `07_sweep_alpha.sh`, `08_generate_lora_data.py`, `09_generate_quality_mix.py`, `10_finalize_dataset.py`, `11_compute_subspace.py`, `12_sweep_agentrobust.py`, `13_capture_reasoning.py`, `14_launch_capture_reason.sh`, `15_validate.py`, `16_sweep_strong.py`, `17_band_battery.py`, `18_reason_sweep.py`, `capture_ext.py`, `capture_patch.py`, `capture_reason.py`, `fetch_harmless.py`, `make_custom_synthetic.py`, `normalize_cyberseceval.py`, `set_alpha.sh`, `set_cfg.sh`.

### 3.4 Data footprint

| Path | Head | Workstation repo | Committed? |
|---|---|---|---|
| `data/raw` | 52M | MISSING | ❌ not committed (regen via `00_fetch_datasets.sh`) |
| `data/normalized` | 776K | 772K | ✅ all 7 `*.jsonl` committed |
| `data/prepared` | 768K | 764K | ✅ all 6 `*.jsonl` + `manifest.json` committed |
| `data/lora/*.jsonl` | present | only `dataset_info.json` | ❌ generated (contains working exploit code); regen via 08/09/10 |

Also committed under `data/`: `SOURCES_provenance.json`, `lora/dataset_info.json`.

**`.gitignore` caveat (backup integrity):** the ignore rules for `direction/acts_*.pt`, `data/raw`, and `data/lora/*.jsonl` carry inline `#` comments on the same line, so git treats the entire line as the pattern and `git check-ignore` matches nothing for them. Those large files are absent from git only because they were never `git add`ed — **not** because they are actively ignored. If such a file ever lands in the tree, git will NOT auto-ignore it. Effective working ignore rules today: `**/__pycache__/`, `*.pyc`, `eval/*.json`.

---

## 4. Client wiring (all secrets REDACTED)

### 4.1 AnythingLLM cyber workspace

From `Workspace.where({slug:"glm-5-dot-2-cyber"})` inside the `anythingllm` container (proxy VM `debian@86.119.45.254`):

| Field | Value |
|---|---|
| name | `GLM-5.2 Cyber` |
| slug | `glm-5-dot-2-cyber` |
| chatModel | `glm-5.2-nvfp4-cyber` |
| chatProvider | `generic-openai` |
| chatMode | `automatic` |
| hasPrompt | `true` (system prompt set) |

Instance-level GENERIC_OPEN_AI provider env (`docker exec anythingllm printenv`):

```
GENERIC_OPEN_AI_BASE_PATH=http://host.docker.internal:18081/v1
GENERIC_OPEN_AI_MODEL_PREF=mimo-v2.5-nvfp4
GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT=200000
GENERIC_OPEN_AI_API_KEY=<REDACTED>   (sk-sparks-…)
```

The instance default `GENERIC_OPEN_AI_MODEL_PREF` is `mimo-v2.5-nvfp4`, but the **workspace `chatModel=glm-5.2-nvfp4-cyber` overrides it** — the cyber workspace uses the cyber model. Base path points at the reasoning proxy on the container host (`http://host.docker.internal:18081/v1` = `172.17.0.1:18081`). Token limit 200000.

### 4.2 OpenCode — `/home/blacksheeep/.config/opencode/opencode.json`

Provider `cybersec-fhnw-vllm`:
- id `cybersec-fhnw-vllm`, name `CyberSec FHNW vLLM`, npm `@ai-sdk/openai-compatible`
- baseURL `https://llm.cybersec-fhnw.org/v1`
- apiKey `<REDACTED>` (`sk-sparks-…`)
- `timeout: false`, `chunkTimeout: 1800000`

Model entry `glm-5.2-nvfp4-cyber`: id `glm-5.2-nvfp4-cyber`, name `GLM 5.2 Cyber`, family `glm`, `reasoning: true`, `tool_call: true`, `temperature: true`, `limit.context 1048576` (1M), `limit.output 65536`, `options.chat_template_kwargs.thinking: true`.

### 4.3 Proxy chain

Public entrypoint for clients: **`https://llm.cybersec-fhnw.org/v1`**

```
AnythingLLM → reasoning proxy 172.17.0.1:18081 (host.docker.internal:18081)
OpenCode    → public HTTPS https://llm.cybersec-fhnw.org/v1
                                   ↓
                           public proxy (nginx + keepalive) → keepalive :18080
                                   ↓
                           head :8000 (container vllm-glm52-cyber, model glm-5.2-nvfp4-cyber, TP=8)
```

- OpenCode enters at the public HTTPS endpoint (Bearer `sk-sparks-…`, redacted).
- AnythingLLM enters one hop earlier via the reasoning wrapper at `172.17.0.1:18081`, which fronts the keepalive proxy.
- Keepalive proxy: `:18080 → :8000` (head).
- Verified directly: OpenCode baseURL, AnythingLLM base path (`18081`), workspace/model mapping. The public nginx→keepalive `:18080` hop and the `18081→18080→8000` ordering are per documented topology (not re-verified on the proxy VM in this read-only pass).

---

## 5. Full restore procedure

### 5.1 Prerequisites NOT in the repo (restore these first if missing)

**(a) Model weights — `/home/blacksheeep/models/GLM-5.2-NVFP4`**
- 433 GB, 47 safetensors shards + `config.json`, `generation_config.json`, `hf_quant_config.json`, `chat_template.jinja`. Too large for git.
- Present on all 8 nodes (each rank reads its local `/models` mount).
- **Source:** HF `nvidia/GLM-5.2-NVFP4` (full NVFP4, no prune). Fastest restore = copy from a surviving node over the 200G RDMA fabric; cold restore = re-download from HF using `hf_transfer` with Xet OFF (~42 MB/s reliable; per MEMORY `hf-xet-download-throttle`).

**(b) Docker image — `vllm-node-tf5-glm52:base`**
- id `137ffa76b425`, ~19 GB, `sha256:137ffa76b4254dda2a9a234f721c985edd3f956c04bff909ade65d8cb30b35c8`. Confirmed on all 8 nodes.
- Not stored as a binary in git, but the **build recipe is vendored:** `remote/glm52-gb10/image/spark-vllm-docker/` (eugr @ `cf0d5f6`, wheel cache stripped) + `remote/glm52-gb10/image/BUILD-IMAGE.md`.
- Build: `eugr/spark-vllm-docker` `build-and-copy.sh --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 --tf5` (vLLM `ab66606`, post-0.23.0, `GlmMoeDsa` + indexer + MTP). The image is a **vanilla eugr build** — none of the GB10/sm_121 fixes are baked in; all customization is runtime-mounted.
- Distribute: `remote/glm52-gb10/distribute-image.sh` (`docker save | rsync RDMA | docker load`).

**(c) NCCL 2.30.4 `LD_PRELOAD` lib**
- `/home/blacksheeep/models/nccl-2.30.4/libnccl.so.2` (248,786,184 bytes, sha `0bf24802…`). Not in git (binary).
- Regenerate: `pip download nvidia-nccl-cu13==2.30.4` → take `nvidia/nccl/lib/libnccl.so.2`. Provenance pinned in `remote/glm52-gb10/EXTERNAL-DEPS.md`.

**(d) Base runtime — `~/vllm-glm52/runtime/` (mostly captured in `remote/glm52-gb10/`)**
- `launch.sh` ← `remote/glm52-gb10/start_glm52_8node.sh` (tracked).
- `glm52-sparse-patches.sh` (base prod patch) ← tracked at `remote/glm52-gb10/glm52-sparse-patches.sh`.
- `glm52-sparse-patches-dcp.sh` (dcp-512k/dcp-1m base) ← tracked in `remote/glm52-gb10/`.
- `start_glm52_config.sh` ← tracked.
- glm-triton kernels (`~/glm-triton`, 272K; DCP variant `~/glm-triton-dcp`) ← tracked as `remote/glm52-gb10/kernels/*.py` (9 files).
- b12x wheel ← vendored at `remote/glm52-gb10/wheels/b12x-0.23.0-py3-none-any.whl` (also staged at `~/models/wheels/`; eager prod doesn't use it — only matters with cudagraph).
- The **cyber overlay** (`glm52-sparse-patches-cyber.sh`) is **generated at launch by `05`** from `glm52-sparse-patches.sh` + the arming snippet — no need to restore it separately.

### 5.2 Normal restore (direction artifact survives in git — the common case)

The deployed `mode=subspace` hook uses matrix `R` from `cyber_subspace.pt`, which is in git and byte-identical to the served/staged copy — so the load-bearing artifact is fully recoverable from the repo.

1. Ensure the repo checkout is on the head at `~/glm52-cyber-ablation/` (scripts + `direction/cyber_subspace.pt` + `direction/cyber_direction.pt`).
2. Launch (gated; stops prior model containers incl. `model-router` which holds `:8000`):
   ```bash
   CONFIRM_CYBER=YES ~/glm52-cyber-ablation/scripts/05_launch_cyber_model.sh prod
   ```
   `05` stages `04_ablation_patch.py` + `cyber_direction.pt` + `cyber_subspace.pt` + `ablate_config.json` to `/models/cyber/` on all 8 nodes, builds `glm52-sparse-patches-cyber.sh` (= base prod patch + arming snippet), distributes it, then execs `launch.sh` with `SERVED_MODEL_NAME=glm-5.2-nvfp4-cyber`, `CONTAINER=vllm-glm52-cyber`, `GPU_MEMORY_UTILIZATION=0.82`, `MAX_MODEL_LEN=524288` (`prod`/non-DCP-512k, ~22.5 tok/s).
3. Pin the exact deployed band/config on all 8 nodes (05 defaults already write these):
   ```bash
   ~/glm52-cyber-ablation/scripts/set_cfg.sh mode=subspace alpha=2.0 n_dirs=12 layer_lo=25 layer_hi=70
   ```
4. Verify:
   ```bash
   curl -s http://192.168.88.101:8000/v1/models              # → glm-5.2-nvfp4-cyber
   curl -s -o /dev/null -w '%{http_code}\n' http://192.168.88.101:8000/health   # → 200
   ```
   Container logs should show `[cyber-ablation] subspace loaded: R(...)` and `hooked N DeepseekV2DecoderLayer ... mode=subspace`. Confirm `ablate_config.json` sha256 = `30c9152…d1fd0b0` on all 8 nodes. `alpha=0` = pristine baseline (reversible, live).

### 5.3 Regenerate the direction from scratch (only if the direction artifact is lost)

The prepared bucket JSONLs are committed, so the direction is regenerable **without** re-fetching raw datasets (00 only needed if the normalized/prepared buckets are also lost).

1. **(00, only if prepared buckets also lost)** `00_fetch_datasets.sh` → re-fetch/normalize benchmarks into `data/normalized` + `data/prepared`. Skip if `data/prepared/*.jsonl` is intact (it is committed).
2. **(02)** Bring the base model up with the capture hook, then capture 3-bucket prompt-position activations:
   ```bash
   CONFIRM_GLM52=YES ~/glm52-cyber-ablation/scripts/02_launch_capture.sh   # serves glm-5.2-nvfp4, container vllm-glm52-capture
   python3 ~/glm52-cyber-ablation/scripts/02_extract_activations.py        # → direction/acts_{cyber_harmful,noncyber_harmful,harmless}.pt
   ```
   (First request JIT-compiles ~3 min — normal, not a wedge.)
3. **(03)** `python3 03_compute_direction.py` → `direction/cyber_direction.pt` (`directions`/`directions_full`/`r_general`/`best_layers`).
4. **(11)** `python3 11_compute_subspace.py --k 16` → `direction/cyber_subspace.pt` (SVD of cyber-vs-harmless acts, layer 43, d=6144; `R` = the ablation matrix the deployed hook uses).
5. Then run the normal launch (§5.2: `05 prod` + `set_cfg L25-70 a2.0`).

Reasoning-position acts (if needed for the reasoning variant) regenerate via `13_capture_reasoning.py` / `14_launch_capture_reason.sh` → `reason/acts_*.pt`.

---

## 6. Committed vs. cluster-only

**In git (`remote/glm52-cyber-ablation/` unless noted):**
- All scripts `00–18` + `capture_*`, `fetch_harmless.py`, `make_custom_synthetic.py`, `normalize_cyberseceval.py`, `set_alpha.sh`, `set_cfg.sh` (incl. the load-bearing `04_ablation_patch.py`, `05_launch_cyber_model.sh`, `11_compute_subspace.py`, `13_capture_reasoning.py`, `14_launch_capture_reason.sh`).
- Served **subspace** artifact `direction/cyber_subspace.pt` (matches live exactly) + `direction/cyber_subspace_reason.pt`.
- `direction/cyber_direction.pt` — committed but **stale** (older revision than the served copy; not load-bearing for the deployed subspace config).
- `data/normalized/*.jsonl` (7), `data/prepared/*.jsonl` (6) + `manifest.json`, `SOURCES_provenance.json`, `data/lora/dataset_info.json`, `config/cyber_system_prompt.txt`, docs (`RESULTS.md`, `AGENT_ROBUST_CHECKPOINT.md`, FINDINGS/PLAN/README/checkpoints).
- Base GLM-5.2 runtime under `remote/glm52-gb10/`: `start_glm52_8node.sh` (`launch.sh`), `glm52-sparse-patches.sh`, `glm52-sparse-patches-dcp.sh`, `start_glm52_config.sh`, `kernels/*.py` (9), `wheels/b12x-0.23.0-py3-none-any.whl`, image build recipe (`image/`), `EXTERNAL-DEPS.md`, `distribute-image.sh`.

**Cluster-only (NOT in git) — with how to restore each:**
- **Model weights** `/home/blacksheeep/models/GLM-5.2-NVFP4` (433 GB) — copy over RDMA from a surviving node, or re-download `nvidia/GLM-5.2-NVFP4` from HF.
- **Docker image** `vllm-node-tf5-glm52:base` (~19 GB) — rebuild from vendored recipe (`--vllm-ref ab6660… --tf5`) and `distribute-image.sh`.
- **NCCL 2.30.4** `~/models/nccl-2.30.4/libnccl.so.2` — `pip download nvidia-nccl-cu13==2.30.4`.
- **Captured activations** `direction/acts_*.pt` (~3.2 GB) and `reason/acts_*.pt` (~384 MB) — regenerable via 02 / 13.
- **`data/raw`** (52M) — regen via `00_fetch_datasets.sh`.
- **`data/lora/*.jsonl`** — regen via 08/09/10 (contains working exploit code).
- **`cyber_subspace_prompt.pt`** — head-only, but byte-identical to the committed served `cyber_subspace.pt` (content covered).
- **Served `cyber_direction.pt`** (`88bde5aa…`, 5,823,083 B) — head-only current revision; only needed for single-direction modes (git copy is stale). Regenerate via 03 or copy from a surviving node.

**Bottom line:** the complete reproducible delta — launcher, ablation hook, deployed **subspace** artifact, band/alpha config, GB10 patches, and triton kernels — is fully in git. Only three items are external binaries (model weights, base docker image, NCCL 2.30.4), each documented with regeneration provenance and each redundantly present on all 8 nodes. The one non-load-bearing gap is that git's `cyber_direction.pt` is an older revision than the currently-served copy — harmless for the deployed subspace config, but restore/regenerate it if switching to a single-direction mode.
