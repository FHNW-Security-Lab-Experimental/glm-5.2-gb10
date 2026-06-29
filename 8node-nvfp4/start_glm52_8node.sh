#!/usr/bin/env bash
set -uo pipefail
#
# launch.sh — start the FULL nvidia/GLM-5.2-NVFP4 (model_type glm_moe_dsa) across
# ALL EIGHT DGX Spark GB10 nodes (sm_121a, aarch64) with TP=8, native sm_121 DSA
# sparse-MLA + lightning indexer, and IN-CHECKPOINT MTP speculative decode.
#
# Ported from CosmicRaisins/glm-5.2-gb10/launch.sh (proven on 4 nodes, TP=4, AWQ
# 15%-pruned + a SEPARATE reconstructed INT4 MTP draft). The differences here:
#
#   * FULL nvidia/GLM-5.2-NVFP4 (NOT the AWQ 15%-prune). ~58 GB weights/node, fits
#     8 nodes, so NO prune. Weights live on every node at
#     $MODEL_HOST_PATH (=/home/blacksheeep/models/GLM-5.2-NVFP4).
#   * The NVFP4 export SHIPS the layer-78 MTP draft head (791 tensors, eh_proj,
#     num_nextn_predict_layers=1, bf16 draft) — so MTP is IN-CHECKPOINT. The
#     speculative model is the SAME served path, method "mtp" (NOT a separate
#     draft dir like CosmicRaisins' "glm52-mtp-int4-aligned"; NOT deepseek_mtp).
#   * 8 nodes, TP=8 / PP=1 (NOT 4-node TP=4). Our head's RDMA iface + HCA DIFFER
#     from the workers — set PER NODE (head enp1s0f0np0 / rocep1s0f0, workers
#     enP2p1s0f0np0 / roceP2p1s0f0). Never collapse onto one value.
#   * The head CANNOT ssh itself: rank 0 runs LOCALLY (bash -c); ranks 1..7 over
#     ssh, dispatched on the MGMT network (192.168.88.x) but vLLM rendezvous +
#     all NCCL/Gloo/TP traffic ride the RDMA rail (MASTER_ADDR=10.0.0.11).
#   * The two NON-VENDORED CosmicRaisins mods (glm52-sm12x-sparse + glm52-b12x-
#     sparse) are applied as a RUNTIME patch script INSIDE each container at start,
#     BEFORE `vllm serve` — not baked — for iterability. The script is mounted RO
#     from $PATCH_SCRIPT into the container and the entrypoint is wrapped so it
#     runs once, in-place, then exec's vllm serve.
#   * Triton sparse-MLA kernels are mounted RO from $KERNELS_DIR over the image's
#     vLLM tree (same KMOUNTS as upstream: mla/ + ops/deepseek_v4_ops/).
#   * NCCL 2.30.4 aarch64 is LD_PRELOADed from the /models mount
#     (/models/nccl-2.30.4/libnccl.so.2) — the shm_broadcast warmup-wedge fix.
#
# Run FROM THE HEAD node (192.168.88.101). Gated by CONFIRM_GLM52=YES because it
# STOPS the live MiMo-V2.5 (vllm-mimo) + MiniMax containers first.
#
#   CONFIRM_GLM52=YES ./launch.sh   # launch
#   ./launch.sh --dry-run           # print the docker commands without running
#   ./launch.sh --stop              # docker rm -f the container on every node
#
# License: Apache-2.0.

# ============================================================================
# CONFIG — env-var-overridable defaults (repo convention: ${VAR:-default})
# ============================================================================
IMAGE="${IMAGE:-vllm-node-tf5-glm52:base}"   # our built base; b12x+sm12x applied at start
CONTAINER="${CONTAINER:-vllm-glm52}"
RUNTIME_DIR="${RUNTIME_DIR:-/home/blacksheeep/vllm-glm52/runtime}"
LOG_DIR="${LOG_DIR:-/home/blacksheeep/vllm-glm52/logs}"
MODEL_DIR="${MODEL_DIR:-/home/blacksheeep/models}"           # mounted -> /models
MODEL_HOST_PATH="${MODEL_HOST_PATH:-$MODEL_DIR/GLM-5.2-NVFP4}"
MODEL_PATH="${MODEL_PATH:-/models/$(basename "$MODEL_HOST_PATH")}"
SSH_USER="${SSH_USER:-blacksheeep}"

# Triton sparse-MLA kernels (CosmicRaisins kernels/*.py), deployed per node here.
# Mounted RO file-by-file over the image's vLLM tree (same set as upstream KMOUNTS).
KERNELS_DIR="${KERNELS_DIR:-/home/blacksheeep/glm-triton}"

# The runtime patch script that recreates the two non-vendored CosmicRaisins mods
# (glm52-sm12x-sparse + glm52-b12x-sparse). Mounted RO into the container and run
# ONCE before `vllm serve` (see the entrypoint wrapper in start_rank). Lives in the
# runtime dir so an edit here redeploys to every node via $RUNTIME_DIR.
PATCH_SCRIPT="${PATCH_SCRIPT:-$RUNTIME_DIR/glm52-sparse-patches.sh}"

# NCCL 2.30.4 aarch64, staged at $MODEL_DIR/nccl-2.30.4/libnccl.so.2 on every node,
# so it rides the SAME /models mount -> /models/nccl-2.30.4/libnccl.so.2 inside.
NCCL_SO="${NCCL_SO:-/models/nccl-2.30.4/libnccl.so.2}"

CONFIRM_GLM52="${CONFIRM_GLM52:-}"

# ---- topology -------------------------------------------------------------
NNODES="${NNODES:-8}"
TP_SIZE="${TP_SIZE:-8}"            # full model across all 8 GB10; MTP lives in the TP group
PP_SIZE="${PP_SIZE:-1}"           # PP+MTP is unsupported (draft lacks SupportsPP) -> PP=1
MASTER_ADDR="${MASTER_ADDR:-10.0.0.11}"   # RDMA rail address of the head (rank 0)
MASTER_PORT="${MASTER_PORT:-29555}"
API_PORT="${API_PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2-nvfp4}"

# ---- engine settings (Tasks B/C/D) ----------------------------------------
# First boot at 262144 (256k) — raise to 1M only after a coherent long-ctx soak.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# fp8_ds_mla = DeepSeek self-scaled fp8 MLA KV (the CosmicRaisins recipe). Unlike
# plain fp8/auto it carries its own scales, so it is NOT the uncalibrated-salad
# trap that bf16-forced GLM dense-carveout had to avoid.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_ds_mla}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
# 4 = ciprianveg's proven 8-node first-boot value. The per-node KV POOL (not the
# seq count) is the admission limiter — 4 slots over-subscribe a ~0.5-0.8M pool
# and vLLM admits/preempts against the pool (same pattern as the MiniMax 690k
# note). Drop to 1 for the single-stream long-context (512k/1M) profile.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
# Task D: TP=8 runs FULL model depth on every node on GB10 UNIFIED memory (host RAM
# == GPU pool, ~121 GB/node). The profiling forward peak has little headroom, so
# gpu-mem-util must stay conservative for cross-node TP (a high util OOM-wedges the
# WHOLE cluster: ping-only, sshd dead, power-cycle to recover — 2026-06-17). 0.72
# is the recommended TP ceiling; the high-util guard below enforces <=0.78.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.72}"

REASONING_PARSER="${REASONING_PARSER:-glm45}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-glm47}"

# Task C: IN-CHECKPOINT MTP draft. The NVFP4 export ships the layer-78 nextn head,
# so method is "mtp" reading the SAME served checkpoint (no separate draft dir),
# and the draft attention runs the sm_121 FLASHMLA_SPARSE backend. k=3 peaked for
# CosmicRaisins (Z.ai recommends k=5) — starting point, overridable.
ENABLE_MTP="${ENABLE_MTP:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
SPEC_ATTENTION_BACKEND="${SPEC_ATTENTION_BACKEND:-FLASHMLA_SPARSE}"

# Task B: cudagraph. With b12x installed the sparse decode kernel is capture-safe,
# so FULL is the fast path. WITHOUT b12x the Triton flash_mla decode does illegal
# torch.full() under capture -> set CUDAGRAPH_MODE=PIECEWISE (or ENFORCE_EAGER=1).
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\"}}"

# GLM-5.2 config footgun: configuration_glm_moe_dsa.py declares
#   attribute_map = {"head_dim": "qk_rope_head_dim"}
# and config.json ships BOTH head_dim=192 (=qk_nope) and qk_rope_head_dim=64. During
# from_dict the head_dim=192 alias clobbers qk_rope_head_dim to 192, so vLLM builds
# the fused_qkv_a_proj kv shard as 512+192=704 vs the real 512+64=576 ->
# "start (0) + length (704) exceeds dimension size (576)" at init. Force it back to
# 64 (deepseek_v2.py recomputes qk_head_dim itself, so this is safe). JSON stays
# space-free to survive the SSH env hop.
HF_OVERRIDES="${HF_OVERRIDES:-{\"qk_rope_head_dim\":64}}"

EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

# Distributed executor backend.  mp = vLLM-native multi-node (per-rank container
# with --nnodes/--node-rank/--master-addr, the rendezvous this launcher is built
# around).  Task D's decisive 8-node evidence (ciprianveg) is that mp can FAIL to
# come up on 8 GB10 devices and that Ray succeeded — so if an `mp` boot wedges in
# the GLOO/NCCL warmup despite the NCCL 2.30.4 LD_PRELOAD + IB passthrough, switch
# to Ray:  DIST_BACKEND=ray  (and bring up the Ray head/workers FIRST — see the
# bring-up runbook; the per-rank --node-rank flags are dropped automatically for
# ray because Ray assigns ranks itself).  mp is the lower-risk first attempt now
# that the NCCL 2.30.4 shm_broadcast fix is present; ray is the proven recovery.
DIST_BACKEND="${DIST_BACKEND:-mp}"

# Optional cgroup RAM cap. On GB10 unified memory this does NOT contain GPU allocs
# (verified 2026-06-17), so it is only partial host-side protection. Empty = none.
CONTAINER_MEM_LIMIT="${CONTAINER_MEM_LIMIT:-}"

# Stop the LIVE MiMo-V2.5 (vllm-mimo) + MiniMax + every other model container first.
# model-router holds :8000 (the API port GLM binds with --network host) — it MUST be
# stopped or rank0's vllm serve dies with OSError Errno 98 Address already in use,
# and the launcher's /health poll gets a false 200 from the router's empty backend.
STOP_CONTAINERS="${STOP_CONTAINERS:-vllm-mimo sglang-minimax-m3 vllm-minimax-m3 vllm-minimax-m27 vllm-glm51 vllm-glm52 vllm-kimi-k26 vllm-ds4 vllm-ds4-022 vllm-qwen36-fp8 vllm-qwen-lb vllm-nemotron-ultra model-router}"

# Eight nodes. Head RDMA iface/HCA DIFFER from workers — do NOT normalize.
# Format: mgmt_ip:rdma_ip:node_rank:rdma_if:rdma_hca
# rdma_if is the node's OWN rail iface -> used as its GLOO/TP/UCX SOCKET_IFNAME.
NODES=(
  "192.168.88.101:10.0.0.11:0:enp1s0f0np0:rocep1s0f0"
  "192.168.88.102:10.0.0.12:1:enP2p1s0f0np0:roceP2p1s0f0"
  "192.168.88.103:10.0.0.13:2:enP2p1s0f0np0:roceP2p1s0f0"
  "192.168.88.104:10.0.0.14:3:enP2p1s0f0np0:roceP2p1s0f0"
  "192.168.88.105:10.0.0.15:4:enP2p1s0f0np0:roceP2p1s0f0"
  "192.168.88.106:10.0.0.16:5:enP2p1s0f0np0:roceP2p1s0f0"
  "192.168.88.107:10.0.0.17:6:enP2p1s0f0np0:roceP2p1s0f0"
  "192.168.88.108:10.0.0.18:7:enP2p1s0f0np0:roceP2p1s0f0"
)
# Full multi-iface list for NCCL_SOCKET_IFNAME (mgmt + both rail iface names). NCCL
# probes them; GLOO/TP/UCX get the node's OWN single rail iface (per-node lookup).
NCCL_SOCKET_IFNAME_LIST="${NCCL_SOCKET_IFNAME_LIST:-enP7s7,enp1s0f0np0,enP2p1s0f0np0}"
# RoCE/IB HCA list NCCL may bind (both head + worker HCA names; NCCL picks the
# present one per node). Per-node NCCL_IB_HCA is also set to the node's own HCA.
NCCL_IB_HCA_LIST="${NCCL_IB_HCA_LIST:-rocep1s0f0,roceP2p1s0f0}"

SSH_OPTS=(-F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=12)

# ============================================================================
say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
log()  { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

remote() {
  local host="$1"; shift
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "$@"
}

DRYRUN=0; STOP=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRYRUN=1 ;;
    --stop)    STOP=1 ;;
    *) die "unknown arg: $a (use --dry-run or --stop)" ;;
  esac
done

[ "${#NODES[@]}" -ge 1 ] || die "NODES is empty"
[[ "$NNODES" =~ ^[0-9]+$ ]] || die "NNODES must be an integer"
(( NNODES >= 1 && NNODES <= ${#NODES[@]} )) || die "NNODES must be 1..${#NODES[@]}"

# ----------------------------------------------------------------------------
# --stop: reap the container on every node (head locally, workers over ssh).
# ----------------------------------------------------------------------------
if [ "$STOP" = 1 ]; then
  say "stopping '$CONTAINER' on all ${NNODES} nodes"
  for node in "${NODES[@]}"; do
    IFS=: read -r mgmt _ rank _ _ <<<"$node"
    (( rank < NNODES )) || continue
    if [ "$rank" = 0 ]; then
      sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 && printf '   stopped on %s (head, local)\n' "$mgmt"
    else
      remote "$mgmt" "sudo docker rm -f $CONTAINER >/dev/null 2>&1" \
        && printf '   stopped on %s\n' "$mgmt"
    fi
  done
  exit 0
fi

# ----------------------------------------------------------------------------
# Triton sparse-MLA kernel mounts (bound RO over the image's vLLM tree). Same set
# as upstream CosmicRaisins KMOUNTS — paths are inside the image's vLLM install.
# ----------------------------------------------------------------------------
MLA="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla"
OPS="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/deepseek_v4_ops"

# ----------------------------------------------------------------------------
# create_container — start the idle container on ONE node, with that node's OWN
# rail iface/HCA in the per-node RDMA env. Defined as a function so it can be both
# called locally (head) and shipped over ssh to workers (declare -f).
# Args: <rdma_ip> <rail_iface> <rail_hca>
# ----------------------------------------------------------------------------
create_container() {
  local host_ip="$1"      # this node's RDMA rail IP (VLLM_HOST_IP)
  local net_if="$2"       # this node's OWN rail iface (GLOO/TP/UCX SOCKET_IFNAME)
  local hca="$3"          # this node's OWN RoCE HCA (NCCL_IB_HCA)

  mkdir -p "$LOG_DIR" "$RUNTIME_DIR"
  local stamp; stamp="$(date +%Y%m%d-%H%M%S)"
  # Rotate ONLY the live per-rank logs (glm52-rank<N>.log), never the already
  # stamped backups. The old glob (glm52-rank*.log) re-rotated backups on every
  # boot, so the name grew a stamp each time until `mv` failed with "File name
  # too long" and — under set -e — aborted the launch, losing the crash log.
  # Tolerate any mv failure (|| true) and prune to the newest 12 backups.
  for live_log in "$LOG_DIR"/glm52-rank[0-9].log "$LOG_DIR"/glm52-rank[0-9][0-9].log; do
    [[ -f "$live_log" ]] && mv "$live_log" "${live_log%.log}.${stamp}.log" 2>/dev/null || true
  done
  ls -1t "$LOG_DIR"/glm52-rank*.*.log 2>/dev/null | tail -n +13 | xargs -r rm -f 2>/dev/null || true
  [[ -d "$MODEL_HOST_PATH" ]] || { echo "Missing model dir on $(hostname): $MODEL_HOST_PATH" >&2; exit 1; }
  [[ -f "$KERNELS_DIR/sparse_mla_kernels.py" ]] || { echo "Missing Triton kernels on $(hostname): $KERNELS_DIR (run bootstrap step 3)" >&2; exit 1; }
  [[ -f "$PATCH_SCRIPT" ]] || { echo "Missing patch script on $(hostname): $PATCH_SCRIPT" >&2; exit 1; }
  [[ -f "$MODEL_DIR/nccl-2.30.4/libnccl.so.2" ]] || { echo "Missing NCCL 2.30.4 on $(hostname): $MODEL_DIR/nccl-2.30.4/libnccl.so.2" >&2; exit 1; }
  sudo docker image inspect "$IMAGE" >/dev/null

  sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  for old in $STOP_CONTAINERS; do
    [[ "$old" == "$CONTAINER" ]] && continue
    sudo docker stop "$old" >/dev/null 2>&1 || true
  done

  # `docker rm -f` returns before the driver actually frees the dead container's
  # unified memory; loading the new 465 GB model while the prior ~70 GB is still
  # held makes WorkerProc init fail under memory pressure (observed 2026-06-28:
  # back-to-back config swaps left ~70 GB held → "WorkerProc initialization
  # failed"). Wait (≤90 s, best-effort) for this node's used memory to fall back
  # to baseline before starting the new container.
  for _w in $(seq 1 45); do
    used_mb="$(free -m | awk '/^Mem:/{print $3}')"
    if (( used_mb < 16000 )); then break; fi
    sleep 2
  done

  local memcap=()
  if [[ -n "${CONTAINER_MEM_LIMIT:-}" ]]; then
    memcap=(--memory "$CONTAINER_MEM_LIMIT" --memory-swap "$CONTAINER_MEM_LIMIT")
  fi

  # Optional NCCL transport tuning — env-overridable, DEFAULT OFF (= NCCL auto, no change).
  # For the NCCL_PROTO=LL128 decode-latency trial. NCCL_PROTO is process-global, so it also
  # hits the bandwidth-bound prefill all_gatherv + TP all-reduce — A/B BOTH decode and prefill.
  local nccl_extra=()
  [[ -n "${NCCL_PROTO:-}" ]] && nccl_extra+=(-e "NCCL_PROTO=${NCCL_PROTO}")
  [[ -n "${NCCL_MIN_NCHANNELS:-}" ]] && nccl_extra+=(-e "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS}")
  [[ -n "${NCCL_MAX_NCHANNELS:-}" ]] && nccl_extra+=(-e "NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS}")
  [[ -n "${CUDA_DEVICE_MAX_CONNECTIONS:-}" ]] && nccl_extra+=(-e "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}")

  # IB passthrough is REQUIRED: without --device /dev/infiniband + IPC_LOCK +
  # memlock=-1, NCCL silently drops to TCP (~12 vs 30+ tok/s). This (with NCCL
  # 2.30.4 LD_PRELOAD) is what avoids the prior TP GLOO warmup wedge.
  sudo docker run -d \
    --name "$CONTAINER" \
    --entrypoint /bin/bash \
    --restart no \
    --network host \
    --ipc host \
    --shm-size 10gb \
    --gpus all \
    --cap-add IPC_LOCK \
    "${memcap[@]}" \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864 \
    --device /dev/infiniband:/dev/infiniband \
    -v "$MODEL_DIR:/models" \
    -v "$LOG_DIR:/workspace/logs" \
    -v "$RUNTIME_DIR:/workspace/runtime" \
    -v "$PATCH_SCRIPT:/workspace/glm52-sparse-patches.sh:ro" \
    -v "$KERNELS_DIR/sparse_mla_kernels.py:$MLA/sparse_mla_kernels.py:ro" \
    -v "$KERNELS_DIR/sparse_mla_env.py:$MLA/sparse_mla_env.py:ro" \
    -v "$KERNELS_DIR/sm12x_sparse_mla_attn.py:$MLA/sm12x_sparse_mla_attn.py:ro" \
    -v "$KERNELS_DIR/patch_flashmla_ops.py:$MLA/patch_flashmla_ops.py:ro" \
    -v "$KERNELS_DIR/flashmla_sparse.py:$MLA/flashmla_sparse.py:ro" \
    -v "$KERNELS_DIR/sm12x_deep_gemm_fallbacks.py:$OPS/sm12x_deep_gemm_fallbacks.py:ro" \
    -v "$KERNELS_DIR/sm12x_mqa.py:$OPS/sm12x_mqa.py:ro" \
    -v "$KERNELS_DIR/b12x_sparse_helpers.py:$OPS/b12x_sparse_helpers.py:ro" \
    -e VLLM_HOST_IP="$host_ip" \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}" \
    -e LD_PRELOAD="$NCCL_SO" \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
    -e GLM52_BIND_HOST_TRITON=1 \
    -e GLM52_MQA_LOGITS_TRITON=1 \
    -e GLM52_PAGED_MQA_TRITON=1 \
    -e GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192 \
    -e GLM52_B12X_MLA=1 \
    -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}" \
    -e TRITON_CACHE_DIR=/workspace/logs/.tritoncache \
    -e NCCL_NET=IB \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA="$hca" \
    -e NCCL_IB_HCA_LIST="$NCCL_IB_HCA_LIST" \
    -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME_LIST" \
    -e GLOO_SOCKET_IFNAME="$net_if" \
    -e TP_SOCKET_IFNAME="$net_if" \
    -e UCX_NET_DEVICES="$net_if" \
    -e OMPI_MCA_btl_tcp_if_include="$net_if" \
    -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_IB_ROCE_VERSION_NUM=2 \
    -e NCCL_IB_ADDR_FAMILY=AF_INET \
    -e NCCL_CROSS_NIC=1 \
    -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -e NCCL_DEBUG=WARN \
    "${nccl_extra[@]}" \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    "$IMAGE" \
    -lc 'sleep infinity' >/dev/null

  # Apply the two non-vendored CosmicRaisins mods ONCE, in-place, before serving:
  #   (1) glm52-sm12x-sparse — short-circuit fp8_fp4_mqa_logits /
  #       fp8_fp4_paged_mqa_logits / tf32_hc_prenorm_gemm to the sm12x_* fallbacks
  #       BEFORE the DeepGEMM _missing() gate; relax the SparseAttnIndexer ctor so
  #       sm_121 never requires has_deep_gemm().
  #   (2) glm52-b12x-sparse — pip install --no-deps b12x==0.23.0 + the fused_indexer
  #       score-mode patch (the capture-safe sparse decode the cudagraph FULL path
  #       needs). The script is idempotent; run.sh exits non-zero is fatal here so
  #       we don't silently serve an unpatched (TCP-slow / capture-crashing) engine.
  sudo docker exec "$CONTAINER" bash -lc 'bash /workspace/glm52-sparse-patches.sh' \
    || { echo "glm52 sparse patch FAILED on $(hostname) — refusing to serve unpatched" >&2; exit 1; }
}

# ----------------------------------------------------------------------------
# start_rank — exec `vllm serve` for ONE rank inside its (already-patched)
# container. Workers (rank>=1) start --headless; rank 0 serves the API. Defined as
# a function so it can be called locally (head) and shipped over ssh to workers.
# Arg: <node_rank>
# ----------------------------------------------------------------------------
start_rank() {
  local rank="$1"
  local log_file="/workspace/logs/glm52-rank${rank}.log"
  local headless=() eager=() spec=() compile=() hfov=() multinode=()

  # mp = vLLM-native multi-node: each rank's container is told its own
  # --nnodes/--node-rank/--master-addr and headless on rank>0. ray = Ray assigns
  # ranks itself, so the per-rank rendezvous flags must NOT be passed; instead the
  # Ray cluster (head + workers) must already be up and only rank 0 runs `vllm
  # serve` (Phase 2 handles that ordering for ray).
  if [[ "$DIST_BACKEND" == "ray" ]]; then
    multinode=()
  else
    multinode=(--nnodes "$NNODES" --node-rank "$rank" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT")
    [[ "$rank" != "0" ]] && headless=(--headless)
  fi

  # cudagraph mode: prefer the value the runtime patch wrote from a REAL b12x
  # import probe (/workspace/glm52-b12x.env: FULL if b12x decode imports, else
  # PIECEWISE). Falls back to the env default if the probe file is absent.
  local cg_mode="$CUDAGRAPH_MODE"
  if [[ -f /workspace/glm52-b12x.env ]]; then
    # shellcheck disable=SC1091
    cg_mode="$(. /workspace/glm52-b12x.env 2>/dev/null; echo "${GLM52_CUDAGRAPH_MODE:-$CUDAGRAPH_MODE}")"
  fi
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    eager=(--enforce-eager)
  else
    compile=(--compilation-config "{\"cudagraph_mode\":\"$cg_mode\"}")
  fi
  if [[ "$ENABLE_MTP" == "1" ]]; then
    # IN-CHECKPOINT MTP: method "mtp" reads the layer-78 nextn head from the SAME
    # served checkpoint (no separate draft model dir). Draft attention rides the
    # sm_121 FLASHMLA_SPARSE backend.
    spec=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS},\"attention_backend\":\"${SPEC_ATTENTION_BACKEND}\"}")
  fi
  [[ -n "${HF_OVERRIDES:-}" ]] && hfov=(--hf-overrides "$HF_OVERRIDES")

  local args=(
    vllm serve "$MODEL_PATH"
    --host 0.0.0.0
    --port "$API_PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    --tensor-parallel-size "$TP_SIZE"
    --pipeline-parallel-size "$PP_SIZE"
    "${multinode[@]}"
    "${headless[@]}"
    --distributed-executor-backend "$DIST_BACKEND"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --block-size "$BLOCK_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --reasoning-parser "$REASONING_PARSER"
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
    "${eager[@]}"
    "${spec[@]}"
    "${hfov[@]}"
    "${compile[@]}"
    $EXTRA_VLLM_ARGS
  )

  local cmd
  printf -v cmd '%q ' "${args[@]}"
  if [[ "$DRYRUN" == "1" ]]; then
    printf '   (rank %s) docker exec -d %s bash -lc "cd /workspace && exec %s> %s 2>&1"\n' \
      "$rank" "$CONTAINER" "$cmd" "$log_file"
    return 0
  fi
  sudo docker exec -d "$CONTAINER" bash -lc "cd /workspace && exec ${cmd} > '$log_file' 2>&1"
}

# ----------------------------------------------------------------------------
# Preflight gates
# ----------------------------------------------------------------------------
if [[ "$DRYRUN" != "1" && "$CONFIRM_GLM52" != "YES" ]]; then
  echo "Refusing to stop the live MiMo-V2.5 (vllm-mimo) + MiniMax runtime without CONFIRM_GLM52=YES" >&2
  exit 2
fi

# Task D OOM guard: cross-node TP runs FULL model depth on every GB10 node on
# unified memory, so a high util target during the profiling forward can over-commit
# and OOM-WEDGE the whole cluster (ping-only, sshd dead, power-cycle to recover —
# 2026-06-17 at TP=8 + MTP + util 0.85). Refuse TP>1 with util>0.78 unless opted in.
if (( TP_SIZE > 1 )); then
  util_x100="${GPU_MEMORY_UTILIZATION/0./}"; util_x100="${util_x100:-0}"
  if [[ "${ALLOW_HIGH_GPU_UTIL:-}" != "YES" ]] && (( ${util_x100%%[!0-9]*} > 78 )); then
    echo "Refusing TP_SIZE=$TP_SIZE with GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION on GB10 unified memory." >&2
    echo "Cross-node TP runs full depth/node; >0.78 util risks an OOM that wedges the whole cluster." >&2
    echo "Use GPU_MEMORY_UTILIZATION<=0.75 (recommended ~0.72) for TP, or ALLOW_HIGH_GPU_UTIL=YES to override." >&2
    exit 2
  fi
fi

say "GLM-5.2-NVFP4 launch: ${NNODES} nodes, TP=${TP_SIZE} PP=${PP_SIZE}, head=${MASTER_ADDR}:${API_PORT}, image=${IMAGE}"
log "MTP=${ENABLE_MTP} k=${NUM_SPECULATIVE_TOKENS} | KV=${KV_CACHE_DTYPE} | max-model-len=${MAX_MODEL_LEN} | util=${GPU_MEMORY_UTILIZATION} | cudagraph=$([[ "$ENFORCE_EAGER" == 1 ]] && echo eager || echo "$CUDAGRAPH_MODE")"
[ "$DRYRUN" = 1 ] && echo "   (dry-run — nothing will be executed)"

if [[ "$DRYRUN" != "1" ]]; then
  log "stopping watchdogs during GLM-5.2 run (neither targets $CONTAINER)"
  sudo systemctl stop sparks-vllm-watchdog.timer sparks-kimi-watchdog.timer >/dev/null 2>&1 || true
fi

# ----------------------------------------------------------------------------
# Phase 1 — create the idle container on every node (head local, workers ssh).
# Ranks 1..7 are brought up before rank 0 so the rendezvous master they dial has
# the workers waiting (matches the upstream "workers first" order).
# ----------------------------------------------------------------------------
for node in "${NODES[@]}"; do
  IFS=: read -r mgmt host_ip rank net_if hca <<<"$node"
  (( rank < NNODES )) || continue
  log "creating GLM-5.2 container rank ${rank} on ${mgmt} (rdma ${host_ip}, if ${net_if}, hca ${hca})"
  if [[ "$DRYRUN" == "1" ]]; then
    if [[ "$rank" == "0" ]]; then
      printf '   (head, LOCAL) create_container %q %q %q\n' "$host_ip" "$net_if" "$hca"
    else
      printf '   (worker, ssh %s@%s) create_container %q %q %q\n' "$SSH_USER" "$mgmt" "$host_ip" "$net_if" "$hca"
    fi
    continue
  fi
  if [[ "$rank" == "0" ]]; then
    create_container "$host_ip" "$net_if" "$hca"
  else
    remote "$mgmt" "IMAGE='$IMAGE' CONTAINER='$CONTAINER' RUNTIME_DIR='$RUNTIME_DIR' LOG_DIR='$LOG_DIR' MODEL_DIR='$MODEL_DIR' MODEL_HOST_PATH='$MODEL_HOST_PATH' KERNELS_DIR='$KERNELS_DIR' PATCH_SCRIPT='$PATCH_SCRIPT' NCCL_SO='$NCCL_SO' STOP_CONTAINERS='$STOP_CONTAINERS' CONTAINER_MEM_LIMIT='$CONTAINER_MEM_LIMIT' MLA='$MLA' OPS='$OPS' NCCL_SOCKET_IFNAME_LIST='$NCCL_SOCKET_IFNAME_LIST' NCCL_IB_HCA_LIST='$NCCL_IB_HCA_LIST' VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS='${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}' TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST:-12.1a}' NCCL_PROTO='${NCCL_PROTO:-}' NCCL_MIN_NCHANNELS='${NCCL_MIN_NCHANNELS:-}' NCCL_MAX_NCHANNELS='${NCCL_MAX_NCHANNELS:-}' CUDA_DEVICE_MAX_CONNECTIONS='${CUDA_DEVICE_MAX_CONNECTIONS:-}' bash -s" < <(
      declare -f create_container
      printf 'create_container %q %q %q\n' "$host_ip" "$net_if" "$hca"
    )
  fi
done

# ----------------------------------------------------------------------------
# Phase 2 — exec vllm serve in every container. Workers (headless) first, head
# (rank 0, serves the API) last.
# ----------------------------------------------------------------------------
for node in "${NODES[@]}"; do
  IFS=: read -r mgmt _ rank _ _ <<<"$node"
  (( rank < NNODES )) || continue
  [[ "$rank" == "0" ]] && continue
  log "starting GLM-5.2 rank ${rank} (worker, headless)"
  if [[ "$DRYRUN" == "1" ]]; then
    DRYRUN=1 start_rank "$rank"
    continue
  fi
  remote "$mgmt" "CONTAINER='$CONTAINER' MODEL_PATH='$MODEL_PATH' SERVED_MODEL_NAME='$SERVED_MODEL_NAME' API_PORT='$API_PORT' TP_SIZE='$TP_SIZE' PP_SIZE='$PP_SIZE' NNODES='$NNODES' MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT' MAX_MODEL_LEN='$MAX_MODEL_LEN' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' KV_CACHE_DTYPE='$KV_CACHE_DTYPE' BLOCK_SIZE='$BLOCK_SIZE' MAX_NUM_SEQS='$MAX_NUM_SEQS' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' REASONING_PARSER='$REASONING_PARSER' TOOL_CALL_PARSER='$TOOL_CALL_PARSER' ENFORCE_EAGER='$ENFORCE_EAGER' CUDAGRAPH_MODE='$CUDAGRAPH_MODE' COMPILATION_CONFIG='$COMPILATION_CONFIG' DIST_BACKEND='$DIST_BACKEND' ENABLE_MTP='$ENABLE_MTP' NUM_SPECULATIVE_TOKENS='$NUM_SPECULATIVE_TOKENS' SPEC_ATTENTION_BACKEND='$SPEC_ATTENTION_BACKEND' HF_OVERRIDES='$HF_OVERRIDES' EXTRA_VLLM_ARGS='$EXTRA_VLLM_ARGS' DRYRUN=0 bash -s" < <(
    declare -f start_rank
    printf 'start_rank %q\n' "$rank"
  )
done

log "starting GLM-5.2 rank 0 (head, serves the API)"
if [[ "$DRYRUN" == "1" ]]; then
  DRYRUN=1 start_rank 0
  say "dry-run complete — nothing was executed"
  exit 0
fi
start_rank 0

# ----------------------------------------------------------------------------
# Phase 3 — wait for /health (full 465 GB NVFP4 load across 8 nodes + cudagraph
# warmup is slow; ~12 min load + ~10 min warmup is normal).
# ----------------------------------------------------------------------------
say "launched — waiting for GLM-5.2 health (full NVFP4 load + cudagraph warmup; allow many minutes)"
echo "   logs:  sudo docker logs -f $CONTAINER   (per node)"
echo "   stop:  ./launch.sh --stop"
for _ in $(seq 1 420); do
  code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 --max-time 5 "http://127.0.0.1:${API_PORT}/health" || true)"
  if [[ "$code" == "200" ]]; then
    log "GLM-5.2 health is 200 — serving as '$SERVED_MODEL_NAME' on :${API_PORT}"
    exit 0
  fi
  if ! sudo docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    log "GLM-5.2 head container exited before health became ready (check $LOG_DIR/glm52-rank0.log)"
    exit 1
  fi
  sleep 10
done

log "GLM-5.2 did not become healthy in time"
exit 1
