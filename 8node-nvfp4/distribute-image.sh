#!/usr/bin/env bash
# Distribute the built GLM-5.2 image from the head to the 7 workers over RDMA.
# Run detached on the head: setsid bash ~/distribute-image.sh </dev/null >~/glm52-imgdist.log 2>&1 &
set -uo pipefail
IMG="${IMG:-vllm-node-tf5-glm52:base}"
TAR=/home/blacksheeep/models/glm52img.tar
WORKERS=(10.0.0.12 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17 10.0.0.18)
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=12"
rm -f ~/.glm52_imgdist_done
echo "[$(date '+%T')] saving $IMG -> $TAR"
docker save "$IMG" -o "$TAR"
echo "[$(date '+%T')] saved $(du -h "$TAR"|cut -f1); fanning to workers"
pids=()
for ip in "${WORKERS[@]}"; do
  ( rsync -a --inplace -e "$SSH" "$TAR" "blacksheeep@$ip:/home/blacksheeep/glm52img.tar"
    $SSH "blacksheeep@$ip" "sudo docker load -i /home/blacksheeep/glm52img.tar && rm -f /home/blacksheeep/glm52img.tar"
    id=$($SSH "blacksheeep@$ip" "sudo docker image inspect $IMG --format '{{.Id}}' 2>/dev/null")
    echo "[$(date '+%T')] $ip loaded: ${id:-FAILED}" ) &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
rm -f "$TAR"
echo "[$(date '+%T')] IMAGE DISTRIBUTION DONE (fail=$fail)"
touch ~/.glm52_imgdist_done
