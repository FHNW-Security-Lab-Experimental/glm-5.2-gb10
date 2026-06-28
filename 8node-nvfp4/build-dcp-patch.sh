#!/usr/bin/env bash
# build-dcp-patch.sh — regenerate the combined DCP in-container patch
# (glm52-sparse-patches-dcp.sh = base sparse patches + DCP items A+B+C) from the two
# committed sources, then (optionally) stage it + the DCP kernels to all 8 nodes.
#
# The launcher mounts a SINGLE patch script, so the DCP path needs base+DCP concatenated.
# This produces that artifact deterministically from glm52-sparse-patches.sh +
# glm52-dcp-patches.sh (the CONFIRM_GLM52_DCP gate is stripped — running it IS the opt-in).
#
#   ./build-dcp-patch.sh                 # just build ./glm52-sparse-patches-dcp.sh
#   ./build-dcp-patch.sh --stage         # build + scp to all 8 nodes + refresh ~/glm-triton-dcp
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# OUT + DCP_SRC are overridable so the PERF#2 variant builds from its own source into
# its own file without touching the production combined patch:
#   DCP_SRC=glm52-dcp-patches-perf2.sh OUT=glm52-sparse-patches-dcp-perf2.sh ./build-dcp-patch.sh
OUT="${OUT:-$HERE/glm52-sparse-patches-dcp.sh}"
DCP_SRC="${DCP_SRC:-$HERE/glm52-dcp-patches.sh}"

cp "$HERE/glm52-sparse-patches.sh" "$OUT"
{
  echo ""
  echo "# ===== appended by build-dcp-patch.sh: ${DCP_SRC##*/} STEP A+B+C (CONFIRM gate stripped) ====="
  sed -n '/^PYDIST=/,$p' "$DCP_SRC"
} >> "$OUT"
chmod +x "$OUT"
bash -n "$OUT"
echo "built $OUT ($(wc -l < "$OUT") lines; sha256 $(sha256sum "$OUT" | cut -c1-12))"

if [[ "${1:-}" == "--stage" ]]; then
  for ip in 101 102 103 104 105 106 107 108; do
    H="blacksheeep@192.168.88.$ip"
    scp -q -o BatchMode=yes -o ConnectTimeout=8 "$OUT" "$H:/home/blacksheeep/vllm-glm52/runtime/glm52-sparse-patches-dcp.sh"
    # ~/glm-triton-dcp = copy of ~/glm-triton with the LSE-returning flashmla_sparse.py overlaid.
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$H" 'rm -rf ~/glm-triton-dcp && cp -r ~/glm-triton ~/glm-triton-dcp'
    scp -q -o BatchMode=yes -o ConnectTimeout=8 "$HERE/kernels/flashmla_sparse.py" "$H:/home/blacksheeep/glm-triton-dcp/flashmla_sparse.py"
    echo "staged $ip"
  done
fi
