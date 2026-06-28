# Building the GLM-5.2 base image `vllm-node-tf5-glm52:base`

`spark-vllm-docker/` here is the build recipe, **vendored from
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) @ `cf0d5f6`** (its
upstream LICENSE is preserved in `spark-vllm-docker/LICENSE`), minus the `wheels/` build cache
and `.git`. This is what produces the base image our GLM-5.2 runtime patches/kernels mount over.

## Build command (on a GB10 builder node)

```bash
cd spark-vllm-docker
./build-and-copy.sh \
  --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 \   # pinned vLLM (post-0.23.0: GlmMoeDsa + indexer/MTP)
  --flashinfer-ref v0.6.13 \                               # pin (live image = flashinfer 0.6.13); else builds main
  --tf5 \                                                  # transformers>=5 (live image = 5.12.1) ; tags vllm-node-tf5
  -t vllm-node-tf5-glm52:base                              # final tag the launcher expects
# GPU arch is auto-detected to TORCH_CUDA_ARCH_LIST=12.1a (GB10/sm_121); Dockerfile default matches.
```

Exact versions baked into the live image (for an exact rebuild, pin these rather than `main`/latest):
**flashinfer 0.6.13**, **torch 2.11.0+cu130**, **transformers 5.12.1**, CUDA base
`nvidia/cuda:13.0.2-devel-ubuntu24.04`. (Our flashinfer fork tracks `main`; pin tag/sha for 0.6.13.)

## Reproducibility status — honest gaps (as of 2026-06-28)

**Scenario that matters: the whole cluster is gone.** What survives is GitHub (this repo + the org
forks). From GitHub alone you can reconstruct **the entire software stack**:

- code / patches / kernels / launchers / starters / b12x wheel — in this repo ✔
- the **base image** — rebuild from the vendored recipe pointed at the org forks (above), now pinned
  **exactly** to the live image (vLLM `ab66606`, flashinfer `v0.6.13`, nccl `v2.30u1`, DeepGEMM
  `nv_dev`, transformers 5.12.1, torch 2.11.0+cu130, CUDA base `13.0.2-devel-ubuntu24.04`) ✔
- NCCL 2.30.4 `.so` — build from the `nccl` fork @ `v2.30u1` (or NVIDIA), sha-verifiable ✔

**The one thing GitHub does NOT hold: the 465 GB model weights.** They live only on HF
(`nvidia/GLM-5.2-NVFP4`) and the (now-gone) nodes. If the cluster is gone, recovery depends on HF
still serving the model — **not under our control**. ⇒ This is the single real cluster-gone gap.

To close it (cluster-gone-proof): **mirror the 465 GB weights to storage we control** — an HF repo
under our org, object storage, or a NAS. (Optionally also `docker save`/GHCR-push the built image for
a faster exact restore than a from-source rebuild — nice-to-have, not required, since the image is
now exactly rebuildable.) A builder still needs network for the CUDA base + PyPI + the GitHub forks.

Notes:
- The build **clones vLLM at `ab66606` and compiles wheels** → needs network + a GB10 builder +
  time/space; it is not hermetic (pulls pip/torch/CUDA deps at build time). That's inherent to a
  from-source image build.
- Resulting image identity (verify): `sha256:137ffa76b4254dda2a9a234f721c985edd3f956c04bff909ade65d8cb30b35c8`, ~19 GB.
- Fan out to the other 7 nodes with `../distribute-image.sh` (`docker save | rsync over RDMA | docker load`).
- **Our GB10/sm_121 + DSA fixes are NOT baked into this image** — they are runtime-applied from the
  repo (`glm52-sparse-patches.sh` + `glm52-dcp-patches.sh` + `kernels/`). So the image is a vanilla
  eugr build; the whole *delta* lives in git.

## Backup build — from the FHNW-Security-Lab-Dependencies forks

> Not the main build path (above uses the vendored recipe + upstream). This is the **disaster
> fallback** if an upstream repo disappears or force-pushes. All four GitHub build deps are forked
> (snapshot) under the org, with the exact refs preserved:

| Build dep (upstream) | Fork (backup) | Pinned ref |
|---|---|---|
| `vllm-project/vllm` | `FHNW-Security-Lab-Dependencies/vllm` | `ab666069935c1f23e8ef56038b4659ac9e8f19f8` ✔ verified |
| `NVIDIA/nccl` | `FHNW-Security-Lab-Dependencies/nccl` | `v2.30u1` ✔ |
| `deepseek-ai/DeepGEMM` | `FHNW-Security-Lab-Dependencies/DeepGEMM` | `nv_dev` ✔ |
| `flashinfer-ai/flashinfer` | `FHNW-Security-Lab-Dependencies/flashinfer` | **`v0.6.13`** (`b9077b9`) ✔ — matches the live image |
| `eugr/spark-vllm-docker` | `FHNW-Security-Lab-Dependencies/spark-vllm-docker` | `cf0d5f6` (also vendored here) |

The CUDA base `nvidia/cuda:13.0.2-devel-ubuntu24.04` is a **registry** image (not a GitHub repo) —
it can't be forked to GitHub; mirror it to a registry or rely on the 8 in-cluster image copies if
Docker Hub is unavailable.

To build the base image entirely from our forks, repoint the (hardcoded) clone URLs in the vendored
`Dockerfile` to the forks, then build as normal:

```bash
cd spark-vllm-docker
ORG=https://github.com/FHNW-Security-Lab-Dependencies
sed -i \
  -e "s#https://github.com/vllm-project/vllm.git#$ORG/vllm.git#" \
  -e "s#https://github.com/flashinfer-ai/flashinfer.git#$ORG/flashinfer.git#" \
  -e "s#https://github.com/NVIDIA/nccl.git#$ORG/nccl.git#" \
  -e "s#https://github.com/deepseek-ai/DeepGEMM.git#$ORG/DeepGEMM.git#" \
  Dockerfile
./build-and-copy.sh --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 --flashinfer-ref v0.6.13 --tf5 -t vllm-node-tf5-glm52:base
```

(`--vllm-ref` is the commit, resolved inside our `vllm` fork. The forks still need network to the
GitHub fork + the registry CUDA base + pip/PyPI at build time — they remove the dependency on the
*original* upstream repos surviving, which is the point of the backup.)

## Recovery / backup of the *built* image (vs rebuilding)
Fastest restore, no rebuild — in priority order:
1. **8 in-cluster copies** — the image is loaded on every node; copy from a survivor over RDMA
   (`distribute-image.sh` re-fans it). Lose one node → no rebuild needed.
2. **`docker save` tarball** to durable off-cluster storage (NAS/object store):
   `docker save vllm-node-tf5-glm52:base | zstd -T0 > glm52-base-137ffa76.tar.zst` (~19 GB → ~10–13 GB);
   restore with `zstd -d | docker load`. Keep the sha (`137ffa76`) for verification.
3. **Push to a registry** (cleanest, versioned): e.g. GHCR under the org —
   `docker tag … ghcr.io/fhnw-security-lab-dependencies/glm52-base:ab66606 && docker push …`
   then `docker pull` to restore on any node.
4. **Rebuild from source** — the vendored recipe (main path) or the forks above (backup path).

See `../EXTERNAL-DEPS.md` for the pinned shas of NCCL / the b12x wheel / the image.
