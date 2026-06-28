# GLM-5.2 external binary dependencies (not in git)

Everything the live 512k deployment needs that is **code** is committed and is
byte-identical across all 8 nodes (verified 2026-06-28): `start_glm52_8node.sh`,
`glm52-sparse-patches.sh`, `kernels/*.py`, `tools/watchdog_glm52_cluster.sh`,
`tools/systemd/sparks-glm52-watchdog.*`. This file pins the three **binary** deps that
are too large (or are upstream artifacts) to vendor in git, so they stay reproducible.

Recovery rule of thumb: each binary exists on **all 8 nodes**, so the fastest restore for
a single rebuilt node is to copy from a surviving node over the 200G RDMA fabric; the
provenance below is for the case where all copies are lost. **Verify by sha256 after any
fetch** — a wrong NCCL/image silently changes behavior.

## 1. Container image `vllm-node-tf5-glm52:base`

- Identity: `sha256:137ffa76b4254dda2a9a234f721c985edd3f956c04bff909ade65d8cb30b35c8`, ~19 GB.
- Build: external repo [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker),
  `build-and-copy.sh --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 --tf5` (pinned
  post-0.23.0 vLLM with `GlmMoeDsa` + indexer/MTP; prebuilt flashinfer/vLLM wheels).
  The base under it is the NGC vLLM image. **The build recipe is now vendored** at
  `image/spark-vllm-docker/` (eugr @ `cf0d5f6`, minus the wheel cache) with build instructions in
  `image/BUILD-IMAGE.md` — so the image is buildable from this repo.
- Important: our GB10/sm_121 fixes are **NOT baked into the image**. The image is a vanilla
  eugr build; all customization is runtime-mounted/applied from this repo
  (`glm52-sparse-patches.sh` + `kernels/`). So the reproducible *delta* is fully in git;
  only the plain base image is external.
- Distribute: `distribute-image.sh` (`docker save | rsync over RDMA | docker load` to the 7
  workers).

## 2. NCCL 2.30.4 (aarch64, CUDA 13.2)

- File: `~/models/nccl-2.30.4/libnccl.so.2` on every node (rides the `/models` mount into
  the container as `/models/nccl-2.30.4/libnccl.so.2`, `LD_PRELOAD`ed).
- Identity: `NCCL version 2.30.4+cuda13.2`, ELF aarch64.
  sha256 `0bf24802ae809c796f216ec2a789c74e3dde8d31ac3c27aa068c8ef67e2436dc`, 248,786,184 bytes.
- Why pinned: it overrides the image's NCCL 2.29.7, which has an aarch64 `shm_broadcast`
  warmup wedge; 2.30.4 + `/dev/infiniband` passthrough is what makes TP=8 stable on GB10.
- Reconstruct: it is the **official NVIDIA NCCL 2.30.4 for CUDA 13.x, aarch64** — e.g. the
  `libnccl.so.2` inside the `nvidia-nccl-cu13==2.30.4` wheel (`pip download nvidia-nccl-cu13==2.30.4`,
  unzip, take `nvidia/nccl/lib/libnccl.so.2`) or the NCCL release tarball from NVIDIA.
  Verify the sha256 above before staging. Too large for git (248 MB); not committed.

## 3. b12x 0.23.0 wheel

- **Vendored in this repo:** `wheels/b12x-0.23.0-py3-none-any.whl`
  (sha256 `22c3496ef400ff594ec3c1ac686deeaacf6681031a6452b3ab6cd3077308d4f4`, 780,992 bytes).
- Also staged at `~/models/wheels/` on every node; the patch script installs it
  (`pip install --no-deps b12x==0.23.0`). Pure `py3-none-any`, so the vendored wheel is the
  pinned fallback if PyPI is unavailable. CuTe-DSL cudagraph-capture-safe sparse decode
  (`GLM52_B12X_MLA=1`); production runs eager so it only matters when cudagraph is enabled.
