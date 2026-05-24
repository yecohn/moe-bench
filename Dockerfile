# moe-bench container: platform + pre-built vLLM and SGLang backend venvs.
#
# Build:
#   docker build -t moe-bench .
# Run:
#   docker run --rm -it --gpus all \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -v $(pwd)/results:/app/moe-bench/results \
#     moe-bench
# Then inside the container:
#   moe-bench run configs/moe_parallelism_4gpu.yaml --report
#
# See BACKENDS.md section 10 for the full reference.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Pinned-but-overridable defaults. Bump these as new releases appear or pass
# --build-arg VLLM_PACKAGE=vllm==X.Y.Z to override at build time.
ARG PYTHON_VERSION=3.12
ARG VLLM_PACKAGE=vllm==0.8.0
ARG SGLANG_PACKAGE=sglang==0.4.8

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common ca-certificates gnupg \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-venv \
      python${PYTHON_VERSION}-dev \
      python3-pip \
      git \
      curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/moe-bench
COPY . .

# Two backend venvs at .backends/{vllm,sglang}/bin/python. Reuses
# scripts/create_backend_venvs.sh so the install logic lives in one place
# (same code path as the host workflow in BACKENDS.md).
RUN VLLM_PACKAGE="${VLLM_PACKAGE}" \
    SGLANG_PACKAGE="${SGLANG_PACKAGE}" \
    PYTHON_BIN="python${PYTHON_VERSION}" \
    scripts/create_backend_venvs.sh all

# Platform venv (moe-bench platform), with moe-bench on PATH.
RUN python${PYTHON_VERSION} -m venv /opt/moe-bench-venv \
 && /opt/moe-bench-venv/bin/pip install --upgrade pip wheel setuptools \
 && /opt/moe-bench-venv/bin/pip install -e . \
 && ln -s /opt/moe-bench-venv/bin/moe-bench /usr/local/bin/moe-bench

# Fail the build now if either backend's import is broken (e.g. a version
# pin that produced an incompatible install). Pure-Python imports, no GPU
# required at build time.
RUN scripts/check_backend_envs.sh

ENTRYPOINT ["/bin/bash"]
