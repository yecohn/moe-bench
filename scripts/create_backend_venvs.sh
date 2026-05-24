#!/usr/bin/env bash
set -euo pipefail

# Create backend-specific Python environments for moe-bench.
#
# Defaults assume this repository layout:
#   /mnt/projects/AI/josh/moe-bench
#   /mnt/projects/AI/josh/vllm
#   /mnt/projects/AI/josh/sglang/python
#
# Usage:
#   ./scripts/create_backend_venvs.sh all
#   ./scripts/create_backend_venvs.sh vllm
#   ./scripts/create_backend_venvs.sh sglang
#
# Useful environment overrides:
#   PYTHON_BIN=python3.12
#   VLLM_REPO=/path/to/vllm
#   SGLANG_REPO=/path/to/sglang/python
#   BACKEND_ENV_DIR=/path/to/moe-bench/.backends
#   INSTALL_MODE=wheel         # wheel | editable | skip. wheel avoids local compilation.
#   VLLM_PACKAGE='vllm'         # can pin, e.g. vllm==0.x.y
#   SGLANG_PACKAGE='sglang'     # can pin, e.g. sglang==0.x.y
#   PIP_ONLY_BINARY=':all:'     # fail instead of building from source; set empty to disable
#   VLLM_PIP_INSTALL_ARGS='...' # extra args appended to pip install
#   SGLANG_PIP_INSTALL_ARGS='...'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PARENT="$(cd "$PROJECT_ROOT/.." && pwd)"
BACKEND_ENV_DIR="${BACKEND_ENV_DIR:-$PROJECT_ROOT/.backends}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_REPO="${VLLM_REPO:-$DEFAULT_PARENT/vllm}"
SGLANG_REPO="${SGLANG_REPO:-$DEFAULT_PARENT/sglang/python}"
INSTALL_MODE="${INSTALL_MODE:-wheel}"
VLLM_PACKAGE="${VLLM_PACKAGE:-vllm}"
SGLANG_PACKAGE="${SGLANG_PACKAGE:-sglang}"
PIP_ONLY_BINARY="${PIP_ONLY_BINARY:-:all:}"
TARGET="${1:-all}"

pip_binary_args() {
  if [[ -n "${PIP_ONLY_BINARY:-}" ]]; then
    printf '%s\n' "--only-binary=$PIP_ONLY_BINARY"
  fi
}

create_venv() {
  local name="$1"
  local env_path="$BACKEND_ENV_DIR/$name"
  if [[ ! -x "$env_path/bin/python" ]]; then
    echo "[moe-bench] creating $name env: $env_path using $PYTHON_BIN"
    mkdir -p "$BACKEND_ENV_DIR"
    "$PYTHON_BIN" -m venv "$env_path"
  else
    echo "[moe-bench] reusing $name env: $env_path"
  fi
  "$env_path/bin/python" -m pip install --upgrade pip wheel setuptools
}

install_vllm() {
  local env_path="$BACKEND_ENV_DIR/vllm"
  create_venv vllm
  if [[ "$INSTALL_MODE" == "skip" ]]; then
    echo "[moe-bench] skipping vLLM install"
    return
  fi
  if [[ "$INSTALL_MODE" == "wheel" ]]; then
    echo "[moe-bench] installing prebuilt vLLM package: $VLLM_PACKAGE"
    echo "[moe-bench] PIP_ONLY_BINARY=${PIP_ONLY_BINARY:-<disabled>} prevents accidental source builds"
    "$env_path/bin/python" -m pip install $(pip_binary_args) ${VLLM_PIP_INSTALL_ARGS:-} "$VLLM_PACKAGE"
  else
    if [[ ! -f "$VLLM_REPO/pyproject.toml" ]]; then
      echo "vLLM repo not found: $VLLM_REPO" >&2
      exit 2
    fi
    echo "[moe-bench] installing vLLM editable from $VLLM_REPO"
    echo "[moe-bench] note: this can compile CUDA/C++ extensions and take a long time"
    "$env_path/bin/python" -m pip install ${VLLM_PIP_INSTALL_ARGS:-} -e "$VLLM_REPO"
  fi
}

install_sglang() {
  local env_path="$BACKEND_ENV_DIR/sglang"
  create_venv sglang
  if [[ "$INSTALL_MODE" == "skip" ]]; then
    echo "[moe-bench] skipping SGLang install"
    return
  fi
  if [[ "$INSTALL_MODE" == "wheel" ]]; then
    echo "[moe-bench] installing prebuilt SGLang package: $SGLANG_PACKAGE"
    echo "[moe-bench] PIP_ONLY_BINARY=${PIP_ONLY_BINARY:-<disabled>} prevents accidental source builds"
    "$env_path/bin/python" -m pip install $(pip_binary_args) ${SGLANG_PIP_INSTALL_ARGS:-} "$SGLANG_PACKAGE"
  else
    if [[ ! -f "$SGLANG_REPO/pyproject.toml" ]]; then
      echo "SGLang python repo not found: $SGLANG_REPO" >&2
      exit 2
    fi
    echo "[moe-bench] installing SGLang editable from $SGLANG_REPO"
    echo "[moe-bench] note: this installs heavy CUDA/Torch dependencies"
    "$env_path/bin/python" -m pip install ${SGLANG_PIP_INSTALL_ARGS:-} -e "$SGLANG_REPO"
  fi
}

case "$TARGET" in
  all)
    install_vllm
    install_sglang
    ;;
  vllm)
    install_vllm
    ;;
  sglang)
    install_sglang
    ;;
  *)
    echo "Usage: $0 [all|vllm|sglang]" >&2
    exit 2
    ;;
esac

cat <<EOF

[moe-bench] done.
Backend Python paths:
  vLLM:   $BACKEND_ENV_DIR/vllm/bin/python
  SGLang: $BACKEND_ENV_DIR/sglang/bin/python

Use these in configs/*.yaml, or keep the defaults if BACKEND_ENV_DIR is $PROJECT_ROOT/.backends.
Verify with:
  ./scripts/check_backend_envs.sh
EOF
