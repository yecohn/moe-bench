#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_ENV_DIR="${BACKEND_ENV_DIR:-$PROJECT_ROOT/.backends}"
VLLM_PYTHON="${VLLM_PYTHON:-$BACKEND_ENV_DIR/vllm/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-$BACKEND_ENV_DIR/sglang/bin/python}"

check_vllm() {
  echo "[moe-bench] checking vLLM: $VLLM_PYTHON"
  "$VLLM_PYTHON" - <<'PY'
import importlib.metadata as md
import torch
import vllm
import vllm.entrypoints.cli.main
print("torch", torch.__version__)
try:
    print("vllm", md.version("vllm"))
except Exception:
    print("vllm", getattr(vllm, "__version__", "unknown"))
print("vllm CLI import ok")
PY
}

check_sglang() {
  echo "[moe-bench] checking SGLang: $SGLANG_PYTHON"
  "$SGLANG_PYTHON" - <<'PY'
import importlib.metadata as md
import torch
import sglang
import sglang.launch_server
import sglang.bench_serving
print("torch", torch.__version__)
try:
    print("sglang", md.version("sglang"))
except Exception:
    print("sglang", getattr(sglang, "__version__", "unknown"))
print("sglang launch_server and bench_serving import ok")
PY
}

check_vllm
check_sglang
