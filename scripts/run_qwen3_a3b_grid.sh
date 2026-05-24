#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/qwen3_a3b_4gpu_grid.yaml}"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/qwen3_a3b_grid_${STAMP}.log"

cleanup_gpu_processes() {
  echo "[grid] cleanup GPU/server processes" | tee -a "$LOG"
  # Best effort: terminate processes owned by this user first. If the process is
  # root-owned and sudo is available, sudo -n is attempted.
  pkill -TERM -f 'sglang.launch_server|vllm.entrypoints.cli.main serve|vllm serve|VLLM::' 2>/dev/null || true
  sleep 8
  local pids
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | sort -u || true)"
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || sudo -n kill $pids 2>/dev/null || true
    sleep 8
  fi
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | sort -u || true)"
  if [[ -n "$pids" ]]; then
    kill -9 $pids 2>/dev/null || sudo -n kill -9 $pids 2>/dev/null || true
    sleep 3
  fi
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tee -a "$LOG"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tee -a "$LOG" || true
}

run_backend() {
  local backend="$1"
  echo "[grid] running backend=$backend" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    moe-bench run "$CONFIG" --backends "$backend" --report 2>&1 | tee -a "$LOG"
}

cleanup_gpu_processes
run_backend vllm
cleanup_gpu_processes
run_backend sglang
cleanup_gpu_processes

# Final combined post-processing/report over the shared run_id directory.
RUN_DIR="results/qwen3-a3b-4gpu-grid"
moe-bench normalize "$RUN_DIR" 2>&1 | tee -a "$LOG"
moe-bench rank "$RUN_DIR" 2>&1 | tee -a "$LOG"
moe-bench report "$RUN_DIR" 2>&1 | tee -a "$LOG"

echo "[grid] done: $RUN_DIR/report.md (+ report.html)" | tee -a "$LOG"
echo "[grid] log: $LOG" | tee -a "$LOG"
