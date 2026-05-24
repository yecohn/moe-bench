#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/moe_parallelism_4gpu.yaml}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
RUN_ID="${RUN_ID:-qwen3-a3b-4gpu-parallelism}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moe-bench-${RUN_ID}}"
DTYPE="${DTYPE:-bfloat16}"

LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/${RUN_ID}_${STAMP}.log"
RUN_DIR="results/${RUN_ID}"

cleanup_gpu_processes() {
  echo "[parallelism] cleanup GPU/server processes" | tee -a "$LOG"
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
  echo "[parallelism] running backend=$backend model=$MODEL run_id=$RUN_ID" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    moe-bench run "$CONFIG" \
      --backends "$backend" \
      --model "$MODEL" \
      --run-id "$RUN_ID" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --dtype "$DTYPE" \
      --report 2>&1 | tee -a "$LOG"
}

cleanup_gpu_processes
run_backend vllm
cleanup_gpu_processes
run_backend sglang
cleanup_gpu_processes

moe-bench normalize "$RUN_DIR" 2>&1 | tee -a "$LOG"
moe-bench rank "$RUN_DIR" 2>&1 | tee -a "$LOG"
moe-bench report "$RUN_DIR" 2>&1 | tee -a "$LOG"

echo "[parallelism] done: $RUN_DIR/report.md (+ report.html)" | tee -a "$LOG"
echo "[parallelism] log: $LOG" | tee -a "$LOG"
