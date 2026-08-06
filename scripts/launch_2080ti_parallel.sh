#!/usr/bin/env bash
set -euo pipefail

# Three-GPU/CPU layout for the dual-socket Xeon + 3x RTX 2080 Ti server.
# Override these variables when launching a different reproduction:
#   LANGEVIN_CONFIG=... FLW_CONFIG=... DIAG_RUN_DIR=... bash scripts/launch_2080ti_parallel.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_LANGEVIN="${GPU_LANGEVIN:-0}"
GPU_FLW="${GPU_FLW:-1}"
LANGEVIN_CONFIG="${LANGEVIN_CONFIG:-configs/neurocomputing/neurocomputing_mnist_to_emnist_langevin_seed4_inhibition_r3_r1.yaml}"
FLW_CONFIG="${FLW_CONFIG:-configs/baseline/frozen_paper_protocol.yaml}"
LANGEVIN_RUN_NAME="${LANGEVIN_RUN_NAME:-langevin_2080ti_gpu${GPU_LANGEVIN}}"
FLW_RUN_NAME="${FLW_RUN_NAME:-flw_2080ti_gpu${GPU_FLW}}"
DIAG_RUN_DIR="${DIAG_RUN_DIR:-}"
DIAG_SCRIPT="${DIAG_SCRIPT:-scripts/eval_partition_group_diagnosis.py}"
DIAG_THREADS="${DIAG_THREADS:-32}"
TRAIN_THREADS="${TRAIN_THREADS:-8}"
INTEROP_THREADS="${INTEROP_THREADS:-1}"
LOG_DIR="${LOG_DIR:-logs/parallel_2080ti}"

mkdir -p "$LOG_DIR"

if [[ -z "$DIAG_RUN_DIR" ]]; then
  echo "DIAG_RUN_DIR is required; it should point to an existing run with resolved_config.json." >&2
  exit 2
fi
mkdir -p "$DIAG_RUN_DIR/diagnostics"
if [[ "$GPU_LANGEVIN" == "$GPU_FLW" ]]; then
  echo "GPU_LANGEVIN and GPU_FLW must be different." >&2
  exit 2
fi
for path in "$LANGEVIN_CONFIG" "$FLW_CONFIG" "$DIAG_SCRIPT"; do
  [[ -e "$path" ]] || { echo "Missing path: $path" >&2; exit 2; }
done

echo "Launching Langevin on physical GPU $GPU_LANGEVIN"
CUDA_VISIBLE_DEVICES="$GPU_LANGEVIN" \
  python scripts/run_baseline.py \
    --config "$LANGEVIN_CONFIG" \
    --device cuda \
    --run-name "$LANGEVIN_RUN_NAME" \
    --torch-threads "$TRAIN_THREADS" \
    --torch-interop-threads "$INTEROP_THREADS" \
    >"$LOG_DIR/$LANGEVIN_RUN_NAME.log" 2>&1 &
PID_LANGEVIN=$!

echo "Launching FLW on physical GPU $GPU_FLW"
CUDA_VISIBLE_DEVICES="$GPU_FLW" \
  python scripts/run_baseline.py \
    --config "$FLW_CONFIG" \
    --device cuda \
    --run-name "$FLW_RUN_NAME" \
    --torch-threads "$TRAIN_THREADS" \
    --torch-interop-threads "$INTEROP_THREADS" \
    >"$LOG_DIR/$FLW_RUN_NAME.log" 2>&1 &
PID_FLW=$!

echo "Launching CPU diagnostics on $DIAG_THREADS threads"
(
  OMP_NUM_THREADS="$DIAG_THREADS" MKL_NUM_THREADS="$DIAG_THREADS" \
    python "$DIAG_SCRIPT" \
      --run-dir "$DIAG_RUN_DIR" \
      --checkpoint-stage task2 \
      --test-task task2 \
      --device cpu \
      --write-json "$DIAG_RUN_DIR/diagnostics/parallel_task2_group_diagnosis.json"
  OMP_NUM_THREADS="$DIAG_THREADS" MKL_NUM_THREADS="$DIAG_THREADS" \
    python "$DIAG_SCRIPT" \
      --run-dir "$DIAG_RUN_DIR" \
      --checkpoint-stage task2 \
      --test-task task1 \
      --device cpu \
      --write-json "$DIAG_RUN_DIR/diagnostics/parallel_task1_group_diagnosis_after_task2.json"
) >"$LOG_DIR/diagnostics_cpu.log" 2>&1 &
PID_DIAG=$!

echo "PIDs: Langevin=$PID_LANGEVIN FLW=$PID_FLW diagnostics=$PID_DIAG"
echo "Logs: $LOG_DIR"
echo "Use: tail -f $LOG_DIR/*.log"

wait "$PID_LANGEVIN"
wait "$PID_FLW"
wait "$PID_DIAG"
echo "All parallel jobs completed successfully."
