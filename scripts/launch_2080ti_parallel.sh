#!/usr/bin/env bash
set -euo pipefail

# Two-GPU training layout for the dual-socket Xeon + 3x RTX 2080 Ti server.
# SixDiagnostics is collected inside each Task-2 training loop.
# Override these variables when launching a different reproduction:
#   LANGEVIN_CONFIG=... FLW_CONFIG=... bash scripts/launch_2080ti_parallel.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_LANGEVIN="${GPU_LANGEVIN:-0}"
GPU_FLW="${GPU_FLW:-1}"
LANGEVIN_CONFIG="${LANGEVIN_CONFIG:-configs/neurocomputing/neurocomputing_mnist_to_emnist_langevin_seed4_inhibition_r3_r1.yaml}"
FLW_CONFIG="${FLW_CONFIG:-configs/baseline/frozen_paper_protocol.yaml}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LANGEVIN_RUN_NAME="${LANGEVIN_RUN_NAME:-langevin_2080ti_${RUN_TAG}_gpu${GPU_LANGEVIN}}"
FLW_RUN_NAME="${FLW_RUN_NAME:-flw_2080ti_${RUN_TAG}_gpu${GPU_FLW}}"
TRAIN_THREADS="${TRAIN_THREADS:-8}"
INTEROP_THREADS="${INTEROP_THREADS:-1}"
LOG_DIR="${LOG_DIR:-logs/parallel_2080ti}"

mkdir -p "$LOG_DIR"

if [[ "$GPU_LANGEVIN" == "$GPU_FLW" ]]; then
  echo "GPU_LANGEVIN and GPU_FLW must be different." >&2
  exit 2
fi
for path in "$LANGEVIN_CONFIG" "$FLW_CONFIG"; do
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

echo "SixDiagnostics: enabled inside both Task-2 training loops"
echo "GPU 2 remains available for post-hoc analysis after training."
echo "PIDs: Langevin=$PID_LANGEVIN FLW=$PID_FLW"
echo "Logs: $LOG_DIR"
echo "Use: tail -f $LOG_DIR/*.log"

wait "$PID_LANGEVIN"
wait "$PID_FLW"
echo "All parallel jobs completed successfully."
