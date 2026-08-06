#!/usr/bin/env bash
# Wait for R1 training to finish, then run R2 routing eval on the R1 checkpoint.
# R1 and R2 share identical training; R2 only differs in eval.routing (method_v1).
# Do NOT retrain for R2 — saves ~15–20 hours.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-/root/miniconda3/envs/ngsg/bin/python}"
LOGDIR=logs
mkdir -p "$LOGDIR"

R1_CFG="configs/ngsg/method_v1_role_train_r1_full_seed0.yaml"
R2_EVAL_CFG="configs/ngsg/method_v1_role_routing_r2_full_seed0.yaml"
R1_RUN="experiments/method_v1_role_train_r1_full_seed0"
CHAIN_LOG="$LOGDIR/method_v1_chain_20260709.log"

exec >>"$CHAIN_LOG" 2>&1
echo "[chain] started at $(date)"

wait_for_config() {
  local cfg="$1"
  local label="$2"
  echo "[chain] waiting for $label to finish (config=$cfg) ..."
  while pgrep -f "run_baseline.py --config $cfg" >/dev/null; do
    sleep 60
  done
  echo "[chain] $label finished at $(date)"
}

if pgrep -f "run_baseline.py --config $R1_CFG" >/dev/null; then
  wait_for_config "$R1_CFG" "R1"
else
  echo "[chain] R1 not running; assuming training already completed"
fi

if [[ ! -f "$R1_RUN/artifacts/model_after_task2.pt" ]]; then
  echo "[chain] ERROR: missing $R1_RUN/artifacts/model_after_task2.pt"
  exit 1
fi

if [[ ! -f "$R1_RUN/result.json" ]]; then
  echo "[chain] ERROR: missing $R1_RUN/result.json"
  exit 1
fi

OUT_JSON="$R1_RUN/diagnostics/method_v1_r2_routing_eval.json"
mkdir -p "$R1_RUN/diagnostics"

echo "[chain] R2 routing eval on R1 checkpoint (no retrain) at $(date)"
"$PY" scripts/eval_method_v1_routing.py \
  --run-dir "$R1_RUN" \
  --config "$R2_EVAL_CFG" \
  --checkpoint-stage task2 \
  --device cuda \
  --output-json "$OUT_JSON" \
  --write-markdown

echo "[chain] wrote $OUT_JSON"
echo "[chain] all done at $(date)"
