#!/usr/bin/env bash
# Train + evaluate one baseline cell. Usage:
#   ./run_baseline.sh <baseline> <task> <gpu>
# baseline ∈ {lstm, mlp, tcrbert, esm2_8m-base, esm2_8m-cpt, esm2_35m-base, esm2_35m-cpt, esmc_300m-base, esmc_300m-cpt}

set -euo pipefail

BASELINE="$1"
TASK="$2"
GPU="${3:-0}"
CD="/home/ubuntu/quest"
RESULTS_ROOT="${CD}/results"

cd "$CD"
export CUDA_VISIBLE_DEVICES="$GPU"

OUT_DIR="${CD}/track_a/.train_artifacts/${BASELINE}/${TASK}"
mkdir -p "$OUT_DIR"

case "$BASELINE" in
  lstm|mlp)
    python -m track_a.lstm_mlp.train --model "$BASELINE" --task "$TASK" \
      --output_dir "$OUT_DIR" --epochs 20 --batch_size 64 --patience 3 --wandb
    python -m track_a.lstm_mlp.evaluate --task "$TASK" \
      --checkpoint "${OUT_DIR}/best.pt" --baseline_name "$BASELINE" \
      --results_root "$RESULTS_ROOT" --max_test_rows 2000
    ;;
  tcrbert)
    python -m track_a.tcrbert.train --task "$TASK" \
      --output_dir "$OUT_DIR" --epochs 5 --batch_size 64 --patience 1 --wandb
    python -m track_a.tcrbert.evaluate --task "$TASK" \
      --checkpoint "${OUT_DIR}/best.pt" --baseline_name "$BASELINE" \
      --results_root "$RESULTS_ROOT" --max_test_rows 2000
    ;;
  esm2_8m-*|esm2_35m-*|esmc_300m-*)
    ENC="${BASELINE%-*}"
    VAR="${BASELINE##*-}"
    EPOCHS=3
    BSZ=64
    if [[ "$ENC" == "esm2_35m" ]]; then EPOCHS=2; BSZ=32; fi
    if [[ "$ENC" == "esmc_300m" ]]; then EPOCHS=2; BSZ=8; fi
    python -m track_a.esm_sft.train --encoder "$ENC" --variant "$VAR" --task "$TASK" \
      --output_dir "$OUT_DIR" --epochs "$EPOCHS" --batch_size "$BSZ" --patience 1 --wandb
    python -m track_a.esm_sft.evaluate --encoder "$ENC" --variant "$VAR" --task "$TASK" \
      --checkpoint "${OUT_DIR}/best.pt" --baseline_name "$BASELINE" \
      --results_root "$RESULTS_ROOT" --max_test_rows 2000
    ;;
  *)
    echo "[run_baseline] unknown baseline: $BASELINE" >&2
    exit 2
    ;;
esac
echo "[run_baseline] DONE: $BASELINE / $TASK"
