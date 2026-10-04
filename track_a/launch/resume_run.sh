#!/usr/bin/env bash
# Run ONE cell, skipping if results already exist (idempotent re-run).
# Reuses existing checkpoint if found.
# Usage: ./resume_run.sh <baseline> <task> <gpu> [esmc_max_rows]
set -e
BASELINE="$1"
TASK="$2"
GPU="${3:-0}"
ESMC_MAX_ROWS="${4:-1000}"  # smaller cap for slow ESM-C evals
CD="/home/ubuntu/quest"
RESULTS_ROOT="${CD}/results"
SUMMARY="${RESULTS_ROOT}/${BASELINE}/${TASK}_summary.json"

if [[ -s "$SUMMARY" ]]; then
  # Already complete — skip
  echo "[resume_run] SKIP (exists): $BASELINE / $TASK"
  exit 0
fi

cd "$CD"
export CUDA_VISIBLE_DEVICES="$GPU"

OUT_DIR="${CD}/track_a/.train_artifacts/${BASELINE}/${TASK}"
mkdir -p "$OUT_DIR"
CKPT="${OUT_DIR}/best.pt"

case "$BASELINE" in
  lstm|mlp)
    if [[ ! -s "$CKPT" ]]; then
      python -m track_a.lstm_mlp.train --model "$BASELINE" --task "$TASK" \
        --output_dir "$OUT_DIR" --epochs 20 --batch_size 64 --patience 3 --wandb
    else
      echo "[resume_run] reusing checkpoint $CKPT"
    fi
    python -m track_a.lstm_mlp.evaluate --task "$TASK" \
      --checkpoint "$CKPT" --baseline_name "$BASELINE" \
      --results_root "$RESULTS_ROOT" --max_test_rows 2000
    ;;
  tcrbert)
    if [[ ! -s "$CKPT" ]]; then
      python -m track_a.tcrbert.train --task "$TASK" \
        --output_dir "$OUT_DIR" --epochs 5 --batch_size 64 --patience 1 --wandb
    else
      echo "[resume_run] reusing checkpoint $CKPT"
    fi
    python -m track_a.tcrbert.evaluate --task "$TASK" \
      --checkpoint "$CKPT" --baseline_name "$BASELINE" \
      --results_root "$RESULTS_ROOT" --max_test_rows 2000
    ;;
  esm2_8m-*|esm2_35m-*|esmc_300m-*)
    ENC="${BASELINE%-*}"
    VAR="${BASELINE##*-}"
    EPOCHS=3; BSZ=64
    [[ "$ENC" == "esm2_35m" ]] && { EPOCHS=2; BSZ=32; }
    [[ "$ENC" == "esmc_300m" ]] && { EPOCHS=2; BSZ=8; }
    MAX_TEST_ROWS=2000
    [[ "$ENC" == "esmc_300m" ]] && MAX_TEST_ROWS="$ESMC_MAX_ROWS"
    if [[ ! -s "$CKPT" ]]; then
      python -m track_a.esm_sft.train --encoder "$ENC" --variant "$VAR" --task "$TASK" \
        --output_dir "$OUT_DIR" --epochs "$EPOCHS" --batch_size "$BSZ" --patience 1 --wandb
    else
      echo "[resume_run] reusing checkpoint $CKPT"
    fi
    python -m track_a.esm_sft.evaluate --encoder "$ENC" --variant "$VAR" --task "$TASK" \
      --checkpoint "$CKPT" --baseline_name "$BASELINE" \
      --results_root "$RESULTS_ROOT" --max_test_rows "$MAX_TEST_ROWS"
    ;;
  deeptcr)
    # DeepTCR has its own conda env; the launcher script handles it
    echo "[resume_run] deeptcr should run via launch_deeptcr.sh, not run_baseline.sh"; exit 2
    ;;
  *)
    echo "[resume_run] unknown baseline: $BASELINE" >&2; exit 2
    ;;
esac
echo "[resume_run] DONE: $BASELINE / $TASK"
