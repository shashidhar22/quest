#!/usr/bin/env bash
# Resume DeepTCR for class II (TRB-II + Paired-II). Uses GPU 3 share.
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/resume_deeptcr.log
exec > >(tee -a "$LOG") 2>&1
echo "[resume/deeptcr] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate track_a_deeptcr
export PYTHONPATH=/home/ubuntu/quest

ENV_LIB=/home/ubuntu/miniforge3/envs/track_a_deeptcr/lib/python3.10/site-packages/nvidia
LD=""
for d in cudnn cublas cuda_runtime cufft curand cusolver cusparse cuda_nvcc; do
  if [ -d "$ENV_LIB/$d/lib" ]; then
    LD="$LD:$ENV_LIB/$d/lib"
  fi
done
export LD_LIBRARY_PATH="${LD#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES="${DEEPTCR_GPU:-3}"

for task in as_trb_ii as_paired_ii; do
  SUM="/home/ubuntu/quest/results/deeptcr/${task}_summary.json"
  if [[ -s "$SUM" ]]; then
    echo "[resume/deeptcr] SKIP (exists): $task"
    continue
  fi
  WORKING_DIR="/home/ubuntu/quest/track_a/deeptcr/.dt_state_gpu/${task}"
  mkdir -p "$WORKING_DIR"
  python -m track_a.deeptcr.run --task "$task" \
    --baseline_name deeptcr --epochs_min 10 \
    --max_test_rows 2000 --working_dir "$WORKING_DIR" \
    --results_root /home/ubuntu/quest/results || { echo "[resume/deeptcr] FAILED $task"; break; }
done
echo "[resume/deeptcr] DONE at $(date -u +%FT%TZ)"
