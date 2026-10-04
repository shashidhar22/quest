#!/usr/bin/env bash
# DeepTCR queue on a GPU. Shares GPU 0 with ESM2-8M (sub-2 GB usage each
# leaves >19 GB headroom on a 23 GB L4).
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/gpu_deeptcr.log
exec > >(tee -a "$LOG") 2>&1
echo "[gpu/deeptcr] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate track_a_deeptcr
export PYTHONPATH=/home/ubuntu/quest

# TF 2.12 needs CUDA 11 libs from pip-installed nvidia-* packages
ENV_LIB=/home/ubuntu/miniforge3/envs/track_a_deeptcr/lib/python3.10/site-packages/nvidia
LD=""
for d in cudnn cublas cuda_runtime cufft curand cusolver cusparse cuda_nvcc; do
  if [ -d "$ENV_LIB/$d/lib" ]; then
    LD="$LD:$ENV_LIB/$d/lib"
  fi
done
export LD_LIBRARY_PATH="${LD#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
# Pin to one GPU so we don't grab all 4
export CUDA_VISIBLE_DEVICES="${DEEPTCR_GPU:-0}"

echo "[gpu/deeptcr] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "[gpu/deeptcr] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for task in as_trb_i as_paired_i as_trb_ii as_paired_ii; do
  echo "[gpu/deeptcr] running deeptcr / $task at $(date -u +%FT%TZ)"
  WORKING_DIR="/home/ubuntu/quest/track_a/deeptcr/.dt_state_gpu/${task}"
  mkdir -p "$WORKING_DIR"
  python -m track_a.deeptcr.run --task "$task" \
    --baseline_name deeptcr --epochs_min 10 \
    --max_test_rows 2000 --working_dir "$WORKING_DIR" \
    --results_root /home/ubuntu/quest/results || { echo "[gpu/deeptcr] FAILED $task"; break; }
done
echo "[gpu/deeptcr] DONE at $(date -u +%FT%TZ)"
