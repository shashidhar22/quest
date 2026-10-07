#!/bin/bash
# Stage 4.3 on-box checks. Verdicts go to $O/run.log via s43_compare.py.
#
# Lays the work out from the GPU count, so the job runs on whatever type had capacity:
#   >=4 GPUs  eval A/B + judge smoke on 0/1; equivalence run R on 2,3 (world size 2);
#             resume run S on 0,1 once R has saved step 1000.
#   1-2 GPUs  the same phases one after another on GPU 0 (world size 1).
# Either way an optimizer step is 1024 rows: batch 16 x accumulation (64/world) x world.
#
# R = arm-2 config under token-budget batching, uninterrupted to step 2000 -> TRAIN_EQUIV.
# S = resumes from a COPY of R's own checkpoint_step_1000 and runs to 2000 -> RESUME.
#     Deviation from plans/stage4-3.md, which said "kill at 1000, resume": the claim tested is
#     the same ("the resumed run's step-2000 eval matches an uninterrupted run's within 2 %"),
#     the checkpoint is identical (saves are atomic and complete before any kill), and it also
#     tests resume across processes, which is the real spot-instance case. It halves S's training.
# Every path/binary is overridable so dryrun.sh can exercise this on a laptop with stubs.
set -u
export PATH=/home/ubuntu/miniforge3/bin:/usr/local/bin:$PATH
Q=${Q:-/home/ubuntu/quest}; O=${O:-/home/ubuntu/s43_out}; OLD=${OLD:-/home/ubuntu/s43_old}
CMP=${CMP:-/home/ubuntu/s43_compare.py}; TORCHRUN=${TORCHRUN:-torchrun}; PY=${PY:-python}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)}
export PYTHONPATH=$Q PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p $O; cd $Q
say(){ echo "[$(date -u '+%F %T')] $*" >> $O/run.log; }
echo $$ > $O/run_s43.pid   # the watchdog waits on THIS pid, never a name pattern

if [ "${NGPU:-1}" -ge 4 ]; then PARALLEL=1; NPROC=2; R_GPUS=2,3; S_GPUS=0,1; EVAL2_GPU=1
else PARALLEL=0; NPROC=1; R_GPUS=0; S_GPUS=0; EVAL2_GPU=0; fi
GA=$((64 / NPROC))   # 16 rows x GA x NPROC = 1024 rows per optimizer step, any GPU count
SAMPLER=$($PY -c 'from quest.training.samplers import TokenBudgetStepSampler as T; print("v2" if hasattr(T, "_cuts") else "v1")' 2>/dev/null || echo unknown)
PHASES=${PHASES:-all}   # all = evals + judge + R + S;  R2 = only the equivalence run (needs the v2 sampler)
# Run dirs and verdict labels carry the sampler generation: v1 = s43_equiv_pointD / TRAIN_EQUIV
# (the 2026-10-06 run, kept intact), v2 = s43_equiv_v2_pointD / TRAIN_EQUIV_V2 (+ V2_VS_V1).
SUF=""; [ "$SAMPLER" = v2 ] && SUF="_v2"; export SUF
say "start pid=$$ ngpu=$NGPU parallel=$PARALLEL world_size=$NPROC grad_accum=$GA sampler=$SAMPLER phases=$PHASES"

EV="--adapter_dir $Q/checkpoints/cpt_stage41_pointD_run05/final --canonical_eval_path $Q/data/cpt_canonical_eval_v2 --base_model esmc_300m --tcr_variant cdr3 --mhc_variant full --eos_strategy raw --max_length 1024 --batch_size 16"
TRAIN="scripts/training/train_cpt_stage2_single.py --dataset_path $Q/data/cpt_datasets/run05_cdr3_fullmhc_proportional_10M \
  --tcr_variant cdr3 --mhc_variant full --max_length 1024 --batch_size 16 --gradient_accumulation_steps $GA \
  --token_budget 16384 --bucket_boundaries 64 128 256 512 1024 \
  --base_model esmc_300m --canonical_eval_path $Q/data/cpt_canonical_eval_v2 --eos_strategy raw --num_epochs 1 \
  --lr 2e-4 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --eval_steps 500 --save_steps 500 --logging_steps 50 \
  --no_compile --attn_implementation sdpa --wandb_project tcrbench-v3-cpt-stage43 --report_to wandb \
  --early_stop_n_patience 3 --early_stop_min_rel 0.01 --early_stop_metric_path pocket overall _ perplexity"
EQ=$Q/checkpoints/s43_equiv${SUF}_pointD; RS=$Q/checkpoints/s43_resume${SUF}_pointD

# stop_after_evals <pid> <log> <n> <label>: TERM the torchrun 90 s after its n-th canonical eval.
stop_after_evals(){
  local pid=$1 log=$2 n=$3 label=$4 t=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    if [ "$(grep -c 'canonical_v2 eval @ step' $log 2>/dev/null)" -ge "$n" ]; then
      say "$label: $n evals after $(( $(date +%s)-t ))s; stopping pid $pid"; sleep 90; kill -TERM $pid; break
    fi
    sleep 30
  done
  while kill -0 $pid 2>/dev/null; do sleep 5; done
}

# --- eval A/B (old lib vs new lib on one adapter) + the Stage 4.2 judge smoke ---
eval_old_and_judge(){
  local t=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 $PY $OLD/cpt_stage3_eval_baseline.py $EV --output_json $O/eval_old.json > $O/eval_old.log 2>&1
  say "eval_old rc=$? secs=$(( $(date +%s)-t ))"
  local J=$O/judge_smoke; mkdir -p $J; t=$(date +%s)
  # The probe is the real smoke: does the scorer's output depend on the input at all? On
  # 2026-10-06 the smoke "passed" (rc=0) while scoring every candidate identically; wave 1 then
  # spent 18 GPU-hours finding out. Its verdict line lands in run.log next to the others.
  CUDA_VISIBLE_DEVICES=0 $PY -m track_a.esm_sft.judge probe --task as_paired_i --adapter none --tcr_variant cdr3 --mhc_variant full > $J/probe.log 2>&1
  say "judge_probe rc=$? $(grep -h '^\[probe\] \(PASS\|FAIL\)' $J/probe.log | tail -1)"
  CUDA_VISIBLE_DEVICES=0 $PY -m track_a.esm_sft.judge train --task as_paired_i --adapter none --tcr_variant cdr3 --mhc_variant full --out_dir $J --smoke > $J/train.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=0 $PY -m track_a.esm_sft.judge eval --task as_paired_i --adapter none --tcr_variant cdr3 --mhc_variant full --ckpt $J/best.pt --out $J/eval.json --smoke > $J/eval.log 2>&1
  say "judge_smoke rc=$? secs=$(( $(date +%s)-t ))"
  $PY /home/ubuntu/s42_check.py gate --eval $J/eval.json 2>&1 | sed 's/^/    smoke /' >> $O/run.log   # advisory on 20 queries
}
eval_new(){
  local t=$(date +%s)
  CUDA_VISIBLE_DEVICES=$EVAL2_GPU $PY scripts/training/cpt_stage3_eval_baseline.py $EV --output_json $O/eval_new.json > $O/eval_new.log 2>&1
  say "eval_new rc=$? secs=$(( $(date +%s)-t ))"
}

# --- R: the equivalence run, uninterrupted to step 2000 ---
start_R(){
  mkdir -p $EQ
  CUDA_VISIBLE_DEVICES=$R_GPUS $TORCHRUN --nproc_per_node=$NPROC --master_port=29761 $TRAIN \
    --output_dir $EQ --wandb_run_name stage43-equiv-pointD-ws$NPROC > $EQ/training.log 2>&1 &
  RPID=$!; say "R (equivalence) torchrun pid=$RPID on GPU(s) $R_GPUS"
}
finish_R(){
  stop_after_evals $RPID $EQ/training.log 4 equivalence
  cp $EQ/training.log $O/train${SUF}.log 2>/dev/null; cp $EQ/early_stop_history.json $O/train_equiv${SUF}_early_stop_history.json 2>/dev/null
  $PY $CMP equiv >> $O/run.log 2>&1     # label TRAIN_EQUIV${SUF^^}; plus V2_VS_V1 when SUF=_v2
}

# --- S: resume from a COPY of R's step-1000 checkpoint, run to step 2000 ---
run_S(){
  local src=$EQ/checkpoint_step_1000 t=$(date +%s)
  while [ ! -f $src/training_state.pt ] && kill -0 $RPID 2>/dev/null; do sleep 15; done
  if [ ! -f $src/training_state.pt ]; then say "S SKIPPED: R never produced a complete $src"; return 1; fi
  rm -rf $RS; mkdir -p $RS; cp -r $src $RS/   # copy, so R's keep-3 cleanup can never race us
  say "S (resume) from $RS/checkpoint_step_1000 after $(( $(date +%s)-t ))s; GPU(s) $S_GPUS"
  CUDA_VISIBLE_DEVICES=$S_GPUS $TORCHRUN --nproc_per_node=$NPROC --master_port=29762 $TRAIN \
    --output_dir $RS --resume_from $RS/checkpoint_step_1000 \
    --wandb_run_name stage43-resume-pointD-ws$NPROC > $RS/training.log 2>&1 &
  SPID=$!; say "S torchrun pid=$SPID"
  stop_after_evals $SPID $RS/training.log 2 resume   # step 1500 and step 2000
}

if [ "$PHASES" = R2 ]; then
  # Standalone re-test: the equivalence run alone, under the v2 sampler.
  [ "$SAMPLER" = v2 ] || { say "R2 ABORT: installed sampler is '$SAMPLER', not v2"; exit 2; }
  rm -rf $EQ; start_R; finish_R; say "R2 done"; exit 0
fi
if [ "$PARALLEL" = 1 ]; then
  start_R
  ( eval_old_and_judge ) & A=$!
  ( eval_new ) & B=$!
  wait $A $B
  $PY $CMP ab >> $O/run.log 2>&1
  ( finish_R ) & RWATCH=$!
  run_S
  wait $RWATCH 2>/dev/null
else
  eval_old_and_judge
  eval_new
  $PY $CMP ab >> $O/run.log 2>&1
  start_R
  finish_R          # R holds the only GPU; S cannot start until it stops
  run_S
fi

cp $RS/training.log $O/resume${SUF}_training.log 2>/dev/null; cp $RS/early_stop_history.json $O/resume${SUF}_early_stop_history.json 2>/dev/null
$PY $CMP resume >> $O/run.log 2>&1     # label RESUME${SUF^^}
say "all jobs done"
