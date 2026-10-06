#!/bin/bash
# Stage 4.3 on-box checks on a 4-GPU box. Verdicts go to $O/run.log via s43_compare.py.
#   GPUs 2,3  token-budget EQUIVALENCE run of the arm-2 config, stopped after its step-2000 eval.
#   GPUs 0,1  eval A/B old vs new (+ the Stage 4.2 judge smoke on GPU 0), then the RESUME check:
#             the same config to its step-1000 checkpoint, kill, `--resume_from auto`, stop after
#             the step-2000 eval. RESUME compares that eval with the equivalence run's.
# Every path/binary is overridable so `dryrun.sh` can exercise this flow on a laptop with stubs.
set -u
export PATH=/home/ubuntu/miniforge3/bin:/usr/local/bin:$PATH
Q=${Q:-/home/ubuntu/quest}; O=${O:-/home/ubuntu/s43_out}; OLD=${OLD:-/home/ubuntu/s43_old}
CMP=${CMP:-/home/ubuntu/s43_compare.py}; TORCHRUN=${TORCHRUN:-torchrun}; PY=${PY:-python}
export PYTHONPATH=$Q PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p $O; cd $Q
say(){ echo "[$(date -u '+%F %T')] $*" >> $O/run.log; }
echo $$ > $O/run_s43.pid   # the watchdog waits on THIS pid, never a name pattern
say "start pid=$$"

EV="--adapter_dir $Q/checkpoints/cpt_stage41_pointD_run05/final --canonical_eval_path $Q/data/cpt_canonical_eval_v2 --base_model esmc_300m --tcr_variant cdr3 --mhc_variant full --eos_strategy raw --max_length 1024 --batch_size 16"
# Arm-2 config under token-budget batching (effective batch 16 x 32 x 2 = 1024 rows/step).
TRAIN="scripts/training/train_cpt_stage2_single.py --dataset_path $Q/data/cpt_datasets/run05_cdr3_fullmhc_proportional_10M \
  --tcr_variant cdr3 --mhc_variant full --max_length 1024 --batch_size 16 --gradient_accumulation_steps 32 \
  --token_budget 16384 --bucket_boundaries 64 128 256 512 1024 \
  --base_model esmc_300m --canonical_eval_path $Q/data/cpt_canonical_eval_v2 --eos_strategy raw --num_epochs 1 \
  --lr 2e-4 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --eval_steps 500 --logging_steps 50 \
  --no_compile --attn_implementation sdpa --wandb_project tcrbench-v3-cpt-stage43 --report_to wandb \
  --early_stop_n_patience 3 --early_stop_min_rel 0.01 --early_stop_metric_path pocket overall _ perplexity"
EQ=$Q/checkpoints/s43_equiv_pointD; RS=$Q/checkpoints/s43_resume_pointD

# stop_after_evals <pid> <log> <n> <label>: TERM the torchrun 90 s after its n-th canonical eval line.
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

# --- GPUs 2,3: equivalence run. Its watcher stops it after the 4th eval (= step 2000) and writes
# the TRAIN_EQUIV verdict at once, so that verdict never waits on the resume check. ---
mkdir -p $EQ
CUDA_VISIBLE_DEVICES=2,3 $TORCHRUN --nproc_per_node=2 --master_port=29761 $TRAIN \
  --output_dir $EQ --save_steps 2000 --wandb_run_name stage43-equiv-pointD-ws2 > $EQ/training.log 2>&1 &
EQPID=$!; say "equivalence torchrun pid=$EQPID"
( stop_after_evals $EQPID $EQ/training.log 4 equivalence
  cp $EQ/training.log $O/train.log 2>/dev/null; cp $EQ/early_stop_history.json $O/train_equiv_early_stop_history.json 2>/dev/null
  $PY $CMP equiv >> $O/run.log 2>&1 ) &
EQWATCH=$!

# --- GPUs 0,1: eval A/B (old lib on GPU 0, new lib on GPU 1), judge smoke on GPU 0 ---
( t=$(date +%s); CUDA_VISIBLE_DEVICES=0 $PY $OLD/cpt_stage3_eval_baseline.py $EV --output_json $O/eval_old.json > $O/eval_old.log 2>&1; say "eval_old rc=$? secs=$(( $(date +%s)-t ))"
  J=$O/judge_smoke; mkdir -p $J; t=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 $PY -m track_a.esm_sft.judge train --task as_paired_i --adapter none --tcr_variant cdr3 --mhc_variant full --out_dir $J --smoke > $J/train.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=0 $PY -m track_a.esm_sft.judge eval --task as_paired_i --adapter none --tcr_variant cdr3 --mhc_variant full --ckpt $J/best.pt --out $J/eval.json --smoke > $J/eval.log 2>&1
  say "judge_smoke rc=$? secs=$(( $(date +%s)-t ))" ) &
A=$!
( t=$(date +%s); CUDA_VISIBLE_DEVICES=1 $PY scripts/training/cpt_stage3_eval_baseline.py $EV --output_json $O/eval_new.json > $O/eval_new.log 2>&1; say "eval_new rc=$? secs=$(( $(date +%s)-t ))" ) &
B=$!
wait $A $B
$PY $CMP ab >> $O/run.log 2>&1

# --- GPUs 0,1: resume check. Start once the equivalence run has finished tokenizing, so the
# second run reads the datasets cache instead of racing it. ---
while kill -0 $EQPID 2>/dev/null && ! grep -q 'Train rows:' $EQ/training.log 2>/dev/null; do sleep 15; done
rm -rf $RS; mkdir -p $RS
RARGS="$TRAIN --output_dir $RS --save_steps 500 --wandb_run_name stage43-resume-pointD-ws2 --resume_from auto"
CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29762 $RARGS >> $RS/training.log 2>&1 &
RPID=$!; t=$(date +%s); say "resume phase 1 torchrun pid=$RPID: train to the step-1000 checkpoint, then kill"
while kill -0 $RPID 2>/dev/null; do
  if [ -f $RS/checkpoint_step_1000/training_state.pt ]; then
    sleep 30; say "step-1000 checkpoint complete after $(( $(date +%s)-t ))s; killing pid $RPID"; kill -TERM $RPID; break
  fi
  sleep 15
done
while kill -0 $RPID 2>/dev/null; do sleep 5; done
CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29762 $RARGS >> $RS/training.log 2>&1 &
RPID=$!; say "resume phase 2 torchrun pid=$RPID: --resume_from auto, stop after the step-2000 eval"
stop_after_evals $RPID $RS/training.log 4 resume

cp $RS/training.log $O/resume_training.log 2>/dev/null; cp $RS/early_stop_history.json $O/resume_early_stop_history.json 2>/dev/null
wait $EQWATCH 2>/dev/null   # RESUME compares against the finished equivalence run
$PY $CMP resume >> $O/run.log 2>&1
say "all jobs done"
