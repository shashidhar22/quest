#!/bin/bash
# Stage 4.3 on-box checks: (1) eval A/B old vs new on GPU 0/1, (2) token-budget training of the
# arm-2 config on GPUs 2,3 (world size 2) stopped after its 4th eval (= step 2000).
set -u
export PATH=/home/ubuntu/miniforge3/bin:/usr/local/bin:$PATH PYTHONPATH=/home/ubuntu/quest
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
Q=/home/ubuntu/quest; O=/home/ubuntu/s43_out; mkdir -p $O; cd $Q
say(){ echo "[$(date -u '+%F %T')] $*" >> $O/run.log; }
EV="--adapter_dir $Q/checkpoints/cpt_stage41_pointD_run05/final --canonical_eval_path $Q/data/cpt_canonical_eval_v2 --base_model esmc_300m --tcr_variant cdr3 --mhc_variant full --eos_strategy raw --max_length 1024 --batch_size 16"
echo $$ > $O/run_s43.pid   # the watchdog waits on THIS pid, never a name pattern
say "start pid=$$"
( t=$(date +%s); CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/s43_old/cpt_stage3_eval_baseline.py $EV --output_json $O/eval_old.json > $O/eval_old.log 2>&1; say "eval_old rc=$? secs=$(( $(date +%s)-t ))"
  # Stage 4.2 smoke on the freed GPU 0: base arm (fresh LoRA), cdr3 x full, tiny train + eval.
  J=$O/judge_smoke; mkdir -p $J; t=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python -m track_a.esm_sft.judge train --task as_paired_i --adapter none --tcr_variant cdr3 --mhc_variant full --out_dir $J --smoke > $J/train.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=0 python -m track_a.esm_sft.judge eval --task as_paired_i --adapter none --tcr_variant cdr3 --mhc_variant full --ckpt $J/best.pt --out $J/eval.json --smoke > $J/eval.log 2>&1
  say "judge_smoke rc=$? secs=$(( $(date +%s)-t ))" ) &
( t=$(date +%s); CUDA_VISIBLE_DEVICES=1 python scripts/training/cpt_stage3_eval_baseline.py $EV --output_json $O/eval_new.json > $O/eval_new.log 2>&1; say "eval_new rc=$? secs=$(( $(date +%s)-t ))" ) &
OUT=$Q/checkpoints/s43_equiv_pointD
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29761 scripts/training/train_cpt_stage2_single.py \
  --dataset_path $Q/data/cpt_datasets/run05_cdr3_fullmhc_proportional_10M --output_dir $OUT \
  --tcr_variant cdr3 --mhc_variant full --max_length 1024 --batch_size 16 --gradient_accumulation_steps 32 \
  --token_budget 16384 --bucket_boundaries 64 128 256 512 1024 --wandb_run_name stage43-equiv-pointD-ws2 \
  --base_model esmc_300m --canonical_eval_path $Q/data/cpt_canonical_eval_v2 --eos_strategy raw --num_epochs 1 \
  --lr 2e-4 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --eval_steps 500 --save_steps 2000 --logging_steps 50 \
  --no_compile --attn_implementation sdpa --wandb_project tcrbench-v3-cpt-stage43 --report_to wandb \
  --early_stop_n_patience 3 --early_stop_min_rel 0.01 --early_stop_metric_path pocket overall _ perplexity \
  > $O/train.log 2>&1 &
TPID=$!; t0=$(date +%s); say "train torchrun pid=$TPID"
while kill -0 $TPID 2>/dev/null; do
  n=$(grep -c 'canonical_v2 eval @ step' $OUT/training.log 2>/dev/null || echo 0)
  if [ "${n:-0}" -ge 4 ]; then say "4 evals done after $(( $(date +%s)-t0 ))s; stopping training"; sleep 90; kill -TERM $TPID; break; fi
  sleep 30
done
wait
say "all jobs done"
python /home/ubuntu/s43_compare.py >> $O/run.log 2>&1
say "compare rc=$?"
