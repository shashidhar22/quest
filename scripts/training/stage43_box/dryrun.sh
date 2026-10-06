#!/bin/bash
# Laptop dry run of run_s43.sh with stubs: a fake torchrun that writes the trainer's log lines,
# checkpoints and early_stop_history.json at high speed, and a fake python for the eval/judge
# commands (the compare script runs under the real python). Exercises every branch of the flow:
# both watchers, the step-1000 kill, the resume relaunch, and all five verdict lines.
# Usage: bash dryrun.sh [python]   -> prints run.log and exits 0 only if every verdict PASSes.
set -u
PYREAL=${1:-python3}
HERE=$(cd "$(dirname "$0")" && pwd); W=$(mktemp -d); Q=$W/quest; O=$W/out; OLD=$W/old
mkdir -p $Q/checkpoints/cpt_stage41_pointD_run05 $Q/scripts/training $Q/track_a $O $OLD $W/bin

# Reference "arm 2" run: step-2000 eval line + history (4th entry = pocket at step 2000).
printf '[canonical_v2 eval @ step 2000] overall_ppl=1.800  cdr3_ppl=2.900  framework_ppl=1.2\n' > $Q/checkpoints/cpt_stage41_pointD_run05/training.log
echo '{"history": [{"metric": 3.0}, {"metric": 2.5}, {"metric": 2.2}, {"metric": 2.0}]}' > $Q/checkpoints/cpt_stage41_pointD_run05/early_stop_history.json

cat > $W/bin/torchrun <<'EOF'
#!/bin/bash
# stub trainer: parses --output_dir/--save_steps/--resume_from, logs like CPTTrainer, 1 step/20 ms
trap 'exit 143' TERM
args=("$@"); out=""; save=2000; resume=""
for ((i=0;i<${#args[@]};i++)); do case "${args[i]}" in --output_dir) out=${args[i+1]};; --save_steps) save=${args[i+1]};; --resume_from) resume=${args[i+1]};; esac; done
step=0; hist="[]"
if [ "$resume" = auto ]; then
  latest=$(ls -d $out/checkpoint_step_*/ 2>/dev/null | sed 's#.*checkpoint_step_\([0-9]*\)/#\1#' | sort -n | tail -1)
  if [ -n "$latest" ] && [ -f $out/checkpoint_step_$latest/training_state.pt ]; then
    step=$latest; hist=$(cat $out/early_stop_history.json | sed 's/.*"history": \(\[.*\]\)}/\1/')
    echo "Resumed from $out/checkpoint_step_$latest: global_step=$latest, continuing at epoch 0 after $latest of 7800 steps"
  fi
fi
echo "Train rows: 9,998,000"
while [ $step -lt 7800 ]; do
  step=$((step+1)); sleep 0.02
  if [ $((step % 500)) -eq 0 ]; then
    m=$(printf '%.4f' "$(echo "2.0 + 0.0001*$step" | bc -l)")   # pocket metric
    printf '[canonical_v2 eval @ step %d] overall_ppl=1.810  cdr3_ppl=2.890  framework_ppl=1.2\n' $step
    hist=$(echo "$hist" | sed "s/\]$/, {\"metric\": $m}]/; s/^\[, /[/")
    echo "{\"history\": $hist}" > $out/early_stop_history.json
  fi
  if [ $((step % save)) -eq 0 ]; then d=$out/checkpoint_step_$step; mkdir -p $d; echo x > $d/adapter_model.safetensors; echo x > $d/training_state.pt; echo "  saved checkpoint: $d"; fi
done
EOF
cat > $W/bin/python <<EOF
#!/bin/bash
# stub python: eval scripts write a json; judge exits 0; the compare script runs for real
case "\$1" in
  *cpt_stage3_eval_baseline.py) for ((i=1;i<=\$#;i++)); do [ "\${!i}" = --output_json ] && { j=\$((i+1)); echo '{"regions": {"overall": {"overall": {"_": {"loss": 0.5}}}}}' > "\${!j}"; }; done ;;
  -m) exit 0 ;;
  *s43_compare.py) exec $PYREAL "\$@" ;;
esac
EOF
chmod +x $W/bin/torchrun $W/bin/python
touch $OLD/cpt_stage3_eval_baseline.py $Q/scripts/training/cpt_stage3_eval_baseline.py

# Shorten the waits so the dry run finishes in ~2 min.
sed -e 's/sleep 90/sleep 1/; s/sleep 30/sleep 1/; s/sleep 15/sleep 1/; s/sleep 5/sleep 1/' $HERE/run_s43.sh > $W/run_s43.sh
Q=$Q O=$O OLD=$OLD CMP=$HERE/s43_compare.py TORCHRUN=$W/bin/torchrun PY=$W/bin/python bash $W/run_s43.sh
echo "=== run.log ==="; cat $O/run.log
echo "=== resume_training.log (resume lines) ==="; grep -n "Resumed from\|eval @ step" $O/resume_training.log
n_pass=$(grep -cE '^(EVAL_AB|TRAIN_EQUIV|RESUME_LOG|RESUME) PASS' $O/run.log); n_bad=$(grep -cE '^(EVAL_AB|TRAIN_EQUIV|RESUME_LOG|RESUME) (FAIL|ERROR)' $O/run.log)
echo "verdicts: $n_pass PASS, $n_bad FAIL/ERROR (expect 4 PASS)"; [ "$n_pass" -eq 4 ] && [ "$n_bad" -eq 0 ]
