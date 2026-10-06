#!/bin/bash
# Laptop dry run of run_s43.sh in BOTH layouts (4-GPU parallel and 1-GPU sequential), with stubs:
# a fake torchrun that writes the trainer's log lines, checkpoints and early_stop_history.json at
# high speed, and a fake python for the eval/judge commands (s43_compare.py runs for real).
# Exercises both watchers, the R->S checkpoint hand-off, the resume relaunch and all verdicts.
# Usage: bash dryrun.sh [python]   -> exits 0 only if every verdict PASSes in both layouts.
set -u
PYREAL=${1:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)

cat > /tmp/s43_stub_torchrun <<'EOF'
#!/bin/bash
# stub trainer: honours --output_dir/--save_steps/--resume_from, logs like CPTTrainer, 20 ms/step
trap 'exit 143' TERM
args=("$@"); out=""; save=500; resume=""
for ((i=0;i<${#args[@]};i++)); do case "${args[i]}" in --output_dir) out=${args[i+1]};; --save_steps) save=${args[i+1]};; --resume_from) resume=${args[i+1]};; esac; done
step=0; hist="[]"
if [ -n "$resume" ]; then
  [ "$resume" = auto ] && resume=$(ls -d $out/checkpoint_step_*/ 2>/dev/null | sed 's#/$##' | sort -t_ -k3 -n | tail -1)
  if [ -n "$resume" ] && [ -f $resume/training_state.pt ]; then
    step=$(basename $resume | sed 's/checkpoint_step_//')
    for ((k=500;k<=step;k+=500)); do hist=$(echo "$hist" | sed "s/\]$/, {\"metric\": 2.0}]/; s/^\[, /[/"); done
    echo "Resumed from $resume: global_step=$step, continuing at epoch 0 after $step of 7800 steps"
  fi
fi
echo "Train rows: 9,998,000"
while [ $step -lt 7800 ]; do
  step=$((step+1)); sleep 0.02
  if [ $((step % 500)) -eq 0 ]; then
    printf '[canonical_v2 eval @ step %d] overall_ppl=1.810  cdr3_ppl=2.890  framework_ppl=1.2\n' $step
    hist=$(echo "$hist" | sed "s/\]$/, {\"metric\": 2.2}]/; s/^\[, /[/")
    echo "{\"history\": $hist}" > $out/early_stop_history.json
  fi
  if [ $((step % save)) -eq 0 ]; then d=$out/checkpoint_step_$step; mkdir -p $d; echo x > $d/adapter_model.safetensors; echo x > $d/training_state.pt; echo "  saved checkpoint: $d"; fi
done
EOF
chmod +x /tmp/s43_stub_torchrun

one_layout(){   # one_layout <ngpu> <label>
  local NG=$1 label=$2
  local W=$(mktemp -d) Q O OLD
  Q=$W/quest; O=$W/out; OLD=$W/old
  mkdir -p $Q/checkpoints/cpt_stage41_pointD_run05 $Q/scripts/training $Q/track_a $O $OLD $W/bin
  printf '[canonical_v2 eval @ step 2000] overall_ppl=1.800  cdr3_ppl=2.900  framework_ppl=1.2\n' > $Q/checkpoints/cpt_stage41_pointD_run05/training.log
  echo '{"history": [{"metric": 3.0}, {"metric": 2.5}, {"metric": 2.2}, {"metric": 2.0}]}' > $Q/checkpoints/cpt_stage41_pointD_run05/early_stop_history.json
  cp /tmp/s43_stub_torchrun $W/bin/torchrun
  cat > $W/bin/python <<EOF
#!/bin/bash
case "\$1" in
  *cpt_stage3_eval_baseline.py) for ((i=1;i<=\$#;i++)); do [ "\${!i}" = --output_json ] && { j=\$((i+1)); echo '{"regions": {"overall": {"overall": {"_": {"loss": 0.5}}}}}' > "\${!j}"; }; done ;;
  -m) exit 0 ;;
  *s43_compare.py) exec $PYREAL "\$@" ;;
esac
EOF
  chmod +x $W/bin/python
  touch $OLD/cpt_stage3_eval_baseline.py $Q/scripts/training/cpt_stage3_eval_baseline.py
  sed -e 's/sleep 90/sleep 1/; s/sleep 30/sleep 1/; s/sleep 15/sleep 1/; s/sleep 5/sleep 1/' $HERE/run_s43.sh > $W/run_s43.sh
  NGPU=$NG Q=$Q O=$O OLD=$OLD CMP=$HERE/s43_compare.py TORCHRUN=$W/bin/torchrun PY=$W/bin/python bash $W/run_s43.sh
  echo "=== [$label] run.log ==="; cat $O/run.log
  local n_pass n_bad
  n_pass=$(grep -cE '^(EVAL_AB|TRAIN_EQUIV|RESUME_LOG|RESUME) PASS' $O/run.log)
  n_bad=$(grep -cE '^(EVAL_AB|TRAIN_EQUIV|RESUME_LOG|RESUME) (FAIL|ERROR)' $O/run.log)
  echo "[$label] verdicts: $n_pass PASS, $n_bad FAIL/ERROR (expect 4 PASS, 0 bad)"
  grep -q "world_size=$3" $O/run.log || { echo "[$label] WRONG LAYOUT: expected world_size=$3"; return 1; }
  [ "$n_pass" -eq 4 ] && [ "$n_bad" -eq 0 ]
}

rc=0
one_layout 4 "4-GPU parallel" 2 || rc=1
one_layout 1 "1-GPU sequential" 1 || rc=1
echo "=== dryrun overall: $([ $rc -eq 0 ] && echo PASS || echo FAIL) ==="
exit $rc
