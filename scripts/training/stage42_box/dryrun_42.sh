#!/bin/bash
# Laptop dry run of the Stage 4.2 wave-1 flow with a stub judge. Covers:
#   A. gate PASS  -> 4 cells in decision order, 2 paired comparisons, report table (1 GPU and 4 GPU)
#   B. gate FAIL  -> stops after cell 1; cells 2-4 never run (the money-saving branch)
#   C. run_all    -> runs a stub Stage 4.3, carries its verdicts forward, then runs wave 1
#   D. resume     -> a finished cell is skipped on a re-run
# Usage: bash dryrun_42.sh [python]   -> exits 0 only if every expectation holds.
set -u
PYREAL=${1:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
fail=0
expect(){ if [ "$2" = "$3" ]; then echo "  ok: $1"; else echo "  FAIL: $1 (got '$2', want '$3')"; fail=1; fi; }

make_stub(){   # make_stub <dir> <iid_recall> <iid_ci_lo>
  cat > $1/python <<EOF
#!/usr/bin/env bash
# stub judge: train writes a checkpoint; eval writes judge-shaped json; compare prints one line.
if [ "\$1" = "-m" ]; then
  sub=\$3
  args=("\$@"); out=""; outdir=""; task=""; adapter=""
  for ((i=0;i<\${#args[@]};i++)); do case "\${args[i]}" in
    --out) out=\${args[i+1]};; --out_dir) outdir=\${args[i+1]};; --task) task=\${args[i+1]};; --adapter) adapter=\${args[i+1]};; esac; done
  case "\$sub" in
    probe) echo "[probe] \${PROBE_VERDICT:-PASS}: scores vary with the peptide"; [ "\${PROBE_VERDICT:-PASS}" = PASS ] && exit 0 || exit 1;;
    train) sleep 0.3; echo x > \$outdir/best.pt; echo "{\"task\":\"\$task\"}" > \$outdir/train_summary.json; exit 0;;
    eval)  sleep 0.3
           if [ "\$adapter" = none ]; then R=0.42; LO=$2; HI=0.50; IID=$2; IIDLO=$3
           else R=0.55; LO=0.50; HI=0.60; IID=0.60; IIDLO=0.55; fi
           cat > \$out <<JSON
{"pool_size": 100, "conditions": {
 "test_level4":    {"n_queries": 4872, "recall_at_10": \$R, "recall_at_10_ci95": [\$LO, \$HI], "mrr": 0.3, "random_recall_at_10": 0.1, "underpowered": false, "ranks": {"r1": 3, "r2": 44}},
 "test_novel_pep": {"n_queries": 1500, "recall_at_10": \$R, "recall_at_10_ci95": [\$LO, \$HI], "mrr": 0.3, "random_recall_at_10": 0.1, "underpowered": false, "ranks": {"r1": 5, "r2": 51}},
 "test_iid":       {"n_queries": 3000, "recall_at_10": \$IID, "recall_at_10_ci95": [\$IIDLO, 0.70], "mrr": 0.4, "random_recall_at_10": 0.1, "underpowered": false, "ranks": {"r1": 2, "r2": 9}}}}
JSON
           exit 0;;
    compare) echo "test_level4: paired n=2  delta recall@10 (B-A) = +0.5000  95% CI [+0.1000, +0.9000]  -> B > A"; exit 0;;
  esac
fi
exec $PYREAL "\$@"
EOF
  chmod +x $1/python
}

run_wave1(){   # run_wave1 <label> <iid_recall> <iid_ci_lo> [ngpu]
  W=$(mktemp -d); mkdir -p $W/bin $W/quest/checkpoints/cpt_stage41_pointD_run05/final $W/out
  make_stub $W/bin "$2" "$3"
  Q=$W/quest O=$W/out PY=$W/bin/python CHECK=$HERE/s42_check.py NO_PUBLISH=1 NGPU=${4:-1} \
    bash $HERE/wave1.sh; echo "wave1 rc=$?"
  echo "--- [$1] wave1.log ---"; cat $W/out/wave1.log
  CELLS=$(ls $W/out | grep -c '^\(base\|armD\)_' || true)
}

echo "=== A. gate PASS: all four cells run (1 GPU) ==="
run_wave1 pass 0.60 0.55 1
expect "cells run" "$CELLS" "4"
expect "paired compare written" "$([ -f $W/out/compare_paired.txt ] && echo yes || echo no)" "yes"
expect "trb compare written" "$([ -f $W/out/compare_trb.txt ] && echo yes || echo no)" "yes"
expect "gate logged PASS" "$(grep -c 'GATE PASS' $W/out/wave1.log)" "1"
expect "sequential on 1 GPU" "$(grep -c 'in parallel' $W/out/wave1.log)" "0"
expect "every cell on gpu 0" "$(grep -c 'train rc=0 .* gpu=0' $W/out/wave1.log)" "4"
expect "report table printed" "$(grep -c '^arm *task *condition' $W/out/wave1.log)" "1"
expect "report has a row per cell x condition" "$(grep -cE '^(base|arm D) +as_' $W/out/wave1.log)" "12"
PASSDIR=$W

echo "=== A4. gate PASS on 4 GPUs: cell 1 alone, then cells 2-4 side by side ==="
run_wave1 pass4 0.60 0.55 4
expect "cells run" "$CELLS" "4"
expect "parallel branch taken" "$(grep -c 'in parallel on GPUs 1,2,3' $W/out/wave1.log)" "1"
expect "cells 2-4 on gpus 1,2,3" "$(grep -oE 'train rc=0 .* gpu=[123]' $W/out/wave1.log | sort -u | wc -l | tr -d ' ')" "3"
expect "both compares written" "$(ls $W/out/compare_*.txt | wc -l | tr -d ' ')" "2"

echo "=== D. re-run skips finished cells ==="
Q=$PASSDIR/quest O=$PASSDIR/out PY=$PASSDIR/bin/python CHECK=$HERE/s42_check.py NO_PUBLISH=1 bash $HERE/wave1.sh >/dev/null 2>&1
expect "all four cells skipped on re-run" "$(grep -c 'already done, skipping' $PASSDIR/out/wave1.log)" "4"

echo "=== P. probe FAIL: scorer ignores its input -> abort BEFORE cell 1 (rc 5) ==="
W=$(mktemp -d); mkdir -p $W/bin $W/quest/checkpoints/cpt_stage41_pointD_run05/final $W/out
make_stub $W/bin 0.60 0.55
PROBE_VERDICT=FAIL Q=$W/quest O=$W/out PY=$W/bin/python CHECK=$HERE/s42_check.py NO_PUBLISH=1 NGPU=1 bash $HERE/wave1.sh; rc=$?
expect "wave1 exit code 5" "$rc" "5"
expect "no cell ran" "$(ls $W/out | grep -c '^\(base\|armD\)_' || true)" "0"
expect "abort logged" "$(grep -c 'ABORT: probe failed' $W/out/wave1.log)" "1"

echo "=== B. gate FAIL: base at the random rate -> stop after cell 1 ==="
run_wave1 fail 0.10 0.08
expect "only cell 1 ran" "$CELLS" "1"
expect "gate logged FAIL" "$(grep -c 'GATE FAIL' $W/out/wave1.log)" "1"
expect "no paired compare" "$([ -f $W/out/compare_paired.txt ] && echo yes || echo no)" "no"

echo "=== C. run_all: a stub Stage 4.3, then wave 1, under one pid ==="
W=$(mktemp -d); mkdir -p $W/bin $W/quest/checkpoints/cpt_stage41_pointD_run05/final $W/out $W/o43
make_stub $W/bin 0.60 0.55
cat > $W/fake_s43.sh <<EOF
sleep 2; echo "EVAL_AB PASS (exactly equal); 0 differing leaves" >> $W/o43/run.log; echo "TRAIN_EQUIV PASS @2000" >> $W/o43/run.log
EOF
cat > $W/wave1_wrap.sh <<EOF
Q=$W/quest O=$W/out PY=$W/bin/python CHECK=$HERE/s42_check.py NO_PUBLISH=1 NGPU=1 bash $HERE/wave1.sh
EOF
O43=$W/o43 O42=$W/out S43=$W/fake_s43.sh WAVE1=$W/wave1_wrap.sh bash $HERE/../stage43_box/run_all.sh
echo "--- chain.log ---"; cat $W/out/chain.log
expect "run_all wrote its pid" "$([ -s $W/o43/run_all.pid ] && echo yes || echo no)" "yes"
expect "4.3 ran first" "$(grep -c 'Stage 4.3 exited rc=0' $W/out/chain.log)" "1"
expect "4.3 verdicts carried forward" "$(grep -c 'TRAIN_EQUIV PASS' $W/out/chain.log)" "1"
expect "wave 1 ran after 4.3" "$(ls $W/out | grep -c '^\(base\|armD\)_')" "4"
expect "wave 1 exit logged" "$(grep -c 'wave 1 exited rc=0' $W/out/chain.log)" "1"

echo; [ $fail -eq 0 ] && echo "=== dryrun_42 PASS ===" || echo "=== dryrun_42 FAIL ==="
exit $fail
