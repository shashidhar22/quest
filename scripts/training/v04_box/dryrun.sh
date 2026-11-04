#!/bin/bash
# Laptop dry run of run_v04.sh. Stubs the trainer and the judge; runs provenance.py,
# s42_check.py and readability.py FOR REAL, because those are what decide the visit.
#
# Exercises both gate branches with the real V03 ranks rather than synthetic ones, by moving the
# rho floor across the measured breakeven of 0.765:
#   floor 0.80 -> G1 PASSES, comparison cells run
#   floor 0.70 -> G1 FAILS,  comparison cells are skipped
# and checks that provenance refuses a comparison across a draw mismatch.
# Exits 0 only if every expectation holds.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PYREAL=${1:-python3}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fail(){ echo "[dryrun] FAIL: $*"; exit 1; }

cat > $TMP/stub_torchrun <<'EOF'
#!/bin/bash
out=""; args=("$@")
for ((i=0;i<${#args[@]};i++)); do [ "${args[i]}" = --output_dir ] && out=${args[i+1]}; done
mkdir -p "$out"; echo "stub trainer: 7800 steps" > "$out/training.log"; exit 0
EOF
# Stub python: fakes ONLY the judge; everything else is the real interpreter, so the gate,
# the predicate and provenance are exercised rather than mocked.
cat > $TMP/stub_python <<EOF
#!/bin/bash
if [ "\$1" = "-m" ] && [ "\$2" = "track_a.esm_sft.judge" ]; then
  out=""; args=("\$@")
  for ((i=0;i<\${#args[@]};i++)); do [ "\${args[i]}" = --out ] && out=\${args[i+1]}; done
  mkdir -p "\$(dirname "\$out")"
  cp "$REPO/tests/fixtures/s42_eval_base_paired_ranks_20261008.json" "\$out"
  exit 0
fi
exec $PYREAL "\$@"
EOF
chmod +x $TMP/stub_torchrun $TMP/stub_python

run_v04(){   # run_v04 <floor> <outdir>
  mkdir -p "$2"
  Q=$REPO O="$2" PY=$TMP/stub_python TORCHRUN=$TMP/stub_torchrun NGPU=4 \
    SEEDS="1 2 3" RHO_FLOOR=$1 BASE_CKPT=/dev/null \
    bash "$HERE/run_v04.sh" > "$2/stdout.log" 2>&1
  echo $?
}

echo "[dryrun] --- branch 1: floor 0.80, above the measured breakeven 0.765 -> expect PASS"
run_v04 0.8 $TMP/pass >/dev/null
grep -q "READABILITY PASS on the LOCKED primary read test_novel_pep / peptide / norm_rank" $TMP/pass/run.log \
  || fail "locked-read gate did not pass at floor 0.80"
grep -q "G1 PASSED: running comparison cells" $TMP/pass/run.log || fail "cells not launched on a pass"
[ -f $TMP/pass/armD_paired/eval.json ] || fail "comparison cell produced no eval"
[ -f $TMP/pass/armD_paired/manifest.json ] || fail "comparison cell wrote no manifest"
echo "[dryrun]     PASS branch ok: gate passed, 3 cells ran, manifests written"

echo "[dryrun] --- branch 2: floor 0.70, below the breakeven -> expect refusal"
run_v04 0.7 $TMP/refuse >/dev/null
grep -q "READABILITY FAIL on the LOCKED primary read" $TMP/refuse/run.log \
  || fail "gate did not fail at floor 0.70"
grep -q "G1 FAILED: comparison cells 2-4 SKIPPED" $TMP/refuse/run.log || fail "cells not skipped on a fail"
[ -f $TMP/refuse/armD_paired/eval.json ] && fail "a comparison cell ran despite the refusal"
grep -q "pre-agreed refusal, not a run failure" $TMP/refuse/run.log || fail "refusal not explained"
echo "[dryrun]     REFUSE branch ok: gate failed, cells skipped, reason recorded"

echo "[dryrun] --- the arm re-baselines ran in both branches, with per-seed manifests"
for s in 1 2 3; do
  for a in armA armD; do
    m=$REPO/checkpoints/v04_${a}_seed${s}/manifest.json
    [ -f "$m" ] || fail "no manifest for ${a} seed ${s}"
    $PYREAL -c "
import json,sys
m=json.load(open('$m'))
assert m['objective']=='token', m
assert m['early_stop']=='off', m
assert m['seed']=='$s', m
assert 'seeds 1 2 3' in m['cotenants'], m
" || fail "manifest for ${a} seed ${s} does not record the objective, the stop or the co-tenancy"
  done
done
echo "[dryrun]     6 arm manifests ok: objective=token, early_stop=off, seed and co-tenancy recorded"

echo "[dryrun] --- provenance must refuse a draw mismatch"
$PYREAL $HERE/provenance.py write --out $TMP/u.json --draw uniform --pool-size 100 --objective token >/dev/null
$PYREAL $HERE/provenance.py write --out $TMP/l.json --draw length_matched --pool-size 100 --objective token >/dev/null
if $PYREAL $HERE/provenance.py check --a $TMP/u.json --b $TMP/l.json --kind strict >/dev/null 2>&1; then
  fail "provenance accepted a uniform-vs-length_matched comparison"
fi
echo "[dryrun]     provenance refuses the draw mismatch"

echo "[dryrun] --- the rejected objective must abort the visit"
out=$(Q=$REPO O=$TMP/rej PY=$TMP/stub_python TORCHRUN=$TMP/stub_torchrun NGPU=4 \
      LOSS_WEIGHTING=row bash "$HERE/run_v04.sh" 2>&1); rc=$?
[ "$rc" -eq 2 ] || fail "row-share did not abort (rc=$rc)"
echo "$out" | grep -q "rejected for production" || fail "abort gave no reason"
echo "[dryrun]     row-share aborts with rc=2"

rm -rf $REPO/checkpoints/v04_arm*_seed*
echo "[dryrun] overall: PASS"
