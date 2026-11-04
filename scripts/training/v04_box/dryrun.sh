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
#
# ISOLATION (repaired 2026-11-04 after the external Stage-4 audit). Everything this script writes
# lives under its own mktemp directory, including the checkpoint root. The previous version pointed
# the launcher at the live checkout, wrote stub training logs and manifests into
# `checkpoints/v04_arm*_seed*`, and then removed that glob -- which after a real visit is exactly
# the six trained adapters the visit was bought to produce, including seeds it had not created.
# Nothing here may write into, or delete from, $REPO. The only repo paths read are fixtures.
#
# These are branch tests of the decision chain, not an end-to-end validation: the trainer and the
# judge are stubs, so a PASS says nothing about GPU behaviour, throughput or numerics.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PYREAL=${1:-python3}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
CKPT=$TMP/checkpoints
fail(){ echo "[dryrun] FAIL: $*"; exit 1; }

# A candidate pool, because an absent pool is no longer a runnable visit: it used to become the
# string POOL_FILE_MISSING in every manifest and so matched itself.
POOL=$TMP/pool.tsv
printf 'peptide\n' > $POOL
for i in $(seq 1 120); do printf 'PEPTIDE%03d\n' "$i" >> $POOL; done

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
  Q=$REPO O="$2" CKPT_ROOT=$CKPT PY=$TMP/stub_python TORCHRUN=$TMP/stub_torchrun NGPU=4 \
    SEEDS="1 2 3" RHO_FLOOR=$1 BASE_CKPT=/dev/null POOL_FILE=$POOL \
    bash "$HERE/run_v04.sh" > "$2/stdout.log" 2>&1
  echo $?
}

# A comparison cell now refuses to evaluate a checkpoint that no step produced. Until
# train-then-evaluate is wired for cells 2-4, the dry run supplies them, so the cell path is still
# exercised; the branch below checks the refusal itself.
seed_cells(){ for c in armD_paired base_trb armD_trb; do mkdir -p "$1/$c"; : > "$1/$c/best.pt"; done; }

echo "[dryrun] --- branch 1: floor 0.80, above the measured breakeven 0.765 -> expect PASS"
mkdir -p $TMP/pass; seed_cells $TMP/pass
rc=$(run_v04 0.8 $TMP/pass)
[ "$rc" -eq 0 ] || fail "the passing branch exited $rc (see $TMP/pass/stdout.log)"
grep -q "READABILITY PASS on the LOCKED primary read test_novel_pep / peptide / norm_rank" $TMP/pass/run.log \
  || fail "locked-read gate did not pass at floor 0.80"
grep -q "G1 PASSED: running comparison cells" $TMP/pass/run.log || fail "cells not launched on a pass"
[ -f $TMP/pass/armD_paired/eval.json ] || fail "comparison cell produced no eval"
[ -f $TMP/pass/armD_paired/manifest.json ] || fail "comparison cell wrote no manifest"
grep -q "V04 complete" $TMP/pass/run.log || fail "the passing branch did not complete"
echo "[dryrun]     PASS branch ok: gate passed, 3 cells ran, manifests written, visit completed"

echo "[dryrun] --- the trb cells must select the trb task, not the paired one"
grep -q "judge base_trb rc=0 task=as_trb_i" $TMP/pass/run.log \
  || fail "base_trb did not run as_trb_i"
grep -q "judge armD_trb rc=0 task=as_trb_i" $TMP/pass/run.log \
  || fail "armD_trb did not run as_trb_i"
$PYREAL -c "
import json
m=json.load(open('$TMP/pass/base_trb/manifest.json'))
assert m['dataset']=='as_trb_i', m
" || fail "the base_trb manifest does not name the trb task"
echo "[dryrun]     both trb cells select as_trb_i, in the command and in the manifest"

echo "[dryrun] --- a tolerated provenance difference must be printed, never silent"
grep -q "TOLERATED by kind=cells: cotenants" $TMP/pass/run.log \
  || fail "the base-vs-armD co-tenancy waiver was not disclosed"
echo "[dryrun]     the co-tenancy waiver is disclosed next to the result it licensed"

echo "[dryrun] --- branch 2: floor 0.70, below the breakeven -> expect refusal"
rc=$(run_v04 0.7 $TMP/refuse)
[ "$rc" -eq 0 ] || fail "a scientific refusal was reported as a run failure (rc=$rc)"
grep -q "READABILITY FAIL on the LOCKED primary read" $TMP/refuse/run.log \
  || fail "gate did not fail at floor 0.70"
grep -q "G1 FAILED: comparison cells 2-4 SKIPPED" $TMP/refuse/run.log || fail "cells not skipped on a fail"
[ -f $TMP/refuse/armD_paired/eval.json ] && fail "a comparison cell ran despite the refusal"
grep -q "pre-agreed refusal, not a run failure" $TMP/refuse/run.log || fail "refusal not explained"
echo "[dryrun]     REFUSE branch ok: gate failed, cells skipped, reason recorded, exit 0"

echo "[dryrun] --- an unbuilt comparison cell must refuse, and fail the visit"
rc=$(run_v04 0.8 $TMP/nockpt)
[ "$rc" -ne 0 ] || fail "the visit succeeded with no trained comparison checkpoint"
grep -q "REFUSED cell armD_paired: no trained comparison checkpoint" $TMP/nockpt/run.log \
  || fail "the missing comparison checkpoint was not refused"
grep -q "V04 FAILED" $TMP/nockpt/run.log || fail "the visit did not report failure"
grep -q "V04 complete" $TMP/nockpt/run.log && fail "the visit announced completion as well as failure"
echo "[dryrun]     an unbuilt cell refuses and the visit fails"

echo "[dryrun] --- the arm re-baselines ran in every branch, with per-seed manifests"
for s in 1 2 3; do
  for a in armA armD; do
    m=$CKPT/v04_${a}_seed${s}/manifest.json
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

echo "[dryrun] --- nothing may have been written into the repo checkpoint tree"
if compgen -G "$REPO/checkpoints/v04_arm*_seed*" >/dev/null; then
  fail "the dry run wrote into $REPO/checkpoints: $(echo $REPO/checkpoints/v04_arm*_seed*)"
fi
echo "[dryrun]     repo checkpoint tree untouched"

echo "[dryrun] --- provenance must refuse a draw mismatch"
$PYREAL $HERE/provenance.py write --out $TMP/u.json --draw uniform --pool-size 100 --objective token >/dev/null
$PYREAL $HERE/provenance.py write --out $TMP/l.json --draw length_matched --pool-size 100 --objective token >/dev/null
if $PYREAL $HERE/provenance.py check --a $TMP/u.json --b $TMP/l.json --kind strict >/dev/null 2>&1; then
  fail "provenance accepted a uniform-vs-length_matched comparison"
fi
echo "[dryrun]     provenance refuses the draw mismatch"

echo "[dryrun] --- the rejected objective must abort the visit"
out=$(Q=$REPO O=$TMP/rej CKPT_ROOT=$CKPT PY=$TMP/stub_python TORCHRUN=$TMP/stub_torchrun NGPU=4 \
      POOL_FILE=$POOL LOSS_WEIGHTING=row bash "$HERE/run_v04.sh" 2>&1); rc=$?
[ "$rc" -eq 2 ] || fail "row-share did not abort (rc=$rc)"
echo "$out" | grep -q "rejected for production" || fail "abort gave no reason"
echo "[dryrun]     row-share aborts with rc=2"

echo "[dryrun] --- an absent candidate pool must abort, not become a matching sentinel"
out=$(Q=$REPO O=$TMP/nopool CKPT_ROOT=$CKPT PY=$TMP/stub_python TORCHRUN=$TMP/stub_torchrun NGPU=4 \
      POOL_FILE=$TMP/does_not_exist.tsv bash "$HERE/run_v04.sh" 2>&1); rc=$?
[ "$rc" -eq 2 ] || fail "a missing candidate pool did not abort (rc=$rc)"
echo "$out" | grep -q "candidate pool not found" || fail "the missing pool gave no reason"
echo "[dryrun]     a missing candidate pool aborts with rc=2"

echo "[dryrun] --- fewer than 4 GPUs must abort rather than silently share GPU 0"
out=$(Q=$REPO O=$TMP/gpu CKPT_ROOT=$CKPT PY=$TMP/stub_python TORCHRUN=$TMP/stub_torchrun NGPU=2 \
      POOL_FILE=$POOL bash "$HERE/run_v04.sh" 2>&1); rc=$?
[ "$rc" -eq 2 ] || fail "a 2-GPU box did not abort (rc=$rc)"
echo "$out" | grep -q "needs 4 GPUs" || fail "the GPU count abort gave no reason"
echo "[dryrun]     a 2-GPU box aborts with rc=2"

echo "[dryrun] overall: PASS"
