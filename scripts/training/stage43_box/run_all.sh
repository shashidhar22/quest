#!/bin/bash
# One job for the whole box visit: the Stage 4.3 checks, then Stage 4.2 wave 1. The watchdog waits
# on THIS pid, so the box lives until both are done (or a ceiling fires) and is never released and
# re-queued for GPU capacity in between — it took 25 h to get a box on 2026-10-06.
set -u
O43=${O43:-/home/ubuntu/s43_out}; O42=${O42:-/home/ubuntu/s42_out}
S43=${S43:-/home/ubuntu/run_s43.sh}; WAVE1=${WAVE1:-/home/ubuntu/wave1.sh}
mkdir -p $O43 $O42
echo $$ > $O43/run_all.pid    # the watchdog waits on THIS pid, never a name pattern
say(){ echo "[$(date -u '+%F %T')] $*" >> $O42/chain.log; }

say "run_all start pid=$$ — Stage 4.3 first"
bash $S43
say "Stage 4.3 exited rc=$?; its verdicts:"
grep -hE '^(EVAL_AB|TRAIN_EQUIV|RESUME_LOG|RESUME) ' $O43/run.log 2>/dev/null | sed 's/^/    /' >> $O42/chain.log
say "starting Stage 4.2 wave 1"
bash $WAVE1
say "wave 1 exited rc=$?"
