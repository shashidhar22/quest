#!/bin/bash
# Phase 3 of a box visit: re-run the Stage 4.3 equivalence test under the v2 sampler (mixed
# steps + row-weighted loss), on the same box, after wave 1 — no second capacity queue. Installs
# the current code from S3 first; nothing else is running by then. Skips itself when the visit
# already ran its main equivalence test under v2 (a fresh launch with the fixed code).
set -u
export PATH=/home/ubuntu/miniforge3/bin:/usr/local/bin:$PATH
Q=${Q:-/home/ubuntu/quest}; O=${O:-/home/ubuntu/s43_out}
say(){ echo "[$(date -u '+%F %T')] $*" >> $O/run.log; }

if grep -aq 'sampler=v2 phases=all' $O/run.log 2>/dev/null; then
  say "phase 3 skipped: this visit's equivalence run already used the v2 sampler"; exit 0
fi
say "phase 3: installing current code and re-running the equivalence test under the v2 sampler"
aws s3 cp s3://fh-pi-warren-h-eco/quest/tcrbench2/_transfer/s43.tgz /tmp/s43.tgz --only-show-errors && tar xzf /tmp/s43.tgz -C $Q \
  || { say "phase 3 ABORT: could not install code"; exit 1; }
aws s3 cp s3://fh-pi-warren-h-eco/quest/tcrbench2/_transfer/s43box.tgz /tmp/s43box.tgz --only-show-errors && tar xzf /tmp/s43box.tgz -C /home/ubuntu \
  || { say "phase 3 ABORT: could not install kit"; exit 1; }
cd $Q && PYTHONPATH=$Q python -m pytest tests/test_token_budget_sampler.py tests/test_cpt_resume.py -q -p no:cacheprovider 2>&1 | tail -1 | sed 's/^/    on-box tests (v2): /' >> $O/run.log
PHASES=R2 bash /home/ubuntu/run_s43.sh
