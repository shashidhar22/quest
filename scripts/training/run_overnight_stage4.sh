#!/bin/bash
# TCRBench v3 — Stage 4 combined overnight pipeline orchestrator.
# Resumable: each phase / point is skipped if its done-marker is PASS.
set -u

cd /home/ubuntu/quest

LOG=/home/ubuntu/quest/checkpoints/stage4_combined_DRIVER.log
mkdir -p /home/ubuntu/quest/checkpoints
touch "$LOG"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo "[$(ts)] === $* ===" | tee -a "$LOG"; }
note()   { echo "[$(ts)] $*" | tee -a "$LOG"; }

banner "Pipeline start"

# --- Pre-flight ---
banner "Pre-flight checks"
bash scripts/training/stage4_preflight.sh 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  note "Pre-flight FAILED — aborting"
  exit 1
fi

# --- Phase 1: cluster recovery ---
if [ -f /home/ubuntu/quest/checkpoints/_phase1_done ] && \
   grep -q "STATUS=PASS" /home/ubuntu/quest/checkpoints/_phase1_done; then
  banner "Phase 1 already PASS — skipping"
else
  banner "Phase 1: cluster recovery"
  python scripts/data_processing/recover_cluster_files.py 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    note "Phase 1 FAILED — aborting"
    exit 1
  fi
fi

# --- Phase 2: build Run 10 ---
if [ -f /home/ubuntu/quest/checkpoints/_phase2_done ] && \
   grep -q "STATUS=PASS" /home/ubuntu/quest/checkpoints/_phase2_done; then
  banner "Phase 2 already PASS — skipping"
else
  banner "Phase 2: build Run 10"
  python scripts/data_processing/build_run10.py 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    note "Phase 2 FAILED — aborting"
    exit 1
  fi
fi

# --- Phase 3: 5-point training ---
banner "Phase 3: Stage 4 5-point training (E -> A -> D -> B -> C)"
python scripts/training/run_stage4_pipeline.py 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  note "Phase 3 FAILED — see log"
  exit 1
fi

# --- Phase 4: summary ---
banner "Phase 4: generate SUMMARY.md"
python scripts/training/build_stage4_summary.py 2>&1 | tee -a "$LOG"

banner "Pipeline complete"
exit 0
