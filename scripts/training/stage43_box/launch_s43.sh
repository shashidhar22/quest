#!/bin/bash
# Run ON the GPU box as ubuntu, right after boot. Installs code, arms BOTH stop layers, starts the job.
set -u
export PATH=/home/ubuntu/miniforge3/bin:/usr/local/bin:$PATH
# One box visit runs Stage 4.3 and then Stage 4.2 wave 1 (run_all.sh), both laid out by GPU count,
# so the ceilings follow it. 4 GPUs: 4.3 ~2-4 h, then wave 1 with 3 cells in parallel ~2-7 h.
# 1 GPU: 4.3 ~5-10 h, then wave 1 sequential ~4-13 h. Each layer is independent of the one below.
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)
if [ "${NGPU:-1}" -ge 4 ]; then MAXH=12; OSMIN=780; else MAXH=22; OSMIN=1380; fi
sudo shutdown -h +$OSMIN 2>/dev/null           # layer 2: OS timer, independent of every script below
cd /home/ubuntu
aws s3 cp s3://fh-pi-warren-h-eco/quest/tcrbench2/_transfer/s43box.tgz /tmp/s43box.tgz --only-show-errors && tar xzf /tmp/s43box.tgz -C /home/ubuntu
bash /home/ubuntu/install.sh > /home/ubuntu/install.log 2>&1
rm -f /home/ubuntu/s43_out/run_all.pid /home/ubuntu/s43_out/run_s43.pid
tmux new-session -d -s s43 'bash /home/ubuntu/run_all.sh'
for i in $(seq 1 30); do [ -s /home/ubuntu/s43_out/run_all.pid ] && break; sleep 2; done
PID=$(cat /home/ubuntu/s43_out/run_all.pid 2>/dev/null)
[ -n "$PID" ] || { echo "LAUNCH_FAIL no pid"; sudo shutdown -h +1; exit 1; }
DST=s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/stage43_check
DST42=s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/stage42_wave1
PUB="aws s3 sync /home/ubuntu/s43_out $DST/s43_out --only-show-errors; aws s3 sync /home/ubuntu/s42_out $DST42 --only-show-errors; for d in s43_equiv_pointD s43_resume_pointD s43_equiv_v2_pointD s43_resume_v2_pointD; do [ -d /home/ubuntu/quest/checkpoints/\$d ] || continue; aws s3 cp /home/ubuntu/quest/checkpoints/\$d/training.log $DST/\${d}_training.log --only-show-errors; aws s3 cp /home/ubuntu/quest/checkpoints/\$d/early_stop_history.json $DST/\${d}_early_stop_history.json --only-show-errors; done"
# layer 1: watchdog on the job PID — DONE / IDLE 45 min / CEILING $MAXH h; publishes, then stops the
# box. IDLE is 45 not 20 because the judge's process name does not match the watchdog's training
# pattern, so during wave 1 only GPU utilisation keeps the idle guard from firing between cells.
tmux new-session -d -s autostop "MAX_HOURS=$MAXH IDLE_MIN=45 LOG=/home/ubuntu/s43_out/autostop.log S3_LOG=$DST/autostop.log bash /home/ubuntu/quest/scripts/training/autostop_when_done.sh $PID '$PUB'"
sleep 5
tmux ls; echo "LAUNCH_OK pid=$PID ngpu=$NGPU max_hours=$MAXH os_timer=${OSMIN}min"; head -3 /home/ubuntu/s43_out/autostop.log 2>/dev/null
