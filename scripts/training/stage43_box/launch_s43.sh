#!/bin/bash
# Run ON the GPU box as ubuntu, right after boot. Installs code, arms BOTH stop layers, starts the job.
set -u
export PATH=/home/ubuntu/miniforge3/bin:/usr/local/bin:$PATH
# The job lays itself out by GPU count, so the ceilings follow it. 4 GPUs: R (2,000 steps on two
# GPUs) runs beside the evals and S, ~2-4 h. 1 GPU: the same 3,000 training steps run one after
# another at a quarter of the parallelism, ~5-10 h depending on whether the token-budget speedup
# lands. Each layer is independent of the one below.
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)
if [ "${NGPU:-1}" -ge 4 ]; then MAXH=4; OSMIN=300; else MAXH=10; OSMIN=660; fi
sudo shutdown -h +$OSMIN 2>/dev/null           # layer 2: OS timer, independent of every script below
cd /home/ubuntu
aws s3 cp s3://fh-pi-warren-h-eco/quest/tcrbench2/_transfer/s43box.tgz /tmp/s43box.tgz --only-show-errors && tar xzf /tmp/s43box.tgz -C /home/ubuntu
bash /home/ubuntu/install.sh > /home/ubuntu/install.log 2>&1
rm -f /home/ubuntu/s43_out/run_s43.pid
tmux new-session -d -s s43 'bash /home/ubuntu/run_s43.sh'
for i in $(seq 1 30); do [ -s /home/ubuntu/s43_out/run_s43.pid ] && break; sleep 2; done
PID=$(cat /home/ubuntu/s43_out/run_s43.pid 2>/dev/null)
[ -n "$PID" ] || { echo "LAUNCH_FAIL no pid"; sudo shutdown -h +1; exit 1; }
DST=s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/stage43_check
PUB="aws s3 sync /home/ubuntu/s43_out $DST/s43_out --only-show-errors; aws s3 cp /home/ubuntu/quest/checkpoints/s43_equiv_pointD/training.log $DST/train_equiv_training.log --only-show-errors; aws s3 cp /home/ubuntu/quest/checkpoints/s43_equiv_pointD/early_stop_history.json $DST/train_equiv_early_stop_history.json --only-show-errors; aws s3 cp /home/ubuntu/quest/checkpoints/s43_resume_pointD/training.log $DST/resume_training.log --only-show-errors"
# layer 1: watchdog on the job PID — DONE / IDLE 20 min / CEILING $MAXH h; publishes, then stops the box
tmux new-session -d -s autostop "MAX_HOURS=$MAXH IDLE_MIN=20 LOG=/home/ubuntu/s43_out/autostop.log S3_LOG=$DST/autostop.log bash /home/ubuntu/quest/scripts/training/autostop_when_done.sh $PID '$PUB'"
sleep 5
tmux ls; echo "LAUNCH_OK pid=$PID ngpu=$NGPU max_hours=$MAXH os_timer=${OSMIN}min"; head -3 /home/ubuntu/s43_out/autostop.log 2>/dev/null
