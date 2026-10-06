#!/bin/bash
# Run ON the GPU box as ubuntu, right after boot. Installs code, arms BOTH stop layers, starts the job.
set -u
export PATH=/home/ubuntu/miniforge3/bin:/usr/local/bin:$PATH
sudo shutdown -h +300 2>/dev/null              # layer 2: OS timer, independent of every script below
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
# layer 1: watchdog on the job PID — DONE / IDLE 20 min / CEILING 4 h (equivalence ~1-2 h on GPUs 2,3
# with the resume check in parallel on GPUs 0,1); publishes, then stops the box
tmux new-session -d -s autostop "MAX_HOURS=4 IDLE_MIN=20 LOG=/home/ubuntu/s43_out/autostop.log S3_LOG=$DST/autostop.log bash /home/ubuntu/quest/scripts/training/autostop_when_done.sh $PID '$PUB'"
sleep 5
tmux ls; echo "LAUNCH_OK pid=$PID"; head -3 /home/ubuntu/s43_out/autostop.log 2>/dev/null
