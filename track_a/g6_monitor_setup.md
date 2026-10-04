# g6 monitor — deployment method

**Method**: cron (preferred over `nohup setsid` loop because cron is already running, gives us automatic restart-on-reboot, and persists across user logouts now that `loginctl enable-linger ubuntu` is in place).

**Crontab entry** (every 5 min):

```
*/5 * * * * /home/ubuntu/miniforge3/bin/python3 /home/ubuntu/quest/scripts/monitor_g6.py >> /home/ubuntu/quest/track_a/monitor/cron.log 2>&1
```

Installed via:

```bash
( crontab -l 2>/dev/null | grep -v "monitor_g6.py" ; \
  echo '*/5 * * * * /home/ubuntu/miniforge3/bin/python3 /home/ubuntu/quest/scripts/monitor_g6.py >> /home/ubuntu/quest/track_a/monitor/cron.log 2>&1' ) | crontab -
```

`cron.service` was already active (since boot 2026-05-03 22:56 UTC); no extra setup needed.

**To switch to nohup loop instead** (in case cron is removed):

```bash
nohup setsid bash -c 'while true; do /home/ubuntu/miniforge3/bin/python3 /home/ubuntu/quest/scripts/monitor_g6.py; sleep 300; done' \
  </dev/null >> /home/ubuntu/quest/track_a/monitor/cron.log 2>&1 &
disown
```
