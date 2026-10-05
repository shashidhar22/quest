#!/bin/bash
# Laptop-side: retry GPU capacity every 15 min; on success launch the Stage 4.2/4.3 job with its
# watchdog armed, verify, and exit. Layer 3 of the auto-stop: if the box never becomes reachable or
# the launch can't be verified, THIS script stops the instance before exiting.
export AWS_PROFILE=DataUserAccess-953915750371 AWS_REGION=us-west-2
B=i-0030c1a9c845b3714; W=/Users/yxu2/Documents/ACS/AWSconnection/warren-gpu.sh
TYPES="g6.12xlarge g5.12xlarge g6.24xlarge"   # run_s43 lays out work across 4 GPUs
B64=$(base64 < /private/tmp/claude-501/-Users-yxu2-Documents-ACS/a8ebc35a-664e-421a-9b96-b4a7ff715def/scratchpad/s43box/launch_s43.sh | tr -d '\n')
log(){ echo "[$(date -u '+%F %T')] $*"; }
for attempt in $(seq 1 96); do                # 96 x 15 min = 24 h
  aws sts get-caller-identity >/dev/null 2>&1 || { log "SSO_EXPIRED — run: aws sso login --profile $AWS_PROFILE"; exit 3; }
  st=$(aws ec2 describe-instances --instance-ids $B --query 'Reservations[].Instances[].State.Name' --output text)
  [ "$st" = stopping ] && { sleep 60; continue; }
  [ "$st" = stopped ] || { log "box is '$st', not stopped — someone else is using it; aborting without touching it"; exit 4; }
  started=""
  for T in $TYPES; do
    aws ec2 modify-instance-attribute --instance-id $B --instance-type "{\"Value\":\"$T\"}" 2>/dev/null
    if aws ec2 start-instances --instance-ids $B >/dev/null 2>/tmp/gpuretry.err; then started=$T; break; fi
  done
  if [ -z "$started" ]; then log "attempt $attempt: no capacity for $TYPES"; sleep 900; continue; fi
  log "attempt $attempt: STARTED as $started"
  aws ec2 wait instance-running --instance-ids $B
  ok=""
  for i in $(seq 1 60); do [ "$(aws ssm describe-instance-information --query "InstanceInformationList[?InstanceId=='$B'].PingStatus" --output text)" = Online ] && { ok=1; break; }; sleep 15; done
  [ -n "$ok" ] || { log "SSM never came online in 15 min — STOPPING the box"; aws ec2 stop-instances --instance-ids $B >/dev/null; exit 5; }
  sleep 45
  out=$(WARREN_BOX=$B $W exec "echo $B64 | base64 -d > /tmp/launch_s43.sh; sudo -u ubuntu bash /tmp/launch_s43.sh; sleep 2" 240 2>&1)
  echo "$out" | grep -E 'LAUNCH_|s43:|autostop:|armed' | sed 's/^/    /'
  if echo "$out" | grep -q 'LAUNCH_OK' && echo "$out" | grep -q 'autostop:' && echo "$out" | grep -q 's43:'; then
    log "LAUNCHED on $started; watchdog armed (MAX_HOURS=3) + OS timer 240 min. Results -> s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/stage43_check/"
    exit 0
  fi
  log "launch NOT verified — STOPPING the box"; aws ec2 stop-instances --instance-ids $B >/dev/null; exit 6
done
log "gave up after 24 h of no capacity"; exit 7
