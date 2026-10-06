#!/bin/bash
# Laptop-side: retry GPU capacity every 15 min; on success launch the Stage 4.2/4.3 job with its
# watchdog armed, verify, and exit. Layer 3 of the auto-stop: if the box never becomes reachable or
# the launch can't be verified, THIS script stops the instance before exiting.
export AWS_PROFILE=DataUserAccess-953915750371 AWS_REGION=us-west-2
B=i-0030c1a9c845b3714; W=/Users/yxu2/Documents/ACS/AWSconnection/warren-gpu.sh
# The instance types live in gpu_types.txt (ordered, commented) and are RE-READ on every attempt,
# so that list can be edited while the loop runs. $TYPES overrides it for a one-off run.
TYPES_FILE=${TYPES_FILE:-"$(cd "$(dirname "$0")" && pwd)/gpu_types.txt"}
read_types(){
  if [ -n "${TYPES:-}" ]; then echo "$TYPES"; return; fi
  sed 's/#.*//' "$TYPES_FILE" 2>/dev/null | tr -s '[:space:]' ' '
}
B64=$(base64 < "$(dirname "$0")/launch_s43.sh" | tr -d '\n')   # the repo copy, read once at start
log(){ echo "[$(date -u '+%F %T')] $*"; }
[ -n "$(read_types)" ] || { log "no instance types in $TYPES_FILE — nothing to try"; exit 2; }
for attempt in $(seq 1 96); do                # 96 x 15 min = 24 h
  aws sts get-caller-identity >/dev/null 2>&1 || { log "SSO_EXPIRED — run: aws sso login --profile $AWS_PROFILE"; exit 3; }
  st=$(aws ec2 describe-instances --instance-ids $B --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
  if [ -z "$st" ]; then                      # API/auth error (seen after laptop wake): retry, never abort on it
    errs=$((${errs:-0} + 1)); log "describe-instances failed (API/auth error $errs/5) — retrying in 60 s"
    [ "$errs" -ge 5 ] && { log "SSO_EXPIRED or AWS unreachable — run: aws sso login --profile $AWS_PROFILE"; exit 3; }
    sleep 60; continue
  fi
  errs=0
  [ "$st" = stopping ] && { sleep 60; continue; }
  [ "$st" = stopped ] || { log "box is '$st', not stopped — someone else is using it; aborting without touching it"; exit 4; }
  started=""; types=$(read_types); why=""
  for T in $types; do
    if ! aws ec2 modify-instance-attribute --instance-id $B --instance-type "{\"Value\":\"$T\"}" 2>/tmp/gpuretry.err; then
      why="$why $T:modify-$(grep -o '([A-Za-z.]*)' /tmp/gpuretry.err | head -1 | tr -d '()')"; continue
    fi
    if aws ec2 start-instances --instance-ids $B >/dev/null 2>/tmp/gpuretry.err; then started=$T; break; fi
    why="$why $T:$(grep -o '([A-Za-z.]*)' /tmp/gpuretry.err | head -1 | tr -d '()')"   # the AWS error class, per type
  done
  if [ -z "$started" ]; then
    # Say "no capacity" only when every refusal really was InsufficientInstanceCapacity: strip
    # those tokens and see whether any type:error is left.
    if echo "$why" | sed 's/[^ ]*:InsufficientInstanceCapacity//g' | grep -q ':'; then
      log "attempt $attempt: refused —$why"
    else
      log "attempt $attempt: no capacity for $types"
    fi
    sleep 900; continue
  fi
  log "attempt $attempt: STARTED as $started"
  aws ec2 wait instance-running --instance-ids $B
  ok=""
  for i in $(seq 1 60); do [ "$(aws ssm describe-instance-information --query "InstanceInformationList[?InstanceId=='$B'].PingStatus" --output text)" = Online ] && { ok=1; break; }; sleep 15; done
  [ -n "$ok" ] || { log "SSM never came online in 15 min — STOPPING the box"; aws ec2 stop-instances --instance-ids $B >/dev/null; exit 5; }
  sleep 45
  out=$(WARREN_BOX=$B $W exec "echo $B64 | base64 -d > /tmp/launch_s43.sh; sudo -u ubuntu bash /tmp/launch_s43.sh; sleep 2" 240 2>&1)
  echo "$out" | grep -E 'LAUNCH_|s43:|autostop:|armed' | sed 's/^/    /'
  if echo "$out" | grep -q 'LAUNCH_OK' && echo "$out" | grep -q 'autostop:' && echo "$out" | grep -q 's43:'; then
    log "LAUNCHED on $started; ceilings are in the LAUNCH_OK line above (the box sets them from its GPU count). Results -> s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/stage43_check/"
    exit 0
  fi
  log "launch NOT verified — STOPPING the box"; aws ec2 stop-instances --instance-ids $B >/dev/null; exit 6
done
log "gave up after 24 h of no capacity"; exit 7
