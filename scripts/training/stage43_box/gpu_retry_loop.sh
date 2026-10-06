#!/bin/bash
# Laptop-side: retry GPU capacity every 15 min; on success launch the Stage 4.2/4.3 job with its
# watchdog armed, verify, and exit. Layer 3 of the auto-stop: if the box never becomes reachable or
# the launch can't be verified, THIS script stops the instance before exiting.
export AWS_PROFILE=DataUserAccess-953915750371 AWS_REGION=us-west-2
B=i-0030c1a9c845b3714; W=/Users/yxu2/Documents/ACS/AWSconnection/warren-gpu.sh
# The instance types (gpu_types.txt) and the settings (loop.conf: POLL_SECONDS, MAX_HOURS) are
# RE-READ on every attempt, so both can be edited while the loop runs. $TYPES overrides the file
# for a one-off run.
HERE=$(cd "$(dirname "$0")" && pwd)
TYPES_FILE=${TYPES_FILE:-"$HERE/gpu_types.txt"}; CONF=${CONF:-"$HERE/loop.conf"}
read_types(){
  if [ -n "${TYPES:-}" ]; then echo "$TYPES"; return; fi
  sed 's/#.*//' "$TYPES_FILE" 2>/dev/null | tr -s '[:space:]' ' '
}
conf(){   # conf <KEY> <default>: last uncommented KEY=value in loop.conf, else the default
  local v; v=$(sed 's/#.*//' "$CONF" 2>/dev/null | grep -E "^[[:space:]]*$1[[:space:]]*=" | tail -1 | cut -d= -f2 | tr -d '[:space:]')
  echo "${v:-$2}"
}
B64=$(base64 < "$HERE/launch_s43.sh" | tr -d '\n')   # the repo copy, read once at start
log(){ echo "[$(date -u '+%F %T')] $*"; }
[ -n "$(read_types)" ] || { log "no instance types in $TYPES_FILE — nothing to try"; exit 2; }
t_loop=$(date +%s); attempt=0
log "loop start: poll=$(conf POLL_SECONDS 900)s max_hours=$(conf MAX_HOURS 24) types=$(read_types)"
while [ $(( $(date +%s) - t_loop )) -lt $(( $(conf MAX_HOURS 24) * 3600 )) ]; do
  attempt=$((attempt + 1))
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
    sleep "$(conf POLL_SECONDS 900)"; continue
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
log "gave up after $(conf MAX_HOURS 24) h without a launch ($attempt attempts)"; exit 7
