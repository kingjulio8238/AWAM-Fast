#!/usr/bin/env bash
# Dedicated watchers for a Vast run; every stdout line is an event (made for the Monitor tool).
#   vast_watch.sh chain <instance> <session> <exit_marker>   stage markers, errors, sweep progress; exits on the marker
#   vast_watch.sh cost  <instance> <session> <cap_usd>       spend thresholds, GPU idle while the session runs, status
# Silence is never success: an ssh failure streak of 3 is itself an event.
set -u
MODE=$1; ID=$2; SESSION=$3; ARG=$4
KEY=${SSH_KEY:-$HOME/.ssh/vast_ed25519}
read -r HOST PORT < <(vastai show instance "$ID" --raw | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['ssh_host'], d['ssh_port'])")
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR -p $PORT root@$HOST"
fails=0

remote() { $SSH "$1" 2>&1 | grep -v -e "Welcome to vast.ai" -e "Have fun" ; return "${PIPESTATUS[0]}"; }

if [ "$MODE" = chain ]; then
  seen=0; last_prog=""
  while true; do
    if out=$(remote "grep -nE 'BRAINLOOP|Traceback|Error|error|Killed|OOM|CUDA out of memory|\[policy\]|\[text\]|\[sweep\]|\[e4\]|$ARG' /workspace/logs/$SESSION.log | grep -v -e Ignoring -e DEPRECATION -e 'warn' ; echo '@@'; cat /workspace/out/progress.json 2>/dev/null; echo; tmux has-session -t =$SESSION 2>/dev/null && echo ALIVE || echo ENDED"); then
      fails=0
      lines=$(printf '%s\n' "$out" | sed -n '1,/^@@$/p' | sed '$d')
      n=$(printf '%s\n' "$lines" | grep -c . || true)
      if [ "$n" -gt "$seen" ]; then printf '%s\n' "$lines" | tail -n $((n - seen)) | cut -c1-300 | sed "s/^/$(date -u +%H:%M) /"; seen=$n; fi
      prog=$(printf '%s\n' "$out" | sed -n '/^@@$/,$p' | python3 -c "
import json,sys
t=sys.stdin.read().split('\n')
try:
    d=json.loads(t[1]); a=d.get('anchor',0); print(f\"{d.get('job')} ep {d.get('episodes_done')}/{d.get('episodes_total')} {d.get('ep_task')} anchor {a//5*5}/{d.get('anchors')} decode {d.get('last_decode_s')}s\")
except Exception: print('')" 2>/dev/null)
      if [ -n "$prog" ] && [ "$prog" != "$last_prog" ]; then echo "$(date -u +%H:%M) PROGRESS $prog"; last_prog=$prog; fi
      if printf '%s\n' "$out" | grep -q "$ARG"; then echo "$(date -u +%H:%M) CHAIN FINISHED"; exit 0; fi
      if printf '%s\n' "$out" | tail -1 | grep -q ENDED; then echo "$(date -u +%H:%M) SESSION ENDED without $ARG"; exit 1; fi
    else
      fails=$((fails + 1)); [ "$fails" -eq 3 ] && echo "$(date -u +%H:%M) SSH FAILING x3: $(printf '%s' "$out" | tail -1 | cut -c1-200)"
    fi
    sleep 30
  done
fi

if [ "$MODE" = cost ]; then
  next=0.5; idle=0
  while true; do
    # spend = account credit at launch (START_CREDIT) minus credit now: authoritative, includes GPU, storage and
    # bandwidth (the instance's *_billed fields are not in GB, and summing rates missed bandwidth in Stage A)
    st=$( { vastai show instance "$ID" --raw; echo '@@'; vastai show user --raw; } 2>&1 | python3 -c "
import json,sys,time,os
raw=sys.stdin.read().split('@@')
try:
    d=json.loads(raw[0]); u=json.loads(raw[1]); up=(time.time()-float(d.get('start_date') or time.time()))/3600
    start=float(os.environ.get('START_CREDIT','nan'))
    print(d.get('actual_status'), round(start-float(u['credit']),3), round(up,2))
except Exception as e: print('GONE', 0, 0)")
    read -r status spent up <<< "$st"
    if [ "$status" = GONE ]; then echo "$(date -u +%H:%M) instance gone (destroyed)"; exit 0; fi
    [ "$status" != running ] && echo "$(date -u +%H:%M) STATUS $status (up ${up}h, \$$spent)"
    if python3 -c "import sys; sys.exit(0 if float('$spent') >= float('$next') else 1)"; then
      echo "$(date -u +%H:%M) SPEND \$$spent after ${up}h (cap \$$ARG)"; next=$(python3 -c "print(round(float('$next')+0.25,2))")
    fi
    python3 -c "import sys; sys.exit(0 if float('$spent') >= 0.9*float('$ARG') else 1)" && echo "$(date -u +%H:%M) NEAR CAP \$$spent of \$$ARG"
    if u=$(remote "tmux has-session -t =$SESSION 2>/dev/null && nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits || echo NOSESSION"); then
      fails=0; u=$(printf '%s' "$u" | tail -1 | tr -d ' ')
      if [ "$u" = NOSESSION ]; then idle=$((idle + 1)); [ "$idle" -eq 3 ] && echo "$(date -u +%H:%M) BOX IDLE: no '$SESSION' session for 3 min, billing \$$spent";
      elif [ "$u" -lt 5 ] 2>/dev/null; then idle=$((idle + 1)); [ "$idle" -eq 3 ] && echo "$(date -u +%H:%M) GPU IDLE ${u}% for 3 min while $SESSION runs"
      else idle=0; fi
    else
      fails=$((fails + 1)); [ "$fails" -eq 3 ] && echo "$(date -u +%H:%M) SSH FAILING x3 (cost watch)"
    fi
    sleep 60
  done
fi
