#!/usr/bin/env bash
# Watch only NEW lines of a remote log (markers / errors), plus progress.json every 5 anchors. One event per line.
#   tail_watch.sh <host> <port> <log> <exit_marker>
set -u
HOST=$1; PORT=$2; LOG=$3; EXIT=$4
KEY=${SSH_KEY:-$HOME/.ssh/vast_ed25519}
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o LogLevel=ERROR -o ConnectTimeout=20 -p $PORT root@$HOST"
PAT='BRAINLOOP|STAGE[A-Z0-9]*_EXIT|Traceback|Error|Killed|OOM|\[sweep\]|\[chain\]|\[e4\]|== C[0-9]|\[c5\]|\[teacher\]'"${EXTRA_PAT:+|$EXTRA_PAT}"
start=${START:-$($SSH "wc -l < $LOG" 2>/dev/null | tr -dc 0-9)}; start=${start:-0}
lastp=""; fails=0
while true; do
  n=$($SSH "wc -l < $LOG" 2>/dev/null | tr -dc 0-9)
  if [ -z "$n" ]; then fails=$((fails + 1)); [ "$fails" -eq 3 ] && echo "$(date -u +%H:%M) SSH FAILING x3"; sleep 45; continue; fi
  fails=0
  if [ "$n" -gt "$start" ]; then
    new=$($SSH "sed -n '$((start + 1)),${n}p' $LOG" 2>/dev/null | grep -E "$PAT" | grep -v -e UserWarning -e DEPRECATION | cut -c1-300)
    [ -n "$new" ] && printf '%s\n' "$new" | sed "s/^/$(date -u +%H:%M) /"
    start=$n
    printf '%s' "$new" | grep -q "$EXIT" && exit 0
  fi
  p=$($SSH "cat /workspace/out/progress.json" 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); print(d['job'], 'ep', d['episodes_done'], '/', d['episodes_total'], d['ep_task'], 'anchor', d['anchor']//5*5, '/', d['anchors'])
except Exception: pass")
  if [ -n "$p" ] && [ "$p" != "$lastp" ]; then echo "$(date -u +%H:%M) PROGRESS $p"; lastp=$p; fi
  sleep 45
done
