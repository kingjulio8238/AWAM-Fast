#!/usr/bin/env bash
# Vast.ai lifecycle for the brainloop lab. Every command prints what it does; nothing here runs implicitly.
#   env/vast.sh offers                      # list RTX 5090 offers that fit (read-only)
#   env/vast.sh up <offer_id> <cap_hours>   # rent; the box stops itself after cap_hours (server-side cap)
#   env/vast.sh push <instance_id>          # copy the repo (incl. research/) to /workspace/AWAM-Fast
#   env/vast.sh run <instance_id> "<cmd>"   # run a command in a detached tmux session, logs in /workspace/logs
#   env/vast.sh pull <instance_id>          # copy /workspace/out and /workspace/logs back to research/brainloop/results
#   env/vast.sh cost <instance_id>          # $/h, uptime, spend so far
#   env/vast.sh down <instance_id>          # destroy (stops all billing)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
IMAGE=${IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel}
DISK=${DISK:-200}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/vast_ed25519}
SSHO="-i $SSH_KEY -o StrictHostKeyChecking=no -o BatchMode=yes"
QUERY='gpu_name=RTX_5090 num_gpus=1 reliability>0.98 disk_space>=200 cpu_ram>=60 inet_down>=500 cuda_vers>=12.8 rentable=true'

ssh_args() {   # -> "-p PORT root@HOST"
  vastai show instance "$1" --raw | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('-p %s root@%s' % (d['ssh_port'], d['ssh_host']))"
}

case "${1:-}" in
  offers)   # ranked by projected total for HOURS (default 3) + GB downloaded (default 45): bandwidth is billed too
    vastai search offers "$QUERY" -o 'dph+' --raw | python3 -c "
import json,sys,os
h=float(os.environ.get('HOURS',3)); gb=float(os.environ.get('GB',45))
o=sorted(json.load(sys.stdin), key=lambda x: x['dph_total']*h + gb*x.get('inet_down_cost',0))
for x in o[:10]:
    print('%10s  \$%.3f/h  down \$%.4f/GB  total(%.1fh,%dGB) \$%.2f  cpu_ram=%3dGB  %s' % (x['id'], x['dph_total'],
          x.get('inet_down_cost',0), h, gb, x['dph_total']*h + gb*x.get('inet_down_cost',0), x['cpu_ram']/1024, x.get('geolocation')))" ;;
  up)
    OFFER=$2; CAP_H=$3
    CAP_S=$(python3 -c "print(int(float('$CAP_H')*3600))")
    # Server-side lifetime cap, so a laptop disconnect costs one capped run. At the cap the watchdog
    #  1. destroys the instance through the Vast API if VAST_CAP_KEY is set (a key passed to the box: use a
    #     restricted one), else
    #  2. SIGTERMs PID 1 so the container exits (a stopped instance bills storage only). Whether PID 1 honours
    #     SIGTERM depends on the image's init; `vast.sh cost` after the cap is how to confirm it did.
    KEYENV=""
    if [ -n "${VAST_CAP_KEY:-}" ]; then KEYENV="-e VAST_CAP_KEY=$VAST_CAP_KEY"; fi
    ONSTART="mkdir -p /workspace/logs; (sleep $CAP_S; echo \"[watchdog] \$(date -u) cap ${CAP_H}h reached\" >> /workspace/logs/watchdog.log; sync; \
      if [ -n \"\$VAST_CAP_KEY\" ]; then curl -s -X DELETE \"https://console.vast.ai/api/v0/instances/\$CONTAINER_ID/?api_key=\$VAST_CAP_KEY\" >> /workspace/logs/watchdog.log; fi; \
      kill -TERM 1) > /dev/null 2>&1 &"
    echo "renting offer $OFFER image=$IMAGE disk=${DISK}GB cap=${CAP_H}h"
    vastai create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --ssh --direct --cancel-unavail --label brainloop \
      --env "-e PYTHONUNBUFFERED=1 -e NVIDIA_DRIVER_CAPABILITIES=all $KEYENV" --onstart-cmd "$ONSTART" --raw ;;
  push)
    ID=$2; read -r -a S <<< "$(ssh_args "$ID")"
    tar -C "$REPO" --exclude .git --exclude '__pycache__' --exclude 'research/brainloop/results' -czf /tmp/awam_push.tgz .
    scp $SSHO -P "${S[1]}" /tmp/awam_push.tgz "${S[2]}:/tmp/awam_push.tgz"
    ssh $SSHO "${S[@]}" "mkdir -p /workspace/AWAM-Fast && tar -C /workspace/AWAM-Fast -xzf /tmp/awam_push.tgz && echo pushed" ;;
  run)
    ID=$2; CMD=$3; NAME=${4:-lab}; read -r -a S <<< "$(ssh_args "$ID")"
    ssh $SSHO "${S[@]}" "mkdir -p /workspace/logs && cd /workspace/AWAM-Fast && \
      { [ -f /workspace/logs/$NAME.log ] && mv /workspace/logs/$NAME.log /workspace/logs/$NAME.\$(date +%s).log || true; } && \
      tmux new-session -d -s $NAME \"export PYTHONPATH=/workspace/AWAM-Fast/src:/workspace/AWAM-Fast; { $CMD ; } 2>&1 | tee -a /workspace/logs/$NAME.log\" && echo started $NAME" ;;
  tail)
    ID=$2; NAME=${3:-lab}; read -r -a S <<< "$(ssh_args "$ID")"
    ssh $SSHO "${S[@]}" "tail -n ${4:-30} /workspace/logs/$NAME.log; cat /workspace/out/progress.json 2>/dev/null; echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" ;;
  pull)
    ID=$2; read -r -a S <<< "$(ssh_args "$ID")"
    mkdir -p "$REPO/research/brainloop/results"
    rsync -az -e "ssh $SSHO -p ${S[1]}" "${S[2]}:/workspace/out/" "$REPO/research/brainloop/results/out/" || true
    rsync -az -e "ssh $SSHO -p ${S[1]}" "${S[2]}:/workspace/logs/" "$REPO/research/brainloop/results/logs/" || true
    echo "pulled to research/brainloop/results" ;;
  cost)
    vastai show instance "$2" --raw | python3 -c "
import json,sys,time; d=json.load(sys.stdin)
up=(time.time()-float(d.get('start_date') or time.time()))/3600
bw=sum(float(d.get(a) or 0)*float(d.get(b) or 0) for a,b in (('inet_down_billed','inet_down_cost'),('inet_up_billed','inet_up_cost')))
print('status=%s  \$%.3f/h  up %.2fh  compute ~\$%.2f  bandwidth ~\$%.2f  (invoices: vastai show invoices)' % (d.get('actual_status'), d['dph_total'], up, up*d['dph_total'], bw))" ;;
  down)
    echo y | vastai destroy instance "$2" ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
