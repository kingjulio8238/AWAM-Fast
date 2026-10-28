#!/usr/bin/env bash
# Stage D4 (A, STAGE_D_PLAN.md): confirmation on 5 tasks x 20 paired episodes. The 3 Stage C/D3 tasks already have
# seed-0 episodes 0-7, so they add seed 1 x 12; the 2 new tasks run seed 0 x 20. WIN = the reflex variant from D3.
# Terminal marker STAGED4_EXIT=<status>.
cd /workspace/AWAM-Fast
export PYTHONPATH=/workspace/AWAM-Fast/src:/workspace/AWAM-Fast
O=/workspace/out_d
WIN=${WIN:-reflex}
C="python -u research/brainloop/env/run_c5.py --out $O/c5a --latency-json $O/d2/latency_d.json --shift-ckpt none
   --motor-ckpt none --surprise-ckpt none"
step() { local name=$1; shift; local t0=$(date +%s); echo "== D4 $name start"; "$@"; local rc=$?; echo "== D4 $name rc=$rc took $(( $(date +%s) - t0 ))s"; return $rc; }
for cfgmode in "$WIN realtime" "released sync" "cold8async realtime"; do
  set -- $cfgmode
  step "$1_$2_old3" $C --tasks click_alarmclock beat_block_hammer open_microwave --seed 1 --episodes 12 --configs $1 --modes $2 \
    || { echo "STAGED4_EXIT=failed"; exit 1; }
  step "$1_$2_new2" $C --tasks stack_blocks_two press_stapler --seed 0 --episodes 20 --configs $1 --modes $2 \
    || { echo "STAGED4_EXIT=failed"; exit 1; }
done
echo "STAGED4_EXIT=done"
