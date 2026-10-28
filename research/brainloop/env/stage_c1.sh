#!/usr/bin/env bash
# Stage C1 smoke (research/brainloop/STAGE_C_PLAN.md): every C2-C5 step at toy scale, timed, so the full run can be
# projected. Logs to /workspace/logs/c1.log via vast.sh run. Terminal marker: STAGEC1_EXIT=<status>.
cd /workspace/AWAM-Fast
export PYTHONPATH=/workspace/AWAM-Fast/src:/workspace/AWAM-Fast HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
R="python -u research/brainloop/run.py"
O=/workspace/out_c1
A="--assets /workspace/assets --out $O"
T="click_alarmclock beat_block_hammer"
step() { local name=$1; shift; local t0=$(date +%s); echo "== C1 $name start"; "$@"; local rc=$?; echo "== C1 $name rc=$rc took $(( $(date +%s) - t0 ))s"; return $rc; }
[ -n "${SKIP_TO_T1:-}${SKIP_TO_TEACHER:-}${SKIP_TO_SIM:-}" ] || TASKS="$T" step fetch bash research/brainloop/env/fetch_big.sh || { echo "STAGEC1_EXIT=fetch_failed"; exit 1; }
[ -n "${SKIP_TO_T1:-}${SKIP_TO_TEACHER:-}${SKIP_TO_SIM:-}" ] || MIN_HF_BPS=0 TASKS="$T" step bootstrap bash research/brainloop/env/bootstrap.sh || { echo "STAGEC1_EXIT=bootstrap_failed"; exit 1; }
[ -n "${SKIP_TO_TEACHER:-}" ] && rm -rf $O/d1 $O/d2; [ -n "${SKIP_TO_T1:-}${SKIP_TO_SIM:-}" ] || step teacher $R teacher $A --tasks $T --d1-demos 3 --d2-from 1 --d2-every 1 || { echo "STAGEC1_EXIT=teacher_failed"; exit 1; }
[ -n "${SKIP_TO_SIM:-}" ] || step train_motor $R train_motor $A --steps 200 --batch 8 --heldout-from 1 || { echo "STAGEC1_EXIT=t1_failed"; exit 1; }
[ -n "${SKIP_TO_SIM:-}" ] || step train_t23 $R train_t23 $A --t2-epochs 5 --t23-eval-from 2 || echo "== C1 train_t23 FAILED (continuing)"
[ -n "${SKIP_TO_SIM:-}" ] || step eval_c $R eval_c $A --heldout-from 1 --t23-eval-from 2 || echo "== C1 eval_c FAILED (continuing)"
step sim_setup bash research/brainloop/env/sim_setup.sh || { echo "STAGEC1_EXIT=sim_setup_failed"; exit 1; }
step c5_smoke python -u research/brainloop/env/run_c5.py --tasks click_alarmclock --episodes 2 \
  --configs released warm1 --modes sync realtime --out $O/c5 \
  --motor-ckpt $O/t1/motor_t1.pt --shift-ckpt $O/t2/shift_t2.pt --surprise-ckpt $O/t3/surprise_t3.pt \
  || { echo "STAGEC1_EXIT=c5_failed"; exit 1; }
echo "STAGEC1_EXIT=done"
