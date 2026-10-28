#!/usr/bin/env bash
# Stages C2-C4 on the box that ran C1 (weights already present): all-task demos, teacher data, T1/T2/T3, offline eval.
# Terminal marker: STAGEC24_EXIT=<status>.
cd /workspace/AWAM-Fast
export PYTHONPATH=/workspace/AWAM-Fast/src:/workspace/AWAM-Fast HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
R="python -u research/brainloop/run.py"
O=/workspace/out_c
A="--assets /workspace/assets --out $O"
ALL=$(ls research/../configs/robotwin/tasks 2>/dev/null || ls configs/robotwin/tasks)
step() { local name=$1; shift; local t0=$(date +%s); echo "== C2 $name start"; "$@"; local rc=$?; echo "== C2 $name rc=$rc took $(( $(date +%s) - t0 ))s"; return $rc; }
TASKS="$ALL" step fetch_demos bash research/brainloop/env/fetch_big.sh || { echo "STAGEC24_EXIT=fetch_failed"; exit 1; }
step extract_demos $R fetch $A --tasks $ALL || { echo "STAGEC24_EXIT=extract_failed"; exit 1; }
# T1 (stale-cache motor tier) is off the critical path after C1: three variants all made it worse, and warm
# replans at 66 ms already reach the gate. D1 now only feeds T2/T3, so it runs on the D2 episodes only.
step teacher $R teacher $A --d1-from 16 --d1-demos 20 --d1-every 2 --d2-from 16 --d2-every 2 || { echo "STAGEC24_EXIT=teacher_failed"; exit 1; }
step train_t23 $R train_t23 $A --t2-epochs 30 || echo "== C2 train_t23 FAILED (continuing)"
step eval_c $R eval_c $A --heldout-from 16 || echo "== C2 eval_c FAILED (continuing)"
echo "STAGEC24_EXIT=done"
