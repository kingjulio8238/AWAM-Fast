#!/usr/bin/env bash
# Stage B chain (see research/brainloop/STAGE_B.md). Waits for bootstrap, then runs every stage in value order.
cd /workspace/AWAM-Fast
export PYTHONPATH=/workspace/AWAM-Fast/src:/workspace/AWAM-Fast
SPEC="click_alarmclock:1,beat_block_hammer:1,stack_blocks_three:1,handover_block:1,open_microwave:1,hanging_mug:1"
A="--assets /workspace/assets --out /workspace/out --spec $SPEC"
R="python -u research/brainloop/run.py"
while tmux has-session -t =boot 2>/dev/null; do sleep 10; done
grep -q "BRAINLOOP bootstrap DONE" /workspace/logs/boot.log || { echo "STAGEB_EXIT=bootstrap_failed"; exit 1; }
$R proof $A && $R sweep $A && $R chain $A
$R e5b $A --e5b-mode core
$R e5b $A --e5b-mode fastjoint
$R e4 $A --e4-anchors 1
$R analyze --out /workspace/out
echo "STAGEB_EXIT=done"
