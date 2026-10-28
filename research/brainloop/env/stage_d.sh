#!/usr/bin/env bash
# Stage D1-D3 (research/brainloop/STAGE_D_PLAN.md): setup, D2 latency on the 5090, D3 the fresh-vision reflex tier
# in the simulator (3 tasks x 8 episodes, seed 0, paired with the Stage C controls). Terminal marker STAGED_EXIT=<s>.
cd /workspace/AWAM-Fast
export PYTHONPATH=/workspace/AWAM-Fast/src:/workspace/AWAM-Fast HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
R="python -u research/brainloop/run.py"
O=/workspace/out_d
A="--assets /workspace/assets"
step() { local name=$1; shift; local t0=$(date +%s); echo "== D $name start"; "$@"; local rc=$?; echo "== D $name rc=$rc took $(( $(date +%s) - t0 ))s"; return $rc; }
mkdir -p $O/d2
if [ -z "${SKIP_SETUP:-}" ]; then
  TASKS=beat_block_hammer step fetch bash research/brainloop/env/fetch_big.sh || { echo "STAGED_EXIT=fetch_failed"; exit 1; }
  MIN_HF_BPS=0 TASKS=beat_block_hammer step bootstrap bash research/brainloop/env/bootstrap.sh || { echo "STAGED_EXIT=bootstrap_failed"; exit 1; }
  step sim_setup bash research/brainloop/env/sim_setup.sh || { echo "STAGED_EXIT=sim_setup_failed"; exit 1; }
fi
if [ -z "${SKIP_D2:-}" ]; then
  step d2_reflex $R e5b $A --out $O/d2 --tasks beat_block_hammer --episodes 1 --e5b-mode reflex || { echo "STAGED_EXIT=d2_failed"; exit 1; }
  BRAINLOOP_E5_ONLY=motor step d2_motor $R e5 $A --out $O/d2 --tasks beat_block_hammer --episodes 1 || { echo "STAGED_EXIT=d2_failed"; exit 1; }
  python research/brainloop/env/d2_latency.py $O/d2 > $O/d2/latency_d.json || { echo "STAGED_EXIT=d2_parse_failed"; exit 1; }
  echo "BRAINLOOP d2 $(python -c "import json; d=json.load(open('$O/d2/latency_d.json')); print({k: round(v) for k, v in d.items() if not isinstance(v, dict)})")"
fi
C="python -u research/brainloop/env/run_c5.py --tasks click_alarmclock beat_block_hammer open_microwave --episodes ${EPISODES:-8}
   --out $O/c5 --latency-json $O/d2/latency_d.json --shift-ckpt none --motor-ckpt none --surprise-ckpt none"
step d3_reflex $C --configs reflex --modes realtime sync || { echo "STAGED_EXIT=d3_failed"; exit 1; }
step d3_reflex2 $C --configs reflex2 --modes realtime || echo "== D d3_reflex2 FAILED (continuing)"
echo "STAGED_EXIT=done"
