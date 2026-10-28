#!/usr/bin/env bash
# On a fresh GPU box running pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel: install the lab, prove the environment,
# fetch assets. Idempotent. Logs to /workspace/logs/bootstrap.log. Terminal marker: "BRAINLOOP bootstrap DONE".
set -euo pipefail
REPO=${REPO:-/workspace/AWAM-Fast}
ASSETS=${ASSETS:-/workspace/assets}
TASKS=${TASKS:-"click_alarmclock beat_block_hammer stack_blocks_three handover_block open_microwave hanging_mug"}
mkdir -p /workspace/logs
exec > >(tee -a /workspace/logs/bootstrap.log) 2>&1
echo "== bootstrap $(date -u +%FT%TZ) on $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# fail fast on a host with a slow path to Hugging Face (Stage B attempt 1: Taiwan host managed ~5 MB/s, i.e. hours
# for the 40 GB of weights); measured on one weight file for 12 s
HF_BPS=$(curl -sL --max-time 12 -o /dev/null -w '%{speed_download}' https://huggingface.co/JetLM/SDAR-1.7B-Chat/resolve/main/model.safetensors 2>/dev/null; true)
HF_BPS=${HF_BPS:-0}   # curl exits 28 at --max-time but still prints the measured speed
echo "HF throughput: $(python -c "print(round(float('$HF_BPS')/1e6,1))") MB/s"
python -c "import sys; sys.exit(0 if float('$HF_BPS') >= ${MIN_HF_BPS:-30e6} else 1)" || { echo "BRAINLOOP bootstrap FAILED slow_hf"; exit 1; }
free -g | head -2; df -h /workspace | tail -1
export DEBIAN_FRONTEND=noninteractive
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux rsync ffmpeg libgl1 libglib2.0-0 >/dev/null)
cd "$REPO"
TV=$(python -c "import torch; print(torch.__version__)")
pip install -q -r research/brainloop/env/requirements-lab.txt
pip install -q --no-deps -e .
NV=$(python -c "import torch; print(torch.__version__)")
[ "$TV" = "$NV" ] || { echo "pip changed torch $TV -> $NV"; exit 1; }
# plain HTTP downloads: the xet / hf_transfer backends stalled at 0 MB/s on two hosts (Stage B) while curl on the same
# files ran at 50 MB/s
export PYTHONPATH="$REPO/src:$REPO" HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
python research/brainloop/env/check_env.py
python -m pytest -q -p no:cacheprovider research/brainloop/tests/test_cpu_tiny.py
python -u research/brainloop/run.py fetch --assets "$ASSETS" --tasks $TASKS
df -h /workspace | tail -1
echo "BRAINLOOP bootstrap DONE"
