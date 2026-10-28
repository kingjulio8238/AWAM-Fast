#!/usr/bin/env bash
# Multi-stream fetch of the two single-file checkpoints (single-stream HF downloads crawled on Vast), then the normal
# fetch for everything else (it verifies these by hash and skips them).
set -euo pipefail
command -v aria2c >/dev/null || (apt-get update -qq >/dev/null && apt-get install -y -qq aria2 >/dev/null)
W=/workspace/assets/weights
rm -f $W/SDAR-1.7B-Chat/.cache/huggingface/download/*.incomplete
mkdir -p $W/SDAR-1.7B-Chat $W/Qwen3-VL-2B-Instruct
t0=$(date +%s)
[ "$(stat -c %s $W/SDAR-1.7B-Chat/model.safetensors 2>/dev/null)" = 4063515640 ] || \
aria2c -q -x 16 -s 16 -k 64M --allow-overwrite=true -d $W/SDAR-1.7B-Chat -o model.safetensors \
  https://huggingface.co/JetLM/SDAR-1.7B-Chat/resolve/main/model.safetensors &
[ "$(stat -c %s $W/Qwen3-VL-2B-Instruct/model.safetensors 2>/dev/null)" = 4255140312 ] || \
aria2c -q -x 16 -s 16 -k 64M --allow-overwrite=true -d $W/Qwen3-VL-2B-Instruct -o model.safetensors \
  https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/model.safetensors &
wait
# the 10 GB release checkpoint and the demo zips, same way
R=/workspace/assets/release; Z=/workspace/assets/robotwin_zips; mkdir -p $R $Z
[ -s $R/wam.pt ] && [ "$(stat -c %s $R/wam.pt)" = 9959529581 ] || aria2c -q -x 16 -s 16 -k 64M --allow-overwrite=true -d $R -o wam.pt \
  https://huggingface.co/miniFranka/agentic-wam-posttrain-robotwin-suit/resolve/main/wam.pt
for t in ${TASKS:-click_alarmclock beat_block_hammer stack_blocks_three handover_block open_microwave hanging_mug}; do
  mkdir -p $Z/dataset/$t
  [ -d /workspace/assets/robotwin/$t/aloha-agilex_clean_50/data ] && continue
  aria2c -q -x 8 -s 8 --allow-overwrite=true -d $Z/dataset/$t -o aloha-agilex_clean_50.zip \
    "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/dataset/$t/aloha-agilex_clean_50.zip" &
done
wait
echo "[fetch_big] done in $(( $(date +%s) - t0 ))s"; ls -la $W/SDAR-1.7B-Chat/model.safetensors $W/Qwen3-VL-2B-Instruct/model.safetensors
