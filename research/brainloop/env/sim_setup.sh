#!/usr/bin/env bash
# RoboTwin 2.0 simulator on the lab box (after bootstrap.sh), inside the lab's own Python (torch 2.7.1 + cu128):
# RoboTwin at the commit codegen/sim pins, its deps minus the torch / huggingface_hub pins, the two upstream patches,
# cuRobo built for this GPU, the assets, an offline HF cache for the instruction processor, and the machine config
# files. Idempotent. Terminal marker: "BRAINLOOP sim_setup DONE".
set -euo pipefail
REPO=${REPO:-/workspace/AWAM-Fast}
RT=${RT:-/workspace/RoboTwin}
ASSETS=${ASSETS:-/workspace/assets}
COMMIT=bf44be51cf5717a5595ce59447f2cf5263d2aa95
export DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace/logs
exec > >(tee -a /workspace/logs/sim_setup.log) 2>&1
echo "== sim_setup $(date -u +%FT%TZ)"
apt-get update -qq >/dev/null
apt-get install -y -qq git unzip ninja-build libvulkan1 vulkan-tools libegl1 libgl1 libglib2.0-0 aria2 ffmpeg >/dev/null
ls /etc/vulkan/icd.d/ /usr/share/vulkan/icd.d/ 2>/dev/null || true
vulkaninfo --summary 2>&1 | grep -iE "deviceName|driverName|GPU" | head -4 || echo "vulkaninfo failed"

[ -d $RT/.git ] || git clone -q https://github.com/RoboTwin-Platform/RoboTwin.git $RT
git -C $RT checkout -q $COMMIT
grep -v -E '^(torch|torchvision|huggingface_hub)([=<> ]|$)' $RT/script/requirements.txt > /tmp/rt_req.txt
pip install -q -r /tmp/rt_req.txt toppra "warp-lang==1.7.2"   # cuRobo 0.7.8 needs wp.torch (gone in warp 1.18)
python -c "import torch; assert torch.__version__.startswith('2.7.1'), torch.__version__"

# the two upstream patches from script/_install.sh
SAPIEN=$(python -c "import sapien, os; print(os.path.dirname(sapien.__file__))")
MPLIB=$(python -c "import mplib, os; print(os.path.dirname(mplib.__file__))")
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $SAPIEN/wrapper/urdf_loader.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $MPLIB/planner.py

# cuRobo (the sim's motion planner), built for this card
if ! python -c "import curobo" 2>/dev/null; then
  [ -d $RT/envs/curobo/.git ] || git clone -q https://github.com/NVlabs/curobo.git $RT/envs/curobo
  git -C $RT/envs/curobo checkout -q v0.7.8 || true
  CAP=$(python -c "import torch; c=torch.cuda.get_device_capability(); print(f'{c[0]}.{c[1]}')")
  ( cd $RT/envs/curobo && TORCH_CUDA_ARCH_LIST="$CAP" MAX_JOBS=16 pip install -q -e . --no-build-isolation )
fi
python -c "import curobo, sapien, mplib, toppra, open3d; print('sim imports OK')"

# assets (objects, embodiments, background textures), multi-stream
if [ ! -f $RT/assets/.brainloop_done ]; then
  pids=""
  for f in objects embodiments background_texture; do
    [ -s $RT/assets/$f.zip ] && [ ! -f $RT/assets/$f.zip.aria2 ] && continue
    aria2c -q -x 16 -s 16 -k 64M --allow-overwrite=true -d $RT/assets -o $f.zip \
      "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/$f.zip" &
    pids="$pids $!"
  done
  # wait on the downloads only: a bare `wait` also waits on the `exec > >(tee ...)` logger and never returns
  [ -z "$pids" ] || wait $pids
  ( cd $RT/assets && for f in objects embodiments background_texture; do unzip -oq $f.zip && rm -f $f.zip; done )
  ( cd $RT && python script/update_embodiment_config_path.py )
  touch $RT/assets/.brainloop_done
fi

# the eval runs HF-offline: cache the Cosmos-Reason1 processor files once
export HF_HOME=/workspace/hf_home
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('nvidia/Cosmos-Reason1-7B'); print('processor cached')"

# machine config: weights paths (box-local copy of the template) and site.yaml for robotwin/hybrid_eval/launch.py
W=$ASSETS/weights
cat > $REPO/configs/paths/default.yaml <<YAML
wan_vae: $W/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
text_encoder: $W/Cosmos-Reason1-7B
code_model: $W/SDAR-1.7B-Chat
code_vision: $W/Qwen3-VL-2B-Instruct
pretrain_ckpt: /dev/null
robotwin_wam_init_ckpt: /dev/null
robotwin_hybrid_init_ckpt: /dev/null
cosmos_ckpt: /dev/null
robotwin_root: $RT
robotwin_official: /dev/null
robotwin_official_labels: /dev/null
robotwin_cap: /dev/null
robotwin_cap_slices: /dev/null
YAML
ICD=$(ls /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json 2>/dev/null | head -1 || true)
cat > $REPO/configs/robotwin/site.yaml <<YAML
python_bin: $(dirname $(command -v python))
agenticwam_root: \${repo}
robotwin_root: $RT
aspire_root: \${agenticwam_root}/codegen/sim
sim_dir: \${aspire_root}
serving_subdir: capsim/serving
seg_venv: /dev/null
checkpoints_root: $W
hf_cache_policy: /workspace/hf_home
hf_cache_vision: /workspace/hf_home
molmo_venv: /dev/null
hf_token_file: null
vulkan_icd: ${ICD:-/etc/vulkan/icd.d/nvidia_icd.json}
ld_library_paths: []
robotwin_release_dir: $ASSETS/release
dataset_stats: $ASSETS/release/dataset_stats.json
code_model: $W/SDAR-1.7B-Chat
robotwin_cap_slices: /dev/null
diffsynth_model_base: $W
ffmpeg: ffmpeg
viewer_out: \${repo}/runs/rollout_viewer
trace_dir: \${repo}/runs/hybrid_traces
watch_log: \${repo}/runs/hybrid_eval_watch.log
vision:
  molmo: {port: 8122, model: allenai/Molmo2-8B, gpu_util: 0.28, max_num_seqs: 1, limit_mm_per_prompt: {image: 1, video: 0}, log: /tmp/molmo_hyb.log}
  sam3: {port: 8114, log: /tmp/sam3_hyb.log}
  graspnet: {port: 8115, log: /tmp/graspnet_hyb.log}
  ready_timeout_s: 900
YAML
echo "BRAINLOOP sim_setup DONE"
