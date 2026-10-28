# Brainloop lab image. Same steps as env/bootstrap.sh, baked: base torch 2.7.1 + CUDA 12.8 (sm_120 kernels, i.e.
# RTX 5090) + the repo's pinned deps. The devel variant is deliberate: Triton (torch.compile / CUDA graphs in E5)
# compiles its launcher with the system C compiler, which the runtime image lacks.
#   docker build -f research/brainloop/env/Dockerfile -t brainloop:latest .
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN apt-get update && apt-get install -y --no-install-recommends git rsync tmux ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/AWAM-Fast
COPY research/brainloop/env/requirements-lab.txt /tmp/requirements-lab.txt
RUN python -c "import torch; open('/tmp/torch_version','w').write(torch.__version__)" \
    && pip install --no-cache-dir -r /tmp/requirements-lab.txt \
    && python -c "import torch; v=open('/tmp/torch_version').read(); assert torch.__version__==v, (torch.__version__, v)"
COPY . /workspace/AWAM-Fast
RUN pip install --no-cache-dir --no-deps -e . \
    && python research/brainloop/env/check_env.py --no-gpu
