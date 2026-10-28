"""Modal launcher for the brainloop lab (E1-E4 numerics; E5 latency needs an RTX 5090, which Modal does not rent).

Always pin the workspace on the command line, and set the server-side cap from the projection:
    MODAL_PROFILE=<profile> BRAINLOOP_TIMEOUT_S=<1.5x projected> modal run research/brainloop/env/modal_app.py::fetch
    MODAL_PROFILE=<profile> BRAINLOOP_TIMEOUT_S=... modal run research/brainloop/env/modal_app.py --cmd proof
    MODAL_PROFILE=<profile> modal volume get brainloop-out / research/brainloop/results/out_modal
GPU via BRAINLOOP_GPU (default L40S). retries=0 everywhere: nothing here is checkpointed mid-stage.
"""
import os
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3]
GPU = os.environ.get("BRAINLOOP_GPU", "L40S")
TIMEOUT_S = int(os.environ.get("BRAINLOOP_TIMEOUT_S", str(2 * 3600)))
TASKS = os.environ.get("BRAINLOOP_TASKS", "click_alarmclock beat_block_hammer stack_blocks_three handover_block "
                                          "open_microwave hanging_mug").split()

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(REPO / "research/brainloop/env/requirements-lab.txt"))
    .env({"PYTHONPATH": "/root/AWAM-Fast/src:/root/AWAM-Fast", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(str(REPO), "/root/AWAM-Fast",
                   ignore=[".git", "**/__pycache__", "research/brainloop/results", "checkpoints", "runs"])
)
assets = modal.Volume.from_name("brainloop-assets", create_if_missing=True)
out = modal.Volume.from_name("brainloop-out", create_if_missing=True)
app = modal.App("brainloop", image=image)
RUN = ["python", "-u", "/root/AWAM-Fast/research/brainloop/run.py"]


@app.function(cpu=8, memory=32768, volumes={"/assets": assets}, timeout=3 * 3600, retries=0)
def fetch():
    subprocess.run(RUN + ["fetch", "--assets", "/assets", "--tasks", *TASKS], check=True)
    assets.commit()


@app.function(gpu=GPU, memory=98304, volumes={"/assets": assets, "/out": out}, timeout=TIMEOUT_S, retries=0)
def stage(cmd: str, extra: str = ""):
    subprocess.run(["python", "/root/AWAM-Fast/research/brainloop/env/check_env.py"], check=True)
    try:
        subprocess.run(RUN + [cmd, "--assets", "/assets", "--out", "/out", "--tasks", *TASKS] + extra.split(),
                       check=True)
    finally:
        out.commit()


@app.local_entrypoint()
def main(cmd: str = "proof", extra: str = ""):
    stage.remote(cmd, extra)
