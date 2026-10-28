#!/usr/bin/env python3
"""Prove the environment before any weights are touched: every import, the torch build, and (with a GPU) that the
installed kernels run on this card. Exits non-zero on the first problem."""
import argparse
import importlib
import json
import sys

MODULES = ["torch", "torchvision", "transformers", "accelerate", "hydra", "omegaconf", "einops", "safetensors", "av",
           "boto3", "h5py", "cv2", "matplotlib", "huggingface_hub", "hf_transfer",
           "agenticwam.models.cosmos_mot.build", "agenticwam.models.cosmos_mot.code_vision",
           "agenticwam.datasets.lerobot.processors.agenticwam_processor",
           "transformers.models.qwen3_vl.modeling_qwen3_vl", "transformers.models.qwen2_5_vl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    rep = {}
    for m in MODULES:
        importlib.import_module(m)
        rep[m] = "ok"
    import torch
    rep["torch"] = torch.__version__
    rep["torch_cuda"] = torch.version.cuda
    if not torch.__version__.startswith("2.7.1"):
        sys.exit(f"torch {torch.__version__}, expected 2.7.1 (pyproject pin)")
    rep["arch_list"] = torch.cuda.get_arch_list() if torch.cuda.is_available() or a.no_gpu else []
    if not a.no_gpu:
        if not torch.cuda.is_available():
            sys.exit("no CUDA device visible")
        cap = torch.cuda.get_device_capability(0)
        rep["device"] = torch.cuda.get_device_name(0)
        rep["capability"] = f"sm_{cap[0]}{cap[1]}"
        if rep["capability"] not in rep["arch_list"]:
            sys.exit(f"this torch build has no kernels for {rep['capability']}: {rep['arch_list']}")
        x = torch.randn(1, 16, 1024, 128, device="cuda", dtype=torch.bfloat16)
        y = torch.nn.functional.scaled_dot_product_attention(x, x, x)
        m = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        rep["sdpa_ok"] = bool(torch.isfinite(y).all())
        rep["matmul_ok"] = bool(torch.isfinite(m @ m).all())
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        rep["gpu_mem_gb"] = round(total / 2 ** 30, 1)
    print(json.dumps(rep, indent=1))
    print("BRAINLOOP check_env DONE", flush=True)


if __name__ == "__main__":
    main()
