"""Build the released RoboTwin WAM policy exactly as the RoboTwin eval does, without the simulator.

The eval chain is scripts/eval_robotwin_release.py -> robotwin/hybrid_eval/launch.py -> robotwin/eval/run_task.py ->
RoboTwin eval_policy.py -> robotwin/deploy/deploy_policy.get_model(usr_args). `build_release_policy` calls the same
`get_model` with the usr_args and environment that chain produces for the `robotwin_wam` preset
(configs/robotwin/models/robotwin_wam.yaml), so model config, checkpoint load, preprocessing, prompt composition
and normalisation are the eval's own code. Two departures, both outside the model:
  * the Cosmos-Reason1 text encoder is replaced by embeddings precomputed with that same encoder class
    (`precompute_text_embeds`), so the 7B encoder never shares the GPU with the policy;
  * the policy's own decode is bypassed by lab/joint.py, which is checked against it (`parity_check`).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
DEPLOY_DIR = REPO / "robotwin" / "deploy"

# robotwin_wam preset env (configs/robotwin/models/robotwin_wam.yaml) + what launch.py derives for it. The three
# knobs the preset sets to their defaults (CONF_THRESH=0, REVEAL_CAP=9999, NN_REPAIR=0) are left unset: unset and
# default are the same decode, and fast_joint_infer treats ANY set value as "research mode".
WAM_PRESET = {"canvas": 128, "nis": 20, "replan": 24, "action_dim": 14, "eval_preproc": "parity_crop",
              "eval_preproc_crop": "0.95", "instruction_type": "seen"}


def release_env(task: str, release_dir: str, dtype: str) -> Dict[str, str]:
    return {
        "TASK": task,
        "AGENTICWAM_TASK_SKILL_PROMPT": str(Path(release_dir) / "task_skill_prompt.json"),
        "AGENTICWAM_TASK_INSTRUCTION_MAP": "",
        "AGENTICWAM_EVAL_PREPROC": WAM_PRESET["eval_preproc"],
        "AGENTICWAM_EVAL_PREPROC_CROP": WAM_PRESET["eval_preproc_crop"],
        "AGENTICWAM_INFER_CODE_TOKENS": str(WAM_PRESET["canvas"]),
        "AGENTICWAM_HYBRID": "0",
        "AGENTICWAM_ACTION_SPACE": "",
        "AGENTICWAM_DEPLOY_DTYPE": dtype,
        "HYDRA_FULL_ERROR": "1",
    }


def _import_deploy_policy():
    for p in (str(REPO), str(REPO / "src"), str(DEPLOY_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import deploy_policy  # noqa: E402
    return deploy_policy


def wrapped_instruction(instruction: str) -> str:
    from agenticwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    return DEFAULT_PROMPT.format(task=instruction)


def _key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def precompute_text_embeds(prompts: Iterable[str], text_encoder_dir: str, out_dir: str, device: str = "cuda",
                           max_len: int = 128) -> int:
    """Embed every wrapped prompt with the eval's own _CosmosTextEncoder; one .pt per prompt."""
    dp = _import_deploy_policy()
    os.makedirs(out_dir, exist_ok=True)
    todo = [p for p in dict.fromkeys(prompts) if not os.path.exists(os.path.join(out_dir, _key(p) + ".pt"))]
    if not todo:
        return 0
    enc = dp._CosmosTextEncoder(text_encoder_dir, device, max_len=max_len)
    for p in todo:
        e = enc.encode(p).detach().to("cpu", torch.bfloat16)
        torch.save({"prompt": p, "emb": e}, os.path.join(out_dir, _key(p) + ".pt"))
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return len(todo)


class CachedTextEncoder:
    """Drop-in for deploy_policy._CosmosTextEncoder that serves precomputed embeddings and fails on a miss."""
    cache_dir: str = ""

    def __init__(self, ckpt_dir, device, max_len=128, dtype=torch.bfloat16):
        self.device = device
        self._cache: Dict[str, torch.Tensor] = {}

    def encode(self, prompt: str) -> torch.Tensor:
        if prompt not in self._cache:
            path = os.path.join(self.cache_dir, _key(prompt) + ".pt")
            if not os.path.exists(path):
                raise KeyError(f"no precomputed embedding for prompt {prompt!r} ({path})")
            d = torch.load(path, map_location="cpu")
            if d["prompt"] != prompt:
                raise RuntimeError(f"embedding cache collision at {path}")
            e = d["emb"]
            if float(e.float().std()) < 1e-3:
                raise RuntimeError(f"cached embedding for {prompt!r} is ~zero (std {float(e.float().std()):.2e})")
            self._cache[prompt] = e.to(self.device)
        return self._cache[prompt]


def build_release_policy(*, task: str, release_dir: str, text_cache_dir: str, paths: Dict[str, str],
                         dtype: str = "bf16", device: str = "cuda"):
    """The released WAM policy via deploy_policy.get_model, as the RoboTwin eval builds it. Returns (policy, load_report)."""
    for k, v in release_env(task, release_dir, dtype).items():
        os.environ[k] = v
    dp = _import_deploy_policy()
    CachedTextEncoder.cache_dir = text_cache_dir
    dp._CosmosTextEncoder = CachedTextEncoder

    from agenticwam.models.cosmos_mot.agenticwam_model import AgenticWAMCosmos
    report: Dict = {}
    orig_load = AgenticWAMCosmos.load_checkpoint

    def _load(self, path, optimizer=None):
        res = orig_load(self, path, optimizer)
        report["unexpected"] = list(res.unexpected_keys)
        report["missing"] = list(res.missing_keys)
        return res

    AgenticWAMCosmos.load_checkpoint = _load
    try:
        overrides = "|".join(f"paths.{k}={v}" for k, v in paths.items())
        usr_args = {
            "sim_cfg_path": str(REPO / "configs" / "robotwin" / "sim_robotwin.yaml"),
            "sim_task": "robotwin_wam",
            "sim_paths_overrides": overrides,
            "use_code_expert": True,
            "code_model_name": paths["code_model"],
            "ckpt_setting": str(Path(release_dir) / "wam.pt"),
            "device": device,
            "dataset_stats_path": str(Path(release_dir) / "dataset_stats.json"),
            "action_horizon": None,
            "replan_steps": WAM_PRESET["replan"],
            "num_inference_steps": WAM_PRESET["nis"],
            "eval_output_dir": None,
        }
        policy = dp.get_model(usr_args)
    finally:
        AgenticWAMCosmos.load_checkpoint = orig_load
    report.update(validate_release_policy(policy, report))
    return policy, report


def validate_release_policy(policy, load: Dict) -> Dict:
    """Assert the settings that make the policy the released WAM; raise on any mismatch."""
    m = policy.model
    problems = []
    unexpected = load.get("unexpected", [])
    if unexpected:
        problems.append(f"{len(unexpected)} checkpoint tensors rejected, e.g. {unexpected[:3]}")
    allowed_missing = ("vae.", "code_vision.vit.")
    bad_missing = [k for k in load.get("missing", []) if not k.startswith(allowed_missing)
                   or k.startswith("code_vision.vit.merger.")]
    if bad_missing:
        problems.append(f"{len(bad_missing)} trainable tensors missing from the checkpoint, e.g. {bad_missing[:3]}")
    checks = {
        "use_code_expert": bool(m.use_code_expert),
        "code_objective": getattr(m.code_expert, "code_objective", None),
        "code_vision": type(getattr(m, "code_vision", None)).__name__,
        "n_cameras": int(m.n_cameras),
        "action_dim": int(m.action_expert.action_dim),
        "infer_code_tokens": int(m._resolve_infer_code_tokens()),
        "mask_token_id": getattr(m.code_expert, "mask_token_id", None),
        "preproc": getattr(policy, "_preproc_mode", None),
        "action_horizon": int(policy.action_horizon),
        "num_video_frames": int(policy.num_video_frames),
        "dtype": str(next(m.action_expert.parameters()).dtype),
    }
    want = {"use_code_expert": True, "code_objective": "joint", "code_vision": "CodeVisionPrefix", "n_cameras": 3,
            "action_dim": WAM_PRESET["action_dim"], "infer_code_tokens": WAM_PRESET["canvas"],
            "preproc": WAM_PRESET["eval_preproc"], "action_horizon": 32, "num_video_frames": 9}
    for k, v in want.items():
        if checks[k] != v:
            problems.append(f"{k}={checks[k]!r}, expected {v!r}")
    if checks["mask_token_id"] is None:
        problems.append("code expert has no mask token")
    if problems:
        raise RuntimeError("released-policy validation failed:\n  " + "\n  ".join(problems))
    return {"checks": checks, "n_missing_allowed": len(load.get("missing", []))}


class Adapter:
    """The policy's own observation / state / prompt handling, exposed as lab/joint.py Inputs."""

    def __init__(self, policy, task: str):
        self.policy = policy
        self.model = policy.model
        self.task = task

    def inputs(self, obs: Dict, instruction: str, *, obs_video_override=None):
        from .joint import Inputs
        os.environ["TASK"] = self.task            # _skill_prompt() looks the task up by this
        p = self.policy
        obs_video = p._build_obs_video(obs) if obs_video_override is None else obs_video_override
        state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
        proprio = p._normalize_state(state).to(p.device)
        wrapped = wrapped_instruction(instruction)
        code_instruction = p._skill_prompt() + wrapped          # deploy_policy's default branch
        prompt_embs = p.text.encode(wrapped).to(p.device)
        return Inputs(obs_video=obs_video, proprio=proprio, prompt_embs=prompt_embs,
                      code_instruction=code_instruction, action_horizon=int(p.action_horizon),
                      num_video_frames=int(p.num_video_frames))

    def proprio(self, qpos: np.ndarray) -> torch.Tensor:
        return self.policy._normalize_state(np.asarray(qpos, dtype=np.float32)).to(self.policy.device)

    def denorm(self, a: torch.Tensor) -> np.ndarray:
        return self.policy._denormalize_action(a)[0]

    def obs_video(self, obs: Dict) -> torch.Tensor:
        return self.policy._build_obs_video(obs)

    @property
    def gen_len(self) -> int:
        return int(self.model._resolve_infer_code_tokens())


def build_tiny_adapter(task: str, release_dir: str, seed: int = 0):
    """CPU stand-in: the tiny model behind the REAL eval preprocessing, processor, stats and skill prompts."""
    dp = _import_deploy_policy()
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from agenticwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from agenticwam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
    from .tiny import TEXT_IN, build_tiny_model

    for k, v in release_env(task, release_dir, "fp32").items():
        os.environ[k] = v
    cfg = dp._compose_sim_cfg(str(REPO / "configs" / "robotwin" / "sim_robotwin.yaml"), None, "robotwin_wam", None)
    OmegaConf.set_struct(cfg, False)
    p = object.__new__(dp.AgenticWAMCosmosRobotWinPolicy)
    p.model = build_tiny_model(seed=seed, adim=14, canvas=WAM_PRESET["canvas"])
    p.device = "cpu"
    p.n_cameras = 3
    p.processor = instantiate(cfg.data.train.processor).eval()
    p.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(Path(release_dir) / "dataset_stats.json")))
    p._resize = ResizeSmallestSideAspectPreserving(args={"img_w": 320, "img_h": 224})
    p._crop = CenterCrop(args={"img_w": 320, "img_h": 224})
    p._norm = Normalize(args={"mean": 0.5, "std": 0.5})
    p._preproc_mode = WAM_PRESET["eval_preproc"]
    p._preproc_crop = float(WAM_PRESET["eval_preproc_crop"])
    p.action_horizon, p.num_video_frames, p.replan_steps, p.num_inference_steps = 32, 9, 24, 20

    class _FakeText:
        def encode(self, prompt):
            g = torch.Generator().manual_seed(int(_key(prompt)[:8], 16))
            return torch.randn(1, 8, TEXT_IN, generator=g)

    p.text = _FakeText()
    return Adapter(p, task)
