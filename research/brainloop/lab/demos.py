"""RoboTwin 2.0 official demonstrations (TianxingChen/RoboTwin2.0, aloha-agilex_clean_50) as simulator observations.

Each episode HDF5 holds JPEG frames of the head/left/right cameras (240x320, the simulator's native size) and the
14-D joint vector. `obs_dict` returns exactly the dict the RoboTwin simulator hands the policy, so frames go
through the policy's own eval preprocessing (`_build_obs_video`) unchanged.

Channel order: the HDF5 JPEGs were written from RGB arrays with cv2, so cv2.imdecode returns RGB as-is (checked
against the episode's mp4 by `check_channel_order`, which fails loudly if that ever stops holding).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np

SIM_CAMERAS = ("head_camera", "left_camera", "right_camera")   # = training cam_high, cam_left_wrist, cam_right_wrist
GRIPPER_DIMS = (6, 13)                                         # left, right gripper in the 14-D joint vector
ARM_DIMS = tuple(i for i in range(14) if i not in GRIPPER_DIMS)


@dataclass
class Episode:
    task: str
    index: int
    path: str
    qpos: np.ndarray                     # [T,14] float32
    instructions: List[str]              # the episode's "seen" instructions
    _h5: h5py.File = field(repr=False, default=None)

    @property
    def T(self) -> int:
        return int(self.qpos.shape[0])

    def frame(self, cam: str, t: int) -> np.ndarray:
        return _decode(self.path, cam, int(t))

    def obs_dict(self, t: int) -> Dict:
        return {"observation": {cam: {"rgb": self.frame(cam, t)} for cam in SIM_CAMERAS},
                "joint_action": {"vector": self.qpos[t].astype(np.float32)}}


@lru_cache(maxsize=4096)
def _decode(path: str, cam: str, t: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        buf = f[f"observation/{cam}/rgb"][t]
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode {cam} frame {t} of {path}")
    return img


def task_root(data_root: str, task: str) -> str:
    return os.path.join(data_root, task, "aloha-agilex_clean_50")


def open_episode(data_root: str, task: str, index: int) -> Episode:
    root = task_root(data_root, task)
    path = os.path.join(root, "data", f"episode{index}.hdf5")
    with h5py.File(path, "r") as f:
        qpos = f["joint_action/vector"][:].astype(np.float32)
        for cam in SIM_CAMERAS:
            n = f[f"observation/{cam}/rgb"].shape[0]
            if n != qpos.shape[0]:
                raise ValueError(f"{path}: {cam} has {n} frames, joint vector has {qpos.shape[0]}")
    with open(os.path.join(root, "instructions", f"episode{index}.json")) as fh:
        instr = json.load(fh)
    seen = list(instr.get("seen") or [])
    if not seen:
        raise ValueError(f"{path}: no seen instructions")
    return Episode(task=task, index=index, path=path, qpos=qpos, instructions=seen)


def gripper_events(qpos: np.ndarray, thr: float = 0.5) -> List[Tuple[int, int, str]]:
    """(t, arm, kind) at every open<->closed crossing of a gripper; t is the first frame of the new state.
    Gripper value 1 = open, 0 = closed in RoboTwin's aloha-agilex joint vector."""
    out = []
    for arm, d in enumerate(GRIPPER_DIMS):
        closed = qpos[:, d] < thr
        for t in np.nonzero(np.diff(closed.astype(int)))[0]:
            out.append((int(t + 1), arm, "close" if closed[t + 1] else "open"))
    return sorted(out)


def check_channel_order(data_root: str, task: str, index: int = 0, n: int = 40) -> Dict[str, float]:
    """Compare decoded HDF5 head frames against the episode mp4 (decoded by cv2 as BGR). The HDF5 decode must
    match the mp4's RGB view better than its BGR view; raise otherwise."""
    root = task_root(data_root, task)
    mp4 = os.path.join(root, "video", f"episode{index}.mp4")
    ep = open_episode(data_root, task, index)
    cap = cv2.VideoCapture(mp4)
    d_rgb, d_bgr = [], []
    for t in range(min(n, ep.T)):
        ok, bgr = cap.read()
        if not ok:
            break
        j = ep.frame("head_camera", t).astype(np.float32)
        bgr = bgr.astype(np.float32)
        d_rgb.append(np.abs(j - bgr[..., ::-1]).mean())
        d_bgr.append(np.abs(j - bgr).mean())
    cap.release()
    if not d_rgb:
        raise RuntimeError(f"could not read {mp4}")
    res = {"mean_abs_vs_rgb": float(np.mean(d_rgb)), "mean_abs_vs_bgr": float(np.mean(d_bgr)), "frames": len(d_rgb)}
    if not res["mean_abs_vs_rgb"] < res["mean_abs_vs_bgr"]:
        raise RuntimeError(f"HDF5 frames look BGR, not RGB: {res}")
    return res
