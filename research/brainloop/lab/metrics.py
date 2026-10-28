"""Small, dependency-free metrics shared by the experiments and the analysis."""
from __future__ import annotations

import difflib
import math
from typing import Dict, Sequence

import numpy as np

from .demos import ARM_DIMS, GRIPPER_DIMS


def action_err(pred: np.ndarray, ref: np.ndarray, h: int) -> Dict[str, float]:
    """Mean |error| over the first h steps, arm joints (rad) and grippers (0..1) separately."""
    h = min(h, len(pred), len(ref))
    if h <= 0:
        return {"arm": float("nan"), "grip": float("nan"), "h": 0}
    d = np.abs(np.asarray(pred[:h], dtype=np.float64) - np.asarray(ref[:h], dtype=np.float64))
    return {"arm": float(d[:, list(ARM_DIMS)].mean()), "grip": float(d[:, list(GRIPPER_DIMS)].mean()), "h": int(h)}


def canvas_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    n = min(len(a), len(b))
    return float(np.mean([int(a[i]) == int(b[i]) for i in range(n)])) if n else float("nan")


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def rel_mse(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(((pred - ref) ** 2).mean() / max(1e-12, (ref ** 2).mean()))


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney AUROC with ties averaged; nan if a class is empty."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def pct(xs: Sequence[float], q: float) -> float:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.percentile(xs, q)) if xs else float("nan")
