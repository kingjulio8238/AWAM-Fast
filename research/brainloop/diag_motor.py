#!/usr/bin/env python3
"""A5 diagnostic: is the motor-tier gap on GPU a kernel-numerics effect or a real difference?

For a few anchors: joint decode with per-step caches; action-only replay with the same caches (unmasked = fused
SDPA kernel, and with an all-True mask = the joint pass's backend); compare as the seed floor is measured
(mean h8 arm / gripper error), against two seeds of the joint decode.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.brainloop.run import _real_adapter, paths_for  # noqa: E402,F401
from research.brainloop.lab import joint  # noqa: E402
from research.brainloop.lab.demos import open_episode  # noqa: E402
from research.brainloop.lab.metrics import action_err  # noqa: E402


class A:  # argparse-like
    assets, dtype, tiny = "/workspace/assets", sys.argv[1] if len(sys.argv) > 1 else "bf16", False


ep = open_episode("/workspace/assets/robotwin", "click_alarmclock", 0)
ad, _ = _real_adapter(A, "click_alarmclock", [ep])
m = ad.model
rows = []
for t in (0, 16, 32):
    inp = ad.inputs(ep.obs_dict(t), ep.instructions[0])
    g = lambda s: torch.Generator(device=m.device).manual_seed(s)  # noqa: E731
    r = joint.joint_decode(m, inp, num_steps=20, gen_len=ad.gen_len, generator=g(1), capture="all")
    r2 = joint.joint_decode(m, inp, num_steps=20, gen_len=ad.gen_len, generator=g(2), static=r.static)
    ref, alt = ad.denorm(r.action), ad.denorm(r2.action)
    out = {"t": t, "seed_floor": action_err(alt, ref, 8)}
    for fm in (False, True):
        g3 = g(1)
        st = r.static
        torch.randn(1, st.obs_latents.shape[1], m.n_cameras * st.Tlpv, *st.obs_latents.shape[3:], device=m.device,
                    dtype=next(m.action_expert.parameters()).dtype, generator=g3)
        a = joint.fast_action(m, None, inp.prompt_embs, inp.proprio, num_steps=20, generator=g3, caches=r.caches,
                              force_mask=fm)
        out["motor_" + ("masked" if fm else "fused")] = action_err(ad.denorm(a), ref, 8)
    rows.append(out)
    print(json.dumps(out), flush=True)
json.dump(rows, open(f"/workspace/out/diag_motor_{A.dtype}.json", "w"), indent=1)
print("BRAINLOOP diag DONE", flush=True)
