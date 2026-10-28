"""Why does T1 move the motor away from the demonstrations? Compare, on the BASE model, the flow-matching loss against
(a) the demo chunk at offsets 0 and 1, (b) the model's own warm-1 chunk; and check the normalisation round trip."""
import json, sys, random
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.brainloop.run import _real_adapter
from research.brainloop.lab import joint
from research.brainloop.lab.demos import open_episode
from research.brainloop.lab.kvmot import action_only_forward
from research.brainloop.lab.train_c import _load_dir, _static_from_d1, _fi, norm_action

class A: assets, dtype, tiny = "/workspace/assets", "bf16", False
d1 = _load_dir("/workspace/out_c1/d1")
eps = {(d["task"], d["ep"]): open_episode("/workspace/assets/robotwin", d["task"], d["ep"]) for d in d1}
ad, _ = _real_adapter(A, d1[0]["task"], list(eps.values()))
m = ad.model; dev = m.device; sch = m.action_scheduler
q = eps[(d1[0]["task"], d1[0]["ep"])].qpos[:32]
rt = ad.denorm(norm_action(ad, q).unsqueeze(0).to(torch.bfloat16))
print("norm round trip max|err| (rad):", float(np.abs(rt - q).max()), " norm range:", float(norm_action(ad, q).abs().max()))
torch.manual_seed(0)
res = {"gt0": [], "gt1": [], "self": []}
for d in d1:
    ep = eps[(d["task"], d["ep"])]
    for r in d["recs"][:6]:
        a = r["a"]
        if a + 34 > ep.T: continue
        _, st = _static_from_d1(ad, ep, r)
        cache = joint.rebuild_cache(m, st, _fi(r))
        tg = {"gt0": norm_action(ad, ep.qpos[a:a+32]), "gt1": norm_action(ad, ep.qpos[a+1:a+33]), "self": r["action"].float()}
        prop = ad.proprio(ep.qpos[a])
        t = torch.full((1,), 500.0, device=dev)
        for name, x0 in tg.items():
            x0 = x0.unsqueeze(0).to(dev)
            noise = torch.randn_like(x0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ctx, cm = m._action_context(st.prompt_embs, prop.unsqueeze(1))
                pre = m.action_expert.pre_dit(action_tokens=sch.add_noise(x0, noise, t), timestep=t, context=ctx, context_mask=cm)
                aux = {"t_mod": pre["t_mod"], "freqs": pre["freqs"], "context": {"context": pre["context"], "mask": pre["context_mask"]}}
                pred = m.action_expert.post_dit(action_only_forward(m.mot, pre["tokens"], aux, cache), pre)
            res[name].append(float(((pred.float() - (noise - x0).float()) ** 2).mean()))
print({k: round(float(np.mean(v)), 5) for k, v in res.items()}, "n=", len(res["gt0"]))
