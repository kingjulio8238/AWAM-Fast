"""CPU checks of the lab's equivalence claims on a tiny model built from the real classes.

    PYTHONPATH=src:. python -m pytest research/brainloop/tests/test_cpu_tiny.py -q
"""
import torch

from research.brainloop.lab import joint
from research.brainloop.lab.kvmot import KVCache, mot_forward
from research.brainloop.lab.tiny import build_tiny_model, tiny_inputs

CANVAS, STEPS, TA = 16, 4, 8


def _setup(seed=0):
    m = build_tiny_model(seed=seed, canvas=CANVAS)
    obs, prop, pe = tiny_inputs(m, seed=seed)
    inp = joint.Inputs(obs_video=obs, proprio=prop, prompt_embs=pe, code_instruction="pick the cup",
                       action_horizon=TA)
    return m, inp


def test_lab_decode_matches_model_decode():
    m, inp = _setup()
    torch.manual_seed(7)
    a_ref, _ = m._infer_action_joint(inp.obs_video, inp.proprio, inp.prompt_embs, TA, num_video_frames=9,
                                     num_inference_steps=STEPS, instruction=inp.code_instruction,
                                     max_code_tokens=CANVAS)
    torch.manual_seed(7)
    r = joint.joint_decode(m, inp, num_steps=STEPS, gen_len=CANVAS)
    assert torch.equal(a_ref, r.action), (a_ref - r.action).abs().max()
    assert int((r.canvas == m.code_expert.mask_token_id).sum()) == 0
    assert torch.isfinite(r.commit_conf).all() and (r.commit_step >= 0).all()


def test_mot_forward_matches_generic_mot():
    m, inp = _setup()
    st = joint.prepare_static(m, inp, CANVAS)
    lv = torch.randn(1, 48, 3 * st.Tlpv, st.obs_latents.shape[3], st.obs_latents.shape[4])
    la = torch.randn(1, TA, 14)
    ids = torch.cat([torch.tensor([st.pids]), torch.full((1, CANVAS), m.code_expert.mask_token_id)], 1)
    sig = torch.tensor(0.7)
    ta = torch.tensor([600.0])
    pv1, pa1, cl1, mask = joint._forward(m, st, lv, la, ids, sig, ta, None, None)
    cap = KVCache()
    pv2, pa2, cl2, _ = joint._forward(m, st, lv, la, ids, sig, ta, mask, cap)
    assert torch.equal(pa1, pa2) and torch.equal(pv1, pv2) and torch.equal(cl1, cl2)
    assert len(cap.k) == m.mot.num_layers and cap.seq_len == mask.shape[0] - TA


def test_action_only_with_per_step_cache_is_exact():
    """k = 0, same step count, per-step caches: the motor tier must reproduce the joint action."""
    m, inp = _setup()
    g = torch.Generator().manual_seed(3)
    r = joint.joint_decode(m, inp, num_steps=STEPS, gen_len=CANVAS, generator=g, capture="all")
    g2 = torch.Generator().manual_seed(3)
    torch.randn(1, 48, 3 * r.static.Tlpv, *r.static.obs_latents.shape[3:], generator=g2)  # skip the video draw
    a = joint.fast_action(m, None, inp.prompt_embs, inp.proprio, num_steps=STEPS, action_horizon=TA,
                          generator=g2, caches=r.caches)
    err = (a - r.action).abs().max().item()
    assert err < 1e-5, err


def test_warm_start_at_zero_with_empty_canvas_is_cold():
    m, inp = _setup()
    r0 = joint.joint_decode(m, inp, num_steps=STEPS, gen_len=CANVAS, generator=torch.Generator().manual_seed(5))
    warm = {"start": 0, "video": torch.zeros_like(r0.latents_video), "action": torch.zeros_like(r0.latents_action),
            "canvas": torch.full((CANVAS,), m.code_expert.mask_token_id)}
    r1 = joint.joint_decode(m, inp, num_steps=STEPS, gen_len=CANVAS, generator=torch.Generator().manual_seed(5),
                            warm=warm)
    # sigma_0 is ~1 but not exactly 1, so the zero warm init perturbs the start slightly
    assert (r1.action - r0.action).abs().max() < 5e-2
    assert r1.steps_run == STEPS


def test_warm_start_mid_schedule_runs_fewer_steps_and_keeps_committed_code():
    m, inp = _setup()
    r0 = joint.joint_decode(m, inp, num_steps=STEPS, gen_len=CANVAS, generator=torch.Generator().manual_seed(5))
    canvas = r0.canvas.clone()
    canvas[: CANVAS // 2] = m.code_expert.mask_token_id
    warm = {"start": STEPS - 2, "video": r0.latents_video, "action": r0.latents_action, "canvas": canvas}
    r1 = joint.joint_decode(m, inp, num_steps=STEPS, gen_len=CANVAS, generator=torch.Generator().manual_seed(6),
                            warm=warm)
    assert r1.steps_run == 2
    assert torch.equal(r1.canvas[CANVAS // 2:], r0.canvas[CANVAS // 2:])
    assert int((r1.canvas == m.code_expert.mask_token_id).sum()) == 0
    assert (r1.commit_step[: CANVAS // 2] >= STEPS - 2).all() and (r1.commit_step[CANVAS // 2:] == -1).all()


def test_fast_action_with_stale_cache_runs():
    m, inp = _setup()
    r = joint.joint_decode(m, inp, num_steps=STEPS, gen_len=CANVAS, generator=torch.Generator().manual_seed(1),
                           capture="final")
    for n in (1, 2, 4):
        a = joint.fast_action(m, r.cache, inp.prompt_embs, inp.proprio + 0.1, num_steps=n, action_horizon=TA,
                              generator=torch.Generator().manual_seed(2))
        assert a.shape == (TA, 14) and torch.isfinite(a).all()


def test_cache_content_matters():
    """Guards the exactness test against vacuity: a cache from another observation must change the action."""
    m, inp = _setup()
    r = joint.joint_decode(m, inp, num_steps=STEPS, gen_len=CANVAS, generator=torch.Generator().manual_seed(1),
                           capture="final")
    other = joint.Inputs(obs_video=-inp.obs_video, proprio=inp.proprio, prompt_embs=inp.prompt_embs,
                         code_instruction="open the drawer", action_horizon=TA)
    r2 = joint.joint_decode(m, other, num_steps=STEPS, gen_len=CANVAS, generator=torch.Generator().manual_seed(1),
                            capture="final")
    a1 = joint.fast_action(m, r.cache, inp.prompt_embs, inp.proprio, num_steps=2, action_horizon=TA,
                           generator=torch.Generator().manual_seed(2))
    a2 = joint.fast_action(m, r2.cache, inp.prompt_embs, inp.proprio, num_steps=2, action_horizon=TA,
                           generator=torch.Generator().manual_seed(2))
    assert (a1 - a2).abs().max() > 1e-4


def test_action_rope_stays_complex_and_follows_module():
    m = build_tiny_model()
    m.to(dtype=torch.bfloat16)
    assert m.action_expert.freqs.is_complex()          # the repo fix must not let .to(dtype) flatten the rotation
    assert m.action_expert.freqs.device == next(m.action_expert.parameters()).device


def test_core_path_matches_default_path_across_observations():
    m, inp = _setup()
    st = joint.prepare_static(m, inp, CANVAS)
    bind = joint.make_core(m, st, CANVAS, action_horizon=TA, compile_mode=None)
    r0 = joint.joint_decode(m, inp, num_steps=STEPS, gen_len=CANVAS, generator=torch.Generator().manual_seed(4), static=st)
    r1 = joint.joint_decode(m, inp, num_steps=STEPS, gen_len=CANVAS, generator=torch.Generator().manual_seed(4), static=st,
                            core=bind(st))
    assert torch.equal(r0.action, r1.action) and torch.equal(r0.canvas, r1.canvas)
    other = joint.Inputs(obs_video=inp.obs_video.flip(-1), proprio=inp.proprio * 0.5, prompt_embs=inp.prompt_embs,
                         code_instruction=inp.code_instruction, action_horizon=TA)
    st2 = joint.prepare_static(m, other, CANVAS)
    a = joint.joint_decode(m, other, num_steps=STEPS, gen_len=CANVAS, generator=torch.Generator().manual_seed(4), static=st2)
    b = joint.joint_decode(m, other, num_steps=STEPS, gen_len=CANVAS, generator=torch.Generator().manual_seed(4), static=st2,
                           core=bind(st2))
    assert torch.equal(a.action, b.action)


def test_rebuild_cache_is_exact():
    m, inp = _setup()
    r = joint.joint_decode(m, inp, num_steps=STEPS, gen_len=CANVAS, generator=torch.Generator().manual_seed(1),
                           capture="final")
    c2 = joint.rebuild_cache(m, r.static, r.final_inputs, action_horizon=TA)
    assert all(torch.equal(a, b) for a, b in zip(r.cache.k, c2.k)) and all(torch.equal(a, b) for a, b in zip(r.cache.v, c2.v))
