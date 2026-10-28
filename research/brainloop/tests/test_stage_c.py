import json
"""CPU checks for Stage C: staleness module, teacher data, T1/T2/T3 trainers, the sim execution modes.

    PYTHONPATH=src:. python -m pytest research/brainloop/tests/test_stage_c.py -q
Needs the tiny assets dir (release/ + robotwin/<task>) in BRAINLOOP_TINY_ASSETS; skipped otherwise.
"""
import os

import numpy as np
import pytest
import torch

from research.brainloop.lab import joint, staleness
from research.brainloop.lab.tiny import build_tiny_model, tiny_inputs

ASSETS = os.environ.get("BRAINLOOP_TINY_ASSETS", "")
needs_assets = pytest.mark.skipif(not os.path.isdir(os.path.join(ASSETS, "release")), reason="no tiny assets")


def test_staleness_zero_init_is_identity():
    m = build_tiny_model()
    obs, prop, pe = tiny_inputs(m)
    inp = joint.Inputs(obs_video=obs, proprio=prop, prompt_embs=pe, code_instruction="x", action_horizon=8)
    r = joint.joint_decode(m, inp, num_steps=3, gen_len=16, generator=torch.Generator().manual_seed(0), capture="final")
    a0 = joint.fast_action(m, r.cache, pe, prop, num_steps=2, action_horizon=8, generator=torch.Generator().manual_seed(1))
    staleness.install(m.action_expert)
    for k in (0, 8, 24):
        a = joint.fast_action(m, r.cache, pe, prop, num_steps=2, action_horizon=8,
                              generator=torch.Generator().manual_seed(1), stale_k=k)
        assert torch.equal(a, a0)
    with torch.no_grad():
        m.action_expert.time_embedding.stale.weight.normal_()
    a8 = joint.fast_action(m, r.cache, pe, prop, num_steps=2, action_horizon=8,
                           generator=torch.Generator().manual_seed(1), stale_k=8)
    assert not torch.equal(a8, a0)


@needs_assets
def test_stage_c_pipeline_end_to_end(tmp_path):
    from research.brainloop.lab import policy_io, teacher, train_c, eval_c
    from research.brainloop.lab.demos import open_episode
    task = "beat_block_hammer"
    data = os.path.join(ASSETS, "robotwin")
    ad = policy_io.build_tiny_adapter(task, os.path.join(ASSETS, "release"))
    # D1 on ep 0 (train) and ep 2 (held out, has D2); D2 on ep 2
    for i in (0, 2):
        ep = open_episode(data, task, i)
        ep.qpos = ep.qpos[:57]                            # keep the CPU test short: 4 anchors at stride 8
        teacher.episode_d1(ad, ep, str(tmp_path), stride=8)
    ep2 = open_episode(data, task, 2); ep2.qpos = ep2.qpos[:57]
    teacher.episode_d2(ad, ep2, str(tmp_path), stride=8)
    res = train_c.train_motor(ad, str(tmp_path / "d1"), data, str(tmp_path / "t1"), steps=6, batch=4, kmax=8,
                              heldout_from=1, log_every=2, save_every=6)
    assert np.isfinite(res["final_loss"])
    pairs = train_c.t23_pairs(str(tmp_path / "d1"), str(tmp_path / "d2"), ks=(8,))
    assert pairs
    for p in pairs:                                      # fake a train/eval split on one episode
        p["ep"] = 0 if p["a"] < 16 else 18
    r2 = train_c.train_shift(pairs, str(tmp_path / "t2"), epochs=2, eval_from=18, device="cpu")
    assert np.isfinite(r2["eval_mse_t2"])
    den = lambda a: ad.denorm(a)  # noqa: E731
    r3 = train_c.train_surprise(pairs, str(tmp_path / "t3"), eval_from=18, epochs=5, denorm=den)
    assert r3["n_train"] > 0
    m1 = eval_c.eval_motor(ad, str(tmp_path / "d1"), str(tmp_path / "d2"), data, str(tmp_path / "t1" / "motor_t1.pt"),
                           str(tmp_path), heldout_from=2, ks=(0, 8))
    assert "t1_k8" in m1 or "t1_k0" in m1


class _FakeEnv:
    def __init__(self, q, lim=40):
        self.q, self.take_action_cnt, self.step_lim, self.eval_success = q.copy(), 0, lim, False

    def take_action(self, a, action_type="qpos"):
        if self.take_action_cnt == self.step_lim or self.eval_success:     # RoboTwin's own guard
            return
        self.take_action_cnt += 1
        self.q = np.asarray(a, dtype=np.float32)

    def get_instruction(self):
        return "hit the block with the hammer"


@needs_assets
@pytest.mark.parametrize("mode,realtime", [("released", False), ("released", True), ("warm1", False),
                                           ("warm1", True), ("brain", True), ("reflex", False), ("reflex", True)])
def test_sim_modes_run_and_account_time(tmp_path, mode, realtime):
    from research.brainloop.lab import policy_io, sim_policy
    from research.brainloop.lab.demos import open_episode
    task = "beat_block_hammer"
    ad = policy_io.build_tiny_adapter(task, os.path.join(ASSETS, "release"))
    pol = ad.policy
    pol.reset = lambda: None
    os.environ["TASK"] = task
    cfg = {"config": mode, "realtime": realtime, "replan": 8}
    if mode == "reflex":         # the Stage D config: the slow decode must leave GPU time for the reflex
        cfg.update(cold_steps=8, latency_ms={"cold": 335.0}, trace_path=str(tmp_path / "trace.jsonl"))
    if mode == "brain":
        staleness.install(pol.model.action_expert)
        torch.save({"action_expert": staleness.state_dict(pol.model.action_expert)}, tmp_path / "m.pt")
        cfg["motor_ckpt"] = str(tmp_path / "m.pt")
    bl = sim_policy.install(pol, cfg)
    ep = open_episode(os.path.join(ASSETS, "robotwin"), task, 0)
    env = _FakeEnv(ep.qpos[0], lim=70)
    while env.take_action_cnt < env.step_lim:
        obs = ep.obs_dict(0)
        obs["joint_action"]["vector"] = env.q
        pol.step(env, obs)
    assert env.take_action_cnt == env.step_lim
    if mode == "released" and realtime:
        assert bl.stats["held"] >= min(bl.steps_for("released"), env.step_lim)   # blocking: the arm waited
    if mode == "brain":
        assert bl.stats["motor"] > 0
    if mode == "reflex":
        assert bl.stats["reflex"] > 0 and bl.stats["cold"] >= 2          # both rates ran
        if realtime:
            assert bl.stats["held"] >= bl.steps_for("cold")              # the first plan was waited for
        bl.report()
        tr = [json.loads(l) for l in open(tmp_path / "trace.jsonl")][-1]["trace"]
        assert len(tr) == env.step_lim and {x["src"] for x in tr} >= {"cold", "reflex"}
