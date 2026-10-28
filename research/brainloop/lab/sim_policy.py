"""C5: the brainloop execution modes inside the RoboTwin simulator, plus real-time (latency-injected) execution.

Installed onto the released policy object by `install(policy, cfg)`; deploy_policy.get_model calls it when
AGENTICWAM_BRAINLOOP is set (a JSON config). RoboTwin's loop is: obs = get_obs(); eval(env, policy, obs), one
control step per call, so every step sees a fresh observation.

Configs:
  released  the shipped behaviour: cold 20-step decode, play 24 actions, replan.
  warm1     warm 1-step replans every `replan` steps, each warm-started from the previous replan.
  brain     warm1 deliberation every `replan` steps, or earlier when the T3 surprise head fires; in between, the T1
            motor tier re-plans every step from the k-step-stale deliberation cache with fresh proprio.
  reflex    two rates on one serial GPU: a slow cold decode (`cold_steps`) every `slow_every` steps moves the code and
            the video prediction forward; in between, every `reflex_every` steps, the reflex tier re-plans the
            action from noise against a cache rebuilt from the FRESH camera frames (joint.reflex_action).

Real-time mode (`realtime: true`): each policy call costs `latency_ms[config]` of wall-clock (measured on the RTX 5090
in Stage B, not the sim box), converted to control steps at `control_hz` (30).
  blocking (released): the robot holds its pose for d steps while the model thinks; the new chunk starts at index 0.
  async (warm1/brain): the robot keeps executing the current plan; the plan computed from obs(t) is adopted at t+d
            from its index d, so actions stay time-aligned.
Every held or executed step counts against RoboTwin's step limit, so a slow policy loses time like a real robot.
"""
from __future__ import annotations

import atexit
import json
import math
import os
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch

from . import joint
from .experiments import _gen, remask, shift_action, shift_video

# Stage B on the RTX 5090: released = the shipped eager 20-step decode; cold = the same decode CUDA-graphed (p95);
# warm1 = one warm joint step + VAE encode, e2e; motor = motor step + VAE encode.
# reflex: an estimate until D2 measures it: encode ~28 (compiled warm-1 e2e 66 - joint step 38) + one compiled joint
# step 37.9 + 4 compiled action steps 17.4 (Stage A/B p95s)
LATENCY_MS_5090 = {"released": 1293.0, "cold": 799.0, "warm1": 66.0, "motor": 9.1 + 10.4, "reflex": 83.0}


class Brainloop:
    def __init__(self, policy, cfg: Dict):
        self.p = policy
        self.m = policy.model
        self.cfg = {"config": "released", "replan": 8, "realtime": False, "control_hz": 30.0, "surprise_thr": 0.5,
                    "min_gap": 4, "cold_every": 0, "remask": 0.5, "cold_steps": 20,
                    "slow_every": 24, "reflex_every": 4, "reflex_steps": 4, "sync_offset": 0, **cfg}
        if self.cfg["config"] == "reflex":
            self.cfg["cold_every"] = 1                                  # every slow plan is a cold decode
        self.mode = self.cfg["config"]
        self.lat = {**LATENCY_MS_5090, **self.cfg.get("latency_ms", {})}
        self.motor_ok = False
        self.shift = None
        self.surprise = None
        if self.cfg.get("motor_ckpt"):
            from .train_c import load_motor
            load_motor(self.m, self.cfg["motor_ckpt"])
            self.motor_ok = True
        if self.cfg.get("shift_ckpt"):
            from .train_c import load_shift
            self.shift = load_shift(self.cfg["shift_ckpt"], self.m.device)
        if self.cfg.get("surprise_ckpt"):
            d = torch.load(self.cfg["surprise_ckpt"], map_location="cpu")
            net = torch.nn.Sequential(torch.nn.Linear(d["dim"], 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
            net.load_state_dict(d["net"])
            self.surprise = (net.eval(), d["mu"], d["sd"])
        if self.mode == "brain" and not self.motor_ok:
            raise RuntimeError("config 'brain' needs motor_ckpt (the T1 weights)")
        self.reset()

    # ---- bookkeeping --------------------------------------------------------------------------------------------
    def reset(self):
        self.t = 0
        self.queue: deque = deque()
        self.prev = None                 # last deliberation result (joint.Result)
        self.t_delib = -10 ** 9
        self.pending = None              # (ready_step, chunk, start_index)
        self.last_cold = False
        self.t_reflex = -10 ** 9
        self.src = None                  # which computation the queue came from (trace only)
        self.trace = []
        self.stats = {"delib": 0, "cold": 0, "motor": 0, "reflex": 0, "held": 0, "surprise_triggers": 0}
        self.env = None

    def report(self):
        """One line per episode in the eval log, so a run can prove the wrapper (and its latency model) was active."""
        if self.t or self.stats["held"]:
            env = self.env
            print("BRAINLOOP_EPISODE " + json.dumps({"config": self.mode, "realtime": self.cfg["realtime"], "t": self.t,
                  "steps": getattr(env, "take_action_cnt", None), "step_lim": getattr(env, "step_lim", None),
                  "success": bool(getattr(env, "eval_success", False)), **self.stats}), flush=True)
            if self.cfg.get("trace_path") and self.trace:
                with open(self.cfg["trace_path"], "a") as fh:
                    fh.write(json.dumps({"stats": self.stats, "t": self.t, "trace": self.trace}) + "\n")

    def steps_for(self, key) -> int:
        if not self.cfg["realtime"]:
            return 0
        return int(math.ceil(self.lat[key] / (1000.0 / self.cfg["control_hz"]) - 1e-9))

    # ---- model calls --------------------------------------------------------------------------------------------
    def _instr(self, task_env):
        return task_env.get_instruction()

    def deliberate(self, obs, task_env):
        p, m = self.p, self.m
        from .policy_io import Adapter
        ad = Adapter(p, os.environ.get("TASK", ""))
        inp = ad.inputs(obs, self._instr(task_env))
        st = joint.prepare_static(m, inp, ad.gen_len)
        gl = ad.gen_len
        ce = self.cfg["cold_every"]
        self.last_cold = self.prev is None or self.mode == "released" or bool(ce and self.stats["delib"] % ce == 0)
        if self.last_cold:
            r = joint.joint_decode(m, inp, num_steps=self.cfg["cold_steps"], gen_len=gl, static=st, capture="final",
                                   generator=_gen(m, 11, self.t))
        else:
            k = self.t - self.t_delib
            if self.shift is not None:
                from .train_c import apply_shift
                vid = apply_shift(self.shift[0], self.shift[1], self.prev.latents_video, st.obs_latents, k)
            else:
                vid = shift_video(self.prev.latents_video, m.n_cameras, st.Tlpv, k)
            warm = {"start": 19, "video": vid, "action": shift_action(self.prev.latents_action, min(k, 31)),
                    "canvas": remask(self.prev.canvas, self.prev.commit_conf, self.cfg["remask"],
                                                m.code_expert.mask_token_id)}
            r = joint.joint_decode(m, inp, num_steps=20, gen_len=gl, static=st, capture="final", warm=warm,
                                   generator=_gen(m, 11, self.t))
            keep = ~torch.isfinite(r.commit_conf)
            r.commit_conf[keep] = self.prev.commit_conf[keep]
        if self.mode == "brain":
            r.cache = joint.rebuild_cache(m, st, r.final_inputs) if r.cache is None else r.cache
        r._inp = inp
        self.stats["cold"] += int(self.last_cold)
        self.prev, self.t_delib = r, self.t
        self.stats["delib"] += 1
        return p._denormalize_action(r.action)[0]

    def motor(self, obs):
        k = self.t - self.t_delib
        prop = self.p._normalize_state(np.asarray(obs["joint_action"]["vector"], dtype=np.float32)).to(self.p.device)
        a = joint.fast_action(self.m, self.prev.cache, self.prev._inp.prompt_embs, prop, num_steps=2,
                              generator=_gen(self.m, 12, self.t), stale_k=min(k, 32))
        self.stats["motor"] += 1
        return self.p._denormalize_action(a)[0]

    def reflex(self, obs, task_env):
        p, m = self.p, self.m
        from .policy_io import Adapter
        ad = Adapter(p, os.environ.get("TASK", ""))
        inp = ad.inputs(obs, self._instr(task_env))
        st = joint.prepare_static(m, inp, ad.gen_len)
        k = self.t - self.t_delib
        vid = shift_video(self.prev.latents_video, m.n_cameras, st.Tlpv, k)
        a = joint.reflex_action(m, st, inp, vid, self.prev.canvas, self.prev.final_inputs["sigma_v"],
                                num_steps=self.cfg["reflex_steps"], generator=_gen(m, 13, self.t))
        self.stats["reflex"] += 1
        return p._denormalize_action(a)[0]

    def _start(self, task_env, obs, chunk, key):
        """Book one computation on the (serial) GPU: the plan from obs(t) is adopted at t+d from its index d."""
        d = self.steps_for(key)
        chunk = [(a, key) for a in chunk]
        if not self.queue:                                               # nothing to run meanwhile
            self._hold(task_env, obs, d)
            self.queue = deque(chunk)
        elif d == 0:
            self.queue = deque(chunk[self.cfg["sync_offset"]:])        # D3 diagnostic: skip the chunk's first steps
        else:
            self.pending = (self.t + d, chunk, d)

    def step_reflex(self, task_env, obs):
        if self.pending is None:                                         # the GPU is free
            if self.prev is None or (self.t - self.t_delib) >= self.cfg["slow_every"]:
                self.t_reflex = self.t
                self._start(task_env, obs, self.deliberate(obs, task_env), "cold")
            elif (self.t - self.t_reflex) >= self.cfg["reflex_every"]:
                self.t_reflex = self.t
                self._start(task_env, obs, self.reflex(obs, task_env), "reflex")
        if not self.queue:
            self._hold(task_env, obs, 1)
            return
        a, self.src = self.queue.popleft()
        self._exec(task_env, a)

    def surprised(self, obs) -> bool:
        if self.surprise is None or self.prev is None:
            return False
        from .train_c import surprise_features
        net, mu, sd = self.surprise
        inp = self.p._build_obs_video(obs)
        lat = joint.encode_window(self.m, inp)
        k = self.t - self.t_delib
        rec = {"latents_video_out": self.prev.latents_video.detach().cpu(), "action": self.prev.action}
        plan = self.p._denormalize_action(self.prev.action)[0]
        f = surprise_features(rec, lat.detach().cpu(), obs["joint_action"]["vector"], k, plan)
        with torch.no_grad():
            s = float(torch.sigmoid(net(((f - mu) / sd).unsqueeze(0))))
        return s > self.cfg["surprise_thr"]

    # ---- the RoboTwin step --------------------------------------------------------------------------------------
    def _exec(self, task_env, action, held=False):
        if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success:
            return
        if self.cfg.get("trace_path"):    # per-step: source plan, and how far the commanded pose is from the current one
            q = self._q_now
            self.trace.append({"t": self.t, "src": "hold" if held else self.src,
                               "dq": float(np.abs(np.asarray(action, dtype=np.float32)[:-1] - q[:-1]).max())})
        task_env.take_action(np.asarray(action, dtype=np.float32), action_type="qpos")
        self.t += 1

    def _hold(self, task_env, obs, n):
        q = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
        for _ in range(n):
            if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success:
                return
            self._exec(task_env, q, held=True)
            self.stats["held"] += 1

    def step(self, task_env, obs):
        self.env = task_env
        self._q_now = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
        if self.pending is not None and self.t >= self.pending[0]:       # an async plan has arrived
            _, chunk, idx = self.pending
            self.queue = deque(chunk[idx:])
            self.pending = None
        if self.mode == "reflex":
            return self.step_reflex(task_env, obs)
        if self.mode == "released":
            if not self.queue:
                chunk = self.deliberate(obs, task_env)
                self._hold(task_env, obs, self.steps_for("released"))   # blocking: the arm waits for the model
                self.queue = deque(chunk[: self.cfg.get("released_replan", self.p.replan_steps)])
                if task_env.take_action_cnt >= task_env.step_lim:
                    return
            self._exec(task_env, self.queue.popleft())
            return
        due = (self.t - self.t_delib) >= self.cfg["replan"] or self.prev is None
        if self.mode == "brain" and not due and (self.t - self.t_delib) >= self.cfg["min_gap"] and self.surprised(obs):
            due = True
            self.stats["surprise_triggers"] += 1
        if due and self.pending is None:
            chunk = self.deliberate(obs, task_env)
            # the queue keeps the whole remaining chunk, so the arm has a plan to run while a slow (cold) one computes
            d = self.steps_for("cold" if self.last_cold else "warm1")
            if not self.queue:                                           # nothing to run meanwhile (episode start)
                self._hold(task_env, obs, d)
                self.queue = deque(chunk)
            elif d == 0:
                self.queue = deque(chunk)
            else:
                self.pending = (self.t + d, chunk, d)
        if self.mode == "brain" and self.prev is not None and not due:
            a = self.motor(obs)
            d = self.steps_for("motor")
            self._exec(task_env, a[min(d, len(a) - 1)])
            if self.queue:
                self.queue.popleft()
            return
        if not self.queue:
            self._hold(task_env, obs, 1)
            return
        self._exec(task_env, self.queue.popleft())


def install(policy, cfg: Dict) -> Brainloop:
    bl = Brainloop(policy, cfg)
    policy._brainloop = bl
    orig_reset = policy.reset

    def step(task_env, observation):
        policy.step_count = bl.t
        bl.step(task_env, observation)

    def reset():
        bl.report()
        orig_reset()
        bl.reset()

    policy.step = step
    policy.reset = reset
    atexit.register(bl.report)
    print("BRAINLOOP_INSTALLED " + json.dumps(bl.cfg), flush=True)
    return bl
