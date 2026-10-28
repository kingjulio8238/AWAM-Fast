# Brainloop: zero-training experiments E1–E5 on the released AW-0 RoboTwin WAM

Question: can AW-0 be run like a nervous system instead of a single synchronous loop? That means a fast motor loop
reading cached plan/foresight, a persistent belief instead of re-denoising from noise, and deliberation only when
surprised. All five experiments use the **released `wam.pt`** with no training. Inputs are the public RoboTwin 2.0
demonstrations (`TianxingChen/RoboTwin2.0`, `aloha-agilex_clean_50`), fed through the eval's own preprocessing.

## Experiments and pre-registered bars

Each verdict is `PASS`, `FAIL` or `INSUFFICIENT DATA`; the last applies below 30 comparisons, or below 10 event
positives for E3. Errors are the median absolute joint error over the first 8 actions, arm joints only, in rad,
measured against a fresh 20-step decode at the same moment. The **seed floor** is two fresh decodes with different
noise; it is the noise level the model itself has.

| # | Question | Method | Pass bar |
|---|---|---|---|
| E1 | Can the action expert run alone on a stale code/video K/V cache? | Cache from the decode at time *a*; at *a+k* run 1/2/4 action-only steps with fresh proprio. Compare to the fresh decode at *a+k*, and to open-loop playback of the old chunk. | 2-step motor error ≤ 1.5× seed floor for k ≤ 8, and below open-loop playback |
| E2 | Does warm-starting from the last prediction match a cold decode? | At *a+k*, start the 20-step schedule at step 20−N from anchor *a*'s video (shifted k steps), action (shifted) and code (50% lowest-confidence tokens re-masked). | 2-step warm error ≤ 1.5× floor, and below a cold 2-step decode |
| E3 | Does AW-0's own prediction error spike at contact events? | Foresight error (predicted vs actual future VAE latents and pixels), seed disagreement, code confidence and code change, scored against gripper open/close events and against E1's open-loop error. | Some online signal reaches AUROC ≥ 0.75 for "event within 8 steps", or Spearman ≥ 0.4 with the open-loop error 8 steps later |
| E4 | Which pixels does the policy depend on? | Drop / downsample ×2 ×4 / 2×3 occlusion grid per view; the drop case is also done per pathway (VAE → video expert, ViT → code expert). Fixed noise. | Descriptive: sensitivity = error / seed floor |
| E5 | How fast is each tier on an RTX 5090? | Stage timings: VAE, ViT, one joint step, full decode at 20/6/4 steps (eager and the repo's compiled `fast_joint`), and the motor tier eager and CUDA-graphed. | Motor p95 + one 30 Hz tick ≤ 150 ms; 4-step decode p95 ≤ 400 ms |

## Validity checks that fail the run loudly (`run.py proof`)

- **Checkpoint load:** zero rejected tensors. Missing tensors are allowed only for the frozen VAE and the ViT tower.
- **Model config:** joint code objective, `CodeVisionPrefix`, 3 cameras, 14-D actions, 128-token canvas,
  `parity_crop` preprocessing.
- **Decode parity:** the lab decode equals the policy's own `_infer_action_chunk` under the same seed, to ≤ 1e-3 rad.
- **Motor-tier exactness:** the action-only pass with per-step caches reproduces the joint action. This is reported
  as a ratio to the seed floor.
- **Model sanity:** the model's error vs the demonstration must be below 0.8× the error of holding the current pose.
- **Image channels:** HDF5 frames must decode as RGB, checked against the episode mp4.
- **Release files:** sha256 must match `manifest.json`.

CPU equivalence tests run on a tiny model built from the real classes:
`PYTHONPATH=src:. python -m pytest research/brainloop/tests -q`. They check lab decode == model decode
bit-for-bit, action-only == joint with per-step caches, the warm-start semantics, and that the cache contents
actually matter.

## Running it

```bash
# Vast (RTX 5090, the paper's GPU); the box kills itself after CAP hours
research/brainloop/env/vast.sh offers
research/brainloop/env/vast.sh up <offer_id> <cap_hours>
research/brainloop/env/vast.sh push <instance_id>
research/brainloop/env/vast.sh run  <instance_id> "bash research/brainloop/env/bootstrap.sh" boot
research/brainloop/env/vast.sh run  <instance_id> "python -u research/brainloop/run.py all --assets /workspace/assets --out /workspace/out"
research/brainloop/env/vast.sh tail <instance_id> lab ; research/brainloop/env/vast.sh pull <instance_id> ; research/brainloop/env/vast.sh down <instance_id>
```

```bash
# Modal (E1–E4 only; no 5090 there)
MODAL_PROFILE=<p> modal run research/brainloop/env/modal_app.py::fetch
MODAL_PROFILE=<p> BRAINLOOP_TIMEOUT_S=<1.5x projection> modal run research/brainloop/env/modal_app.py --cmd sweep
```

- **Outputs:** everything lands in `--out`: `proof.json`, `sweep/*.jsonl`, `e4/*.jsonl`, `e5_latency.json`, then
  `summary.json`, `report.md` and `report.pdf` from `run.py analyze`.
- **Stage marker:** every stage prints `BRAINLOOP <stage> DONE|FAILED`.
- **Monitoring:** `board.py` renders the status board from `progress.json`, the build log and the Vast instance.

## Files

| file | role |
|---|---|
| `lab/joint.py` | the default-path joint decode with K/V capture, warm start, per-pathway pixels; the motor tier |
| `lab/kvmot.py` | MoT forward with K/V capture; action-only forward against a cache |
| `lab/policy_io.py` | builds the released policy through `deploy_policy.get_model`, as the RoboTwin eval does; validation |
| `lab/demos.py` | RoboTwin HDF5 episodes as simulator observations; gripper events |
| `lab/experiments.py` | E1–E3 sweep, E4, E5 |
| `lab/analysis.py` | verdicts, `report.md`, `report.pdf` |
| `lab/tiny.py`, `tests/` | CPU stand-in model and equivalence tests |
| `env/` | Dockerfile, bootstrap, `check_env.py`, Vast and Modal launchers |
