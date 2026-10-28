# Stage C plan: first training runs + simulator validation (draft for approval, 2026-10-27)

Goal: turn the Stage A/B inference wins into a human-speed policy that keeps task success.
1. Make the 5 ms motor tier usable on stale context (E1 failed zero-shot).
2. Close the warm-start gap on long tasks (B1/B3: ~1.6-2x seed floor).
3. Get a working surprise gate (E3 failed zero-shot).
4. Prove it in the RoboTwin simulator, including real-time (latency-injected) execution.

Everything trains on top of the released `wam.pt`. Video and code experts stay frozen; only small new parts or the
0.72B action expert train. Data is the public RoboTwin 2.0 demos.

## C0: local build and CPU tests ($0)
- [ ] Teacher-data generator: for anchors on demo episodes, store the decoded code canvas, the final video latents
      (about 240 KB), and the fresh action chunk. Enough to rebuild the exact terminal K/V cache and train against
      the model's own outputs without storing caches (255 MB each).
- [ ] T1 trainer, staleness-aware motor tier.
  - Rebuild the cache from anchor a (frozen, no_grad), feed proprio at a+k plus a learned staleness embedding of k.
  - Supervise the action at a+k with both GT demo actions and the fresh-decode teacher.
  - k = 0 is in the mix, so the default path is preserved.
  - Full fine-tune of the action expert, or LoRA if memory is tight on 32 GB.
- [ ] T2 learned time-shift for warm starts: a small net (about 10-50M params) that advances the previous final
      video/action latents by k steps. Target: the fresh decode at a+k. Also a training-free linear-extrapolation
      baseline.
- [ ] T3 surprise head: a small MLP on cached features. Label: open-loop / motor error above threshold at a+k.
- [ ] Offline eval: reuse E1 / E2 / B3 on 10 held-out demos per task, never seen by the fine-tune.
- [ ] RoboTwin simulator image: RoboTwin at the pinned commit, SAPIEN + Vulkan, cuRobo, assets (objects 3.7 GB,
      embodiments 0.2 GB, background textures 11 GB). Plus a latency-injection wrapper: the sim keeps executing the
      previous chunk for (measured latency x control rate) steps while the policy thinks.
- [ ] CPU tests: tiny-model training step reduces loss; staleness embedding at k = 0 reproduces the base model.

## C1: smoke tests (one rented GPU)
- [ ] Teacher-gen 100 anchors, train T1/T2/T3 200 steps each, offline-eval 20 anchors; check loss falls, memory fits,
      checkpoint round-trips.
- [ ] Sim: 1 task x 2 episodes, released config, synchronous; check success logging and frame parity with Stage A.
- [ ] Measure seconds per teacher anchor, per training step and per sim episode; reproject C2-C5.

## C2: teacher data
- [ ] 50 tasks x 20 demos, anchors every 8 frames, about 25k anchors at about 0.8 s (compiled 20-step decode).

## C3: training
- [ ] T1: about 10k steps, batch 8, k uniform in {0..24}.
- [ ] T2 and T3: small, about 1 GPU-h together.

## C4: offline evaluation (held-out demos)
- [ ] T1 bar: motor 2-step error at k <= 8 <= 1.5x seed floor and below open-loop playback (E1 rerun).
- [ ] T2 bar: chained warm-1 at hop 10 <= 1.5x floor on >= 5/6 Stage B tasks (B1/B3 rerun).
- [ ] T3 bar: AUROC >= 0.75 for "motor error > threshold" (E3 rerun).
- [ ] Latency of each new part on the 5090 (E5 rerun): motor tier stays <= 10 ms, warm-1 replan stays <= 70 ms.

## C5: simulator validation (RoboTwin, the first task-success numbers)
- [ ] Configs:
  - (a) released 20-step baseline;
  - (b) warm-1 compiled replans;
  - (c) (b) + trained motor tier at 30 Hz + surprise-gated replans.
- [ ] Modes: synchronous, and latency-injected at each config's measured 5090 latency.
- [ ] Subset: 5 tasks x clean x 20 episodes x 3 configs x 2 modes = 600 episodes.
- [ ] Bar: (b)/(c) synchronous success within 5 points of (a); (b)/(c) real-time success above (a) real-time.

## Cost estimates

Vast RTX 5090 at about $0.60/h plus bandwidth. All of these are estimates; C1 re-measures them before anything big
runs.

| stage | GPU-h | Vast $ | notes |
|---|---|---|---|
| C0 | 0 | $0 | local |
| C1 | about 1.0 | about $0.70 | includes about 45 GB of weights plus the sim image and assets |
| C2 | about 5.5 | about $3.40 | 25k anchors x 0.8 s, plus about 40% overhead |
| C3 | about 2.5 | about $1.60 | T1 about 1.5 h, T2+T3 about 1 h |
| C4 | about 0.7 | about $0.45 | |
| C5 | 4-8 | $2.50-5.00 | sim time per episode is the main unknown; measured in C1 |
| **total** | **about 14-18** | **about $9-11** | |

Two cheaper options:
- Modal credits reset Nov 1 (the free tier is per workspace per calendar month). C2-C4 numerics do not need a 5090,
  so they could run on an L40S or H100 within free credits. Only C5's real-time mode and the E5 latency need a 5090.
- C5 can shrink to 3 tasks (about $1.50-3) for a first read.
