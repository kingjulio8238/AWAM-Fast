# Stage D plan: fresh-vision reflex tier (B), then the confirmation run (A)

## Goals (confirmed with the user on 2026-10-28)

| goal | bar |
|---|---|
| Reflex: observation -> corrective action | <= 150 ms p95 **including** one 30 Hz tick, so about 117 ms of compute on the RTX 5090 |
| Deliberation: full replan | <= 400 ms p95 (already met: cold-4 181 ms, cold-8 about 335 ms) |
| The win | a new two-rate architecture (fast reflex plus slow deliberation) that meets the reflex bar, not just a faster policy |
| Quality | RoboTwin success at parity with the released policy at zero latency (released sync): 5 tasks x 20 paired episodes |

## Why a fresh-vision reflex

- Stage C: warm-1 replans (66 ms) stall in closed loop. Their action starts from the shifted old plan at near-zero
  noise, so each replan is playback.
- The T1 motor tier (5 ms) fails because it sees only fresh proprio, never fresh vision.
- The reflex tier fixes both:
  - **Fresh vision:** the new camera frames are VAE- and ViT-encoded, and the code/video K/V cache is rebuilt in one
    joint forward. Video = the last slow plan's prediction, shifted to now, with the fresh frames as conditioning.
    Code = the last slow plan's canvas.
  - **Fresh action:** the action is denoised **from noise** in N action-only steps against that cache, with fresh
    proprio.
- The slow tier (cold-8) still runs every 24 steps. It is what moves the code (sub-task) and the video prediction
  forward.
- Zero training. Estimated compute with the compiled pieces:

  | piece | ms |
  |---|---|
  | VAE + ViT | about 20 |
  | joint forward | about 38 |
  | 4 action steps | about 20 |
  | **total** | **about 78**, so about 111 ms with the tick |

  D2 measures this; the eager version will be slower.
- The latency model in the sim is a single serial GPU. One computation is in flight at a time, a slow decode blocks
  reflex ticks, and every plan is adopted at the index matching its own latency.

## Stages

| stage | what | $ |
|---|---|---|
| D0 | Local build: reflex tier in `lab/joint.py` + `lab/sim_policy.py` (serial-GPU scheduler), latency harness, CPU tests | $0 |
| D1 | Box: one Vast 5090, fetch + bootstrap + sim_setup (C1 measured about 8 min, about 50 GB) | about $0.25 |
| D2 | Latency on the 5090: reflex pieces eager and compiled; cold-2/3/8 decodes measured (cold-8 is interpolated today) | about $0.12 |
| D3 (B) | Sim: `reflex` realtime + `reflex` sync, 3 tasks x 8 episodes, seed 0, paired with the Stage C controls | about $0.55 (+$0.27 for an N=2 variant) |
| D4 (A) | Confirmation on the winner: 5 tasks (click_alarmclock, beat_block_hammer, open_microwave, stack_blocks_two, press_stapler) x 20 episodes, seed-0 episodes reused; released sync + winner realtime (+ cold8async if the winner is the reflex) | $1.45-2.15 |
| **total** | | **$2.40-3.35**; server-side cap $4.00, so it fits $1.52 + a $5 top-up |

## Pre-registered bars

- **B passes** when all three hold:
  - reflex e2e, measured compute plus one tick, is <= 150 ms p95;
  - realtime success >= 18/24 pooled (released sync: 19/24);
  - median paired steps <= 1.15x released sync.
- **B kill:** reflex **sync** succeeds 2 or more episodes per task below released sync. That means the tier is broken
  as a policy, independent of latency. Skip to A with cold-8 and scope step distillation instead.
- **A passes** when:
  - pooled success over 100 episodes is >= released sync minus 5 points;
  - the median paired step ratio is <= 1.2x.
  - Note: the 95% interval at n=100 is about +-9 points, so A shows non-inferiority at that resolution, no finer.

## Order

D0 -> D1 -> D2 -> D3. D4 uses the D3 verdict to pick the configs.
