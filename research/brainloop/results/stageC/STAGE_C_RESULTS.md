# Stage C results (C1 smoke + first simulator read)

Box: Vast RTX 5090 57996977 (Slovakia), destroyed after the run. Spend: $2.89 of the $3.20 Stage C cap (credit
$4.41 -> $1.52). Plan: `../../STAGE_C_PLAN.md`. Raw data: `c1/*/results.jsonl`, `eval_logs/`, `logs/`.

## Headline

The first closed-loop numbers changed the plan:

1. **Warm-1 replans fail in the simulator.** Offline, a warm-1 replan was within 1.46x of the seed floor of a fresh
   decode. In RoboTwin, chains of warm-1 replans stall until the step limit. On `click_alarmclock`: 1/2 successes,
   359-400 steps, against 62 steps for the released policy. This held at every cadence and canvas setting tried.
   - Cause: a warm start at step 19 of 20 adds almost no noise, so a warm-1 plan is roughly the previous plan shifted
     forward. Its tail is padded with the last action, so after about 4 hops the arm runs mostly padding, which is a
     hold.
   - Offline, the warm-1 error was measured over 8 steps. Playback is good enough over that horizon, so the metric
     could not see the stall.
2. **What works: async execution of compiled cold decodes with fewer steps** (`cold4async`, `cold8async`). The arm
   keeps running the current chunk while the next plan computes, and adopts the new plan at the index matching the
   time elapsed since its observation.
   - The real-time latency model is measured 5090 numbers: released eager 1293 ms, cold-4 181 ms; cold-8 is
     interpolated at 335 ms.
   - In real-time mode the released policy holds the arm still for 39 control steps per replan. On short tasks it
     takes about 3x the synchronous step count.
   - `cold8async` in real time matches the step count of the zero-latency released policy (see the table).
3. T1 (the stale-cache motor tier) failed in every variant offline. T2 (the learned time-shift) passed offline (0.0065
   vs 0.0079 rad, 2/2 tasks), but it only improves warm chains, which (1) rules out. C2-C4 as scoped should not run.

## C5 first read (RoboTwin, demo_clean, seed 0, same episodes for every config)

Steps are 30 Hz control steps, including steps the arm is held while the model thinks. "succ." means the mean over
successful episodes only.

| task | config | mode | success | steps (succ.) | steps (all) | held/ep |
|---|---|---|---|---|---|---|
| beat_block_hammer | released | sync | 6/8 | 160 | 220 | 0 |
| beat_block_hammer | cold4async | realtime | 6/8 | 119 | 190 | 6 |
| beat_block_hammer | cold8async | realtime | 8/8 | 124 | 124 | 11 |
| beat_block_hammer | released | realtime | 7/8 | 321 | 331 | 212 |
| click_alarmclock | released | sync | 7/8 | 110 | 146 | 0 |
| click_alarmclock | cold4async | realtime | 8/8 | 80 | 80 | 6 |
| click_alarmclock | cold8async | realtime | 7/8 | 77 | 118 | 11 |
| click_alarmclock | released | realtime | 7/8 | 191 | 217 | 139 |
| open_microwave | released | sync | 6/8 | 405 | 679 | 0 |
| open_microwave | cold4async | realtime | 2/8 | 350 | 1212 | 6 |
| open_microwave | cold8async | realtime | 4/8 | 450 | 975 | 11 |
| open_microwave | released | realtime | 5/8 | 1086 | 1241 | 775 |
| stack_blocks_two | released | realtime | 3/8 | 772 | 793 | 502 |

| config | mode | success (pooled) | steps (succ., pooled) |
|---|---|---|---|
| released | sync | 19/24 | 219 |
| cold4async | realtime | 16/24 | 128 |
| cold8async | realtime | 19/24 | 175 |
| released | realtime | 22/32 | 503 |

Pooled over the three tasks every config ran (click_alarmclock, beat_block_hammer, open_microwave; 24 episodes
each). The step ratio is the median over paired episodes (same seed) where both configs succeeded, against released
sync, which is the zero-latency reference.

| config | latency model | success | steps vs zero-latency released |
|---|---|---|---|
| released, sync | 0 ms | 19/24 | 1.00 (reference) |
| released, realtime | 1293 ms, blocking | 19/24 | 2.72x (n=16) |
| **cold8async, realtime** | 335 ms, async | **19/24** | **1.13x** (n=16) |
| cold4async, realtime | 181 ms, async | 16/24 | 1.03x (n=13) |

Reading:
- Under real latency the shipped policy needs about 2.7x the steps of the same policy at zero latency, almost all of
  it spent holding still.
- cold8async keeps the success rate and finishes within about 13% of zero latency. Against released in real time
  that is 0.40-0.43x the steps, paired per episode.
- cold-4 is as fast but loses on the long task (open_microwave: 2/8, against 6/8 for released sync).
- Caveats:
  - n=8 per cell. The cold8async 4/8 on open_microwave against 6/8 for released sync may be a real quality cost on
    long tasks.
  - The cold-8 latency is interpolated, not measured.
  - Replans happen every 8 steps for the async configs and every 24 for released.
  - stack_blocks_two has only the released real-time row (3/8): that run was stopped to stay under the $3.00 cap.

## C1 smoke diagnostics (click_alarmclock, 2 episodes each)

| config | mode | success | steps | note |
|---|---|---|---|---|
| released | sync | 2/2 | 61, 63 | |
| released | realtime | 2/2 | 180, 243 | 117-156 steps held |
| warm1 (replan 8) | sync | 1/2 | 400, 359 | stalls |
| warm1, replan 24 | sync | 0/2 | 400, 400 | |
| warm1, whole canvas re-masked | sync | 1/2 | 400, 352 | the code canvas is not the cause |
| warm1, cold refresh every 3rd plan | sync | 2/2 | 64, 80 | the cold plan never runs out |
| warm1, cold refresh every 3rd plan | realtime | 0/2 | 400, 400 | |
| cold-20 every 8 steps | sync | 2/2 | 63, 80 | cadence alone is fine |
| cold-20 async | realtime | 2/2 | 79, 81 | |

On `handover_block` every config fails (0/3), including the unmodified released policy without the wrapper. The
wrapper is therefore faithful there; the task is simply hard for this checkpoint on these seeds.

## Offline (C1, toy scale)

- T1 motor tier at k=8 errs 0.026-0.028 rad, against 0.012 for open-loop playback and a seed floor of 0.0055.
  Variants tried: full, light and stale-only fine-tunes, and zero-training warm-motor. All fail.
- T2 learned shift: chained warm-1 error 0.0065 vs 0.0079 for `shift_video`. Forward pass 0.44 ms.
- Motor step (eager, 2 steps): 24 ms on the C1 box.

## Fixes made during C1

- The simulator ran out of GPU memory because the 7B instruction encoder sat on the GPU. With
  `AGENTICWAM_TEXT_DEVICE=cpu` it is encoded on the CPU, once per episode.
- `warp-lang==1.7.2` is pinned, because cuRobo 0.7.8 needs `wp.torch`. ffmpeg is installed.
- The wrapper now prints `BRAINLOOP_INSTALLED` and one `BRAINLOOP_EPISODE` line per episode, and `run_c5.py` fails
  the run if the wrapper is not active. The first smoke had no such check.
- The async queue keeps the whole remaining chunk, so the arm has a plan to run while a slow plan computes.
