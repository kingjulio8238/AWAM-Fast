# Stage A: definition of success (written before launch, 2026-10-27)

Stage A asks whether the measurement pipeline is trustworthy on the real model and whether Stage B is affordable. It
does NOT have to produce PASS verdicts on E1–E3: the mini sweep is too small for that, and any verdict it prints is
provisional.

## Stage A succeeds if all of these hold

| id | criterion | threshold |
|---|---|---|
| A1 | environment on the rented GPU | device is an RTX 5090; torch has sm_120 kernels; SDPA and a bf16 matmul run; 7/7 CPU tests pass on the box |
| A2 | assets | wam.pt, dataset_stats.json and task_skill_prompt.json match the release manifest's sha256 |
| A3 | the released policy is what we think it is | 0 rejected checkpoint tensors; missing ones only VAE / frozen ViT tower; joint code objective, CodeVisionPrefix, 3 cameras, 14-D, canvas 128, parity_crop |
| A4 | the lab measures the model, not something else | lab decode = policy's own `_infer_action_chunk` to <= 1e-3 rad (same seed); same-seed decodes identical |
| A5 | the motor-tier shortcut is exact where it should be | action-only with per-step caches reproduces the joint action to <= 0.5x the seed floor |
| A6 | the model works on these demos | h8 arm error vs demonstration < 0.8x the error of holding the current pose |
| A7 | the mini sweep produces usable data | both episodes (click_alarmclock ep0, stack_blocks_three ep0, <= 40 anchors each) finish without error; >= 30 E1 comparisons at each k <= 8 and >= 30 E2 warm-2 comparisons |
| A8 | latency is measured on the right hardware | E5 completes in bf16 and fp32 on the 5090; compiled paths either run or their error is recorded |
| A9 | cost control | spend <= $1.35 (2.5 h cap); results pulled locally; instance destroyed |
| A10 | Stage B is affordable | projected Stage B cost from measured timings <= $2.40 |

If any of A1–A6 fails, stop immediately, destroy the box, and report the cause.
