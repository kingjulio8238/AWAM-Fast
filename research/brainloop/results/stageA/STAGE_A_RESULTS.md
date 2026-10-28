# Stage A results: 2026-10-27, Vast RTX 5090 (instance 57954094, destroyed)

Raw data: `out/` (proof.json, sweep/*.jsonl, e5_latency_*.json, diag_motor_bf16.json, summary.json, report.md/pdf).
Logs: `logs/`. Success criteria were fixed before launch in `../../STAGE_A.md`.

## Criteria

| id | criterion | result | verdict |
|---|---|---|---|
| A1 | environment | RTX 5090, sm_120 kernels present, SDPA and matmul OK, 7/7 CPU tests on box | PASS |
| A2 | assets | wam.pt, dataset_stats.json and task_skill_prompt.json sha256 = manifest | PASS |
| A3 | released policy | 0 rejected tensors; 505 missing, all VAE / frozen ViT tower; joint, CodeVisionPrefix, 3 cams, 14-D, canvas 128, parity_crop | PASS |
| A4 | lab measures the model | lab decode vs policy `_infer_action_chunk`: 0.0 rad; same-seed re-run: 0.0 | PASS |
| A5 | motor shortcut exact | per-step-cache action-only vs joint: **0.0000 rad** with the joint's attention backend; 0.0012–0.0015 rad (0.22–0.30x seed floor) with the fused backend | PASS (*) |
| A6 | model works | h8 error 0.0072 rad vs hold-still 0.0375 (0.19x; bar < 0.8) | PASS |
| A7 | mini sweep usable | 2/2 episodes, 52 anchors; E1 n = 48 per k <= 8; E2 warm-2 n = 48 | PASS |
| A8 | latency on the 5090 | E5 bf16 + fp32 eager complete; CUDA-graphed motor tier measured; repo `fast_joint` compile OOMs on 32 GB (recorded) | PASS |
| A9 | cost control | **$2.09 actual vs $1.35 cap**: GPU $0.61 + storage $0.07 + **download $1.38 (35 GB at $0.039/GB)** + upload $0.04; results pulled, box destroyed | **FAIL** |
| A10 | Stage B affordable | ~2.75 h on a $0/GB host about $1.8 (1 episode for long tasks), <= $2.40 | PASS |

(*) The proof's automatic check reported "8.4x". It compared a max-abs error against a mean seed floor, which is my
metric bug; `diag_motor.py` re-measured it like-for-like.

A9 root cause: the cap and the projection counted $/h only. Vast bills download bandwidth per GB on this host.
Fixed: `env/vast.sh offers` now ranks by hours x $/h + GB x $/GB, and the cost watcher adds bandwidth.

## Science (mini sweep: click_alarmclock ep0 + stack_blocks_three ep0, 52 anchors; provisional)

Seed floor (two fresh 20-step decodes, different noise): 0.0055 rad h8. The decoded code is identical across seeds.

**E2 warm start: PASS.**

| decode | h8 arm error vs fresh 20-step (rad) |
|---|---|
| warm, 1 step | 0.0080 |
| warm, 2 steps | 0.0079 (1.43x floor) |
| warm, 8 steps | 0.0066 |
| cold, 1 step | 0.0491 |
| cold, 2 steps | 0.0118 |
| cold, 8 steps | 0.0078 |

Code text similarity to the fresh plan is 0.58 (warm) vs 0.13 (cold 2-step). The warm video is farther from the fresh
video (rel-MSE 0.14 vs 0.035) while the actions stay on target.

**E1 stale-cache motor tier: FAIL as a zero-training retrofit.**

| method | k = 0 | k = 4 | k = 8 | k = 16 | k = 24 |
|---|---|---|---|---|---|
| fast2 (h8 rad) | 0.0105 | 0.0177 | 0.0245 | 0.0299 | 0.0376 |
| open-loop playback | — | 0.0107 | 0.0107 | 0.0117 | 0.0130 |

The motor tier drifts with staleness even though it is exact with matched caches (A5). The terminal-step cache plus
few action steps is out of distribution: it needs staleness-offset fine-tuning.

**E3 surprise: insufficient data** (6 event positives, bar 10). Best AUROC for "event within 8 steps" is 0.62
(latent foresight error from 16 steps earlier). Pixel foresight error from 16 steps earlier correlates rho = 0.61
with the motor tier's error at k = 8.

**E5 latency, RTX 5090, p50.**

| stage | bf16 | fp32 |
|---|---|---|
| one joint MoT step (630 video + 536 code + 32 action tokens) | 151 ms | 141 ms |
| decode, 20 steps (released setting) | 3.05 s | 2.93 s |
| decode, 6 steps | 946 ms | 920 ms |
| decode, 4 steps | 648 ms | 632 ms |
| VAE encode (3 views) | 10 ms | 18 ms |
| ViT prefix (3 views) | 25 ms | 29 ms |
| motor tier, eager, 1 / 2 / 4 steps | 49 / 98 / 195 ms | 42 / 84 / 168 ms |
| **motor tier, CUDA-graphed, 1 / 2 / 4 steps** | **4.9 / 9.0 / 17.3 ms** (p95 5.0 / 9.1 / 17.4) | — |

- bf16 is no faster than fp32, so the decode is overhead-bound, not compute-bound.
- Gate: motor p95 9.1 ms plus one 30 Hz tick gives 42 ms (bar 150 ms): PASS. Eager 4-step deliberation at 648 ms
  fails the 400 ms bar; a compiled joint step is still unmeasured.

Repo bugs found:
- `ActionDiT.freqs` is a CPU tensor, so `fast_joint_infer`'s CUDA graphs are skipped ("skipping cudagraphs due to
  cpu device"). Moving it to the GPU is the fix that enabled the 10x motor speedup.
- `fast_joint_infer`'s whole-step compile OOMs on a 32 GB card even after that fix.
