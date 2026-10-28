# Stage B results: 2026-10-27, Vast RTX 5090 (KR instance 57973893, destroyed)

Spend $1.47 total against a $2.40 cap. That includes $0.24 on an abandoned Taiwan host whose Hugging Face downloads
stalled. Data: `out/` (summary.json, report.md/pdf, sweep/, chain/, e4/, e5b_*.json). Bars are from `../../STAGE_B.md`.

Spec: 6 tasks x 1 episode, 387 anchors (decode points). Seed floor (two fresh 20-step decodes with different noise,
h8 arm error) is 0.0055 rad.

## Verdicts

| # | bar | result | verdict |
|---|---|---|---|
| B0 | setup <= 25 min; total <= $2.40 | setup about 45 min (two HF download stalls, fixed: plain HTTP + aria2); $1.47 | spend PASS, setup FAIL |
| B1 | E2 per task: warm-2 <= 1.5x floor and < cold-2 on >= 5/6 tasks | pooled 0.0081 = 1.46x floor (cold-2 0.0112); per task 3/6 (click, hammer, microwave pass; handover 1.66x, mug 1.6x, stack 1.77x) | FAIL (pooled PASS) |
| B2 | E3: AUROC >= 0.75 or Spearman >= 0.4 (30 positives) | best AUROC 0.52; best rho with open-loop error 0.28 | FAIL |
| B3 | chained warm-2 at hop 10 <= 1.5x floor, drift rho <= 0.3 | hop 10 is 0.0121 (2.2x); drift rho 0.08 (none); warm-1 about 0.010 flat vs cold-1 about 0.050 | FAIL (no drift) |
| B4 | warm-2 replan end to end p95 <= 400 ms | compiled 107 ms; eager 154 ms | PASS |
| B5 | compiled joint step p50 <= 50 ms | 37.8 ms, 11.9 GB peak | PASS |
| B6 | foveation justified if >= 50% of head-cam blocks are about 1x floor | 4/6 head blocks <= 1.1x; head x4 downsample 1.14x; wrist views nearly irrelevant to action | PASS (descriptive) |
| B7 | ActionDiT.freqs follows the module's device, dtype untouched | CUDA graphs now captured; repo fast_joint compiles (11.9 GB) | PASS |

## Latency, RTX 5090, bf16, end to end per replan (encode + decode), p95

| | eager | compiled (CUDA graphs) |
|---|---|---|
| cold 20 steps (released setting) | 1293 ms | 799 ms (repo fast_joint: 794 ms) |
| cold 4 steps (paper's setting; paper reports 0.23 s) | 280 ms | 181 ms (repo: 181 ms) |
| warm 2 steps | 154 ms | 107 ms |
| warm 1 step | 91 ms | 66 ms |
| motor tier, 2 action steps (Stage A) | 98 ms | 9.1 ms |

The compiled warm-2 decode differs from eager by 0.0031 rad, about 0.6x the seed floor.

## E4: what the action depends on (sensitivity = perturbed error / seed floor)

- Head camera through the VAE (video) pathway dominates. Dropping it gives 7.4x; dropping it on the VAE path only
  gives 7.1x; dropping it on the ViT (code) path only gives 0.46x.
- Wrist cameras barely move the action on these demos: dropping either is about 1.1-1.2x.
- Resolution: head x4 downsample is 1.14x and head x2 is 0.82x; wrist x4 is 0.4-0.7x.
- Head occlusion matters only in the top-center (1.42x) and center (1.29x) blocks.
- Caveat: 6 anchors, in-distribution training demos.

## E1 (unchanged from Stage A, now at scale)

At 8 steps of staleness the 2-step motor tier errs 0.0193 rad vs 0.0121 for open-loop playback. It needs
staleness-aware training.

## Repo fixes and findings

- `src/agenticwam/models/experts/action_dit.py`: the RoPE table follows the module's device. This enables CUDA graphs.
- The code expert's RoPE is a complex buffer that `.to(dtype)` flattens to its real part, at build and at deploy.
  It is consistent between training and deployment, so this is a retraining lever, not a deployment bug.
- Hugging Face xet / hf_transfer downloads can stall on Vast; plain HTTP + aria2 runs at 100-270 MB/s.
