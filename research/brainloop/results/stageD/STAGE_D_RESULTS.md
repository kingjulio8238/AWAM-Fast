# Stage D results: the fresh-vision reflex tier

Plan and pre-registered bars: `../../STAGE_D_PLAN.md`. Box: Vast RTX 5090 58063344 (KR), $0.534/h.

## D2: latency on the RTX 5090 (p95, ms)

| path | compute | + 30 Hz tick | reflex gate (<= 150 incl. tick) |
|---|---|---|---|
| **reflex, 4 action steps** (encode 25 + compiled joint forward 40 + compiled 4-step motor 17) | **82** | **116** | pass |
| reflex, 2 action steps | 74 | 107 | pass |
| reflex, eager, end to end | 171 | 204 | fail |
| cold-2 / 3 / 4 / 8 / 12 decode, compiled, end to end | 113 / 152 / 190 / 348 / 503 | | |

The compiled reflex number is a **sum of separately measured compiled pieces**, not one compiled pass timed end to
end. Fusing it is follow-up work.

## D3: simulator, 3 tasks x 8 paired episodes (seed 0; controls from Stage C)

| config (realtime) | latency | success | click / hammer / microwave | steps vs released sync | bar |
|---|---|---|---|---|---|
| **reflex** (slow cold-8 every 24 steps + reflex every 4) | 82 ms | **20/24** | 8/8, 7/8, 5/8 | **1.09x** (n=16) | **PASS** |
| reflex, 2 action steps | 74 ms | 9/16 | 7/8, 2/8, (stopped) | | worse; dropped |
| cold8async | 348 ms | 19/24 | 7/8, 8/8, 4/8 | 1.13x | |
| cold4async | 190 ms | 16/24 | 8/8, 6/8, 2/8 | 1.03x | |
| released (realtime) | 1293 ms | 19/24 | 7/8, 7/8, 5/8 | 2.72x | |
| released sync (reference) | 0 | 19/24 | 7/8, 6/8, 6/8 | 1.00x | |

**The pre-registered kill bar tripped anyway.**
- Reflex at zero latency (sync) stalls on some episodes. On click_alarmclock it got 0/5 (stopped early) against 7/8
  for released sync.
- Two fixes were tried, and both recovered only half the episodes, failing on the same seeds:
  - adopting fresh plans from index 3: 4/8;
  - re-planning every 8 steps instead of 4: 4/8.
- With latency modelled the same episodes all succeed (8/8). So plan adoption mid-chunk under latency matters, and
  the tier is not yet robust as a standalone policy. This is an open defect.
- A per-step trace (`trace_path` in the sim config) has been added to diagnose it.

## D4: confirmation, 5 tasks x 20 paired episodes

Pooled over seed 0 episodes 0-7 (Stage C/D3) + seed 1 episodes 0-11 for the first three tasks; seed 0 episodes 0-19
for the other two. Every config sees the same episodes.

| config | reaction latency p95 (+tick) | click | hammer | microwave | stack | stapler | **pooled** | steps vs ref (paired n) | bar: >= ref - 5 pts, <= 1.2x |
|---|---|---|---|---|---|---|---|---|---|
| released sync (reference, zero latency) | 0 | 19 | 17 | 14 | 14 | 18 | **82/100** | 1.00 | - |
| **cold8async** | 348 ms (381) | 19 | 18 | 11 | 18 | 20 | **86/100** | **1.07** (72) | **PASS** |
| reflex | 82 ms (116) | 20 | 18 | 11 | 10 | 16 | 75/100 | 1.20 (63) | FAIL (-7 pts) |

Reading:
- **cold-8 async passes.** It replans in 381 ms including the tick (the deliberation gate is <= 400 ms) and matches or
  beats the zero-latency policy: 86 vs 82, 1.07x the steps. The shipped policy under the same latency needs 2.72x the
  steps (Stage C).
- It probably beats the reference on stacking (18 vs 14) because it replans every 8 steps instead of every 24.
- **The reflex tier meets the 150 ms gate but fails success parity.**
  - On short tasks it is even or ahead (38/40 vs 36/40). Precise placement is where it loses: stacking 10/20 against
    18/20 for cold-8 on the same episodes.
  - On microwave every realtime config loses 3 to the reference (11 vs 14), so that cost is shared, not specific to
    the reflex.
- n=100 gives a 95% interval of about +-9 points.
- The seed-0 cold8async rows on the first three tasks used the interpolated 335 ms latency; all others use the
  measured 348 ms.

## D5: probe of a faster slow tier for the reflex (stack, microwave; seed 0 x 8)

| reflex variant (realtime) | slow tier | GPU share left for the reflex | stack (8) | microwave |
|---|---|---|---|---|
| reflex (D4) | cold-8 every 24 | about 54% | 5/8 | 5/8 |
| reflex_c4s12 | cold-4 (190 ms) every 12 | about 50% | **7/8** | **0/6** (stopped; every episode hit the 1500-step limit) |
| reflex_c8s16 | cold-8 (348 ms) every 16 | about 31% | 3/8 | not run |

- A fresher slow plan helps stacking only when it is cheap enough to leave the reflex GPU time.
- But a 4-step slow tier is too weak for the long microwave task, consistent with cold4async 2/8 there in Stage C.
- No variant fixes both tasks. The stage stopped here, before the box's cap.

## Spend

Stage D: $3.92 of the $4.50 cap (credit $11.48 -> $7.57). Box destroyed; 0 instances.
