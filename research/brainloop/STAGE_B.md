# Stage B: plan and pre-registered bars (approved for local prep 2026-10-27; rental needs a second approval)

Run spec: click_alarmclock:2, beat_block_hammer:2, stack_blocks_three:1, handover_block:1, open_microwave:1,
hanging_mug:1, about 510 anchors. Host with $0/GB download. The cap counts bandwidth.

| # | item | bar |
|---|---|---|
| B0 | rent, bootstrap, monitors (chain + cost incl. bandwidth) | setup <= 25 min; total <= $2.40 incl. bandwidth |
| B1 | full sweep | E2 holds per task: warm-2 <= 1.5x seed floor and < cold-2 on >= 5/6 tasks |
| B2 | E3 at scale | >= 10 event positives; AUROC >= 0.75 (event within 8) or Spearman >= 0.4 with open-loop error |
| B3 | chained warm starts: 10 replans, 8 steps apart, each warm from the previous | 2-step warm error at hop 10 <= 1.5x floor; no drift (Spearman of error vs hop <= 0.3) |
| B4 | replan latency end to end (encode + decode), warm 1/2-step | warm-2 replan p95 <= 400 ms |
| B5 | compiled (CUDA-graphed) joint step that fits 32 GB | joint step p50 <= 50 ms |
| B6 | E4 occlusion, 6 tasks x 2-3 anchors | descriptive map; foveation justified if >= 50% of head-cam blocks are within ~1x floor |
| B7 | ActionDiT.freqs follows the module's device (repo fix; dtype untouched) | CUDA graphs no longer skipped; CPU tests pass |

Stage order on the box: proof -> sweep -> chain -> e5b core -> e5b fastjoint -> e4 -> analyze. E4 runs last so it
is the cheapest to drop if the cap approaches.
