# Brainloop E1–E5 report

## Calibration

- anchors: 387 over 6 episodes; median decode 1.29 s
- action chunk aligned to GT offset 1; model h8 arm error 0.0092 rad vs hold-still 0.0118 (ratio 0.78)
- seed floor h8: arm 0.0055 rad, gripper 0.0007; code canvas agreement between seeds 1.000, text similarity 1.000

## E1 — staleness (median h8 arm error, rad, vs fresh decode)

| method | k=0 | k=4 | k=8 | k=12 | k=16 | k=20 | k=24 |
|---|---|---|---|---|---|---|---|
| open_loop | 0.0000 | 0.0113 | 0.0121 | 0.0121 | 0.0118 | 0.0123 | 0.0143 |
| fast1 | 0.0492 | 0.0529 | 0.0547 | 0.0576 | 0.0613 | 0.0617 | 0.0634 |
| fast2 | 0.0105 | 0.0151 | 0.0193 | 0.0230 | 0.0264 | 0.0293 | 0.0322 |
| fast4 | 0.0090 | 0.0138 | 0.0183 | 0.0216 | 0.0254 | 0.0279 | 0.0307 |

seed floor 0.0055. **Verdict: FAIL** (staleness tolerance k = None)

## E2 — warm start

| method | h8 arm | h24 arm | canvas agree | code text sim | video rel-MSE | n |
|---|---|---|---|---|---|---|
| cold1 | 0.0484 | 0.0498 | 0.352 | 0.099 | 0.0436 | 286 |
| cold2 | 0.0112 | 0.0121 | 0.453 | 0.157 | 0.0389 | 286 |
| cold4 | 0.0096 | 0.0100 | 0.613 | 0.255 | 0.0417 | 286 |
| cold8 | 0.0076 | 0.0081 | 0.730 | 0.481 | 0.0487 | 286 |
| warm1_rho0.5 | 0.0083 | 0.0113 | 0.809 | 0.447 | 0.1557 | 376 |
| warm2_rho0.0 | 0.0083 | 0.0112 | 0.871 | 0.660 | 0.1386 | 376 |
| warm2_rho0.5 | 0.0081 | 0.0109 | 0.852 | 0.528 | 0.1379 | 376 |
| warm2_rho1.0 | 0.0084 | 0.0109 | 0.453 | 0.158 | 0.1385 | 376 |
| warm4_rho0.5 | 0.0077 | 0.0093 | 0.855 | 0.702 | 0.1171 | 376 |
| warm8_rho0.5 | 0.0063 | 0.0070 | 0.875 | 0.714 | 0.0870 | 376 |

**Verdict: PASS** (warm2 0.0081 vs cold2 0.0112)

## E3 — surprise

base rates: {"event_within_8": 0.078, "event_last_8": 0.072, "event_within_16": 0.16, "event_last_16": 0.14}

| signal | AUROC event≤8 | AUROC event≤16 | ρ open-loop k8 | ρ fast2 k8 |
|---|---|---|---|---|
| lat_err_pred16 | 0.516 | 0.504 | -0.027 | 0.125 |
| lat_err_pred32 | 0.418 | 0.418 | 0.112 | 0.201 |
| pix_err_pred4 | 0.299 | 0.453 | 0.267 | 0.493 |
| pix_err_pred16 | 0.353 | 0.389 | 0.117 | 0.492 |
| seed_disagreement | 0.446 | 0.428 | 0.278 | 0.220 |
| code_low_conf | 0.510 | 0.500 | -0.042 | -0.077 |
| code_change | 0.445 | 0.506 | 0.091 | 0.128 |

**Verdict: FAIL** (best AUROC 0.516, best ρ 0.278)

E2 per task (warm2 / cold2 / floor, rad): 3/6 tasks pass

- beat_block_hammer: 0.0075 / 0.0108 / 0.0054 → PASS (n=22)
- click_alarmclock: 0.0066 / 0.0111 / 0.0056 → PASS (n=10)
- handover_block: 0.0103 / 0.0116 / 0.0062 → FAIL (n=64)
- hanging_mug: 0.0088 / 0.0118 / 0.0055 → FAIL (n=75)
- open_microwave: 0.0063 / 0.0110 / 0.0051 → PASS (n=98)
- stack_blocks_three: 0.0094 / 0.0112 / 0.0053 → FAIL (n=107)

## B3 — chained warm starts (median h8 arm error vs fresh, rad)

- 1-step warm: hop1: 0.0103 (cold 0.0527, floor 0.0055), hop2: 0.0073 (cold 0.0454, floor 0.0052), hop3: 0.0098 (cold 0.0501, floor 0.0049), hop4: 0.0126 (cold 0.0481, floor 0.0055), hop5: 0.0099 (cold 0.0502, floor 0.0053), hop6: 0.0074 (cold 0.0433, floor 0.0053), hop7: 0.0072 (cold 0.0515, floor 0.0052), hop8: 0.0116 (cold 0.0493, floor 0.0056), hop9: 0.0118 (cold 0.0433, floor 0.0056), hop10: 0.0111 (cold 0.0530, floor 0.0055); drift ρ=0.15
- 2-step warm: hop1: 0.0129 (cold 0.0106, floor 0.0055), hop2: 0.0070 (cold 0.0111, floor 0.0052), hop3: 0.0100 (cold 0.0117, floor 0.0049), hop4: 0.0097 (cold 0.0116, floor 0.0055), hop5: 0.0100 (cold 0.0126, floor 0.0053), hop6: 0.0075 (cold 0.0117, floor 0.0053), hop7: 0.0078 (cold 0.0108, floor 0.0052), hop8: 0.0103 (cold 0.0125, floor 0.0056), hop9: 0.0117 (cold 0.0114, floor 0.0056), hop10: 0.0121 (cold 0.0119, floor 0.0055); drift ρ=0.08

**Verdict: FAIL** (last hop / floor 2.20, drift ρ 0.08)

## B4/B5 — end-to-end replan latency

| stage | p50 ms | p95 ms |
|---|---|---|
| core/eager_cold20_e2e | 1289.6 | 1292.8 |
| core/eager_cold4_e2e | 279.4 | 279.9 |
| core/eager_warm1_e2e | 90.8 | 90.9 |
| core/eager_warm2_e2e | 153.6 | 154.0 |
| core/compiled_joint_step | 37.8 | 37.9 |
| core/compiled_cold20_e2e | 795.9 | 799.2 |
| core/compiled_cold4_e2e | 180.7 | 180.9 |
| core/compiled_warm1_e2e | 65.9 | 66.0 |
| core/compiled_warm2_e2e | 104.3 | 107.3 |
| fastjoint/fastjoint_nis20_e2e | 792.2 | 794.3 |
| fastjoint/fastjoint_nis4_e2e | 180.4 | 180.6 |

B4 warm-2 replan p95 107.3 ms → PASS ≤400; B5 compiled joint step 37.8 ms → PASS ≤50

## E4 — sensitivity (perturbed error / seed floor; ~1 = no effect)

| perturbation | sensitivity h8 | code canvas agree |
|---|---|---|
| view0/down_x2/both | 0.82 | 0.949 |
| view0/down_x4/both | 1.14 | 0.930 |
| view0/drop/both | 7.36 | 0.844 |
| view0/drop/vae | 7.08 | 1.000 |
| view0/drop/vit | 0.46 | 0.844 |
| view0/occl_r0c0/both | 0.90 | 0.922 |
| view0/occl_r0c1/both | 1.42 | 0.883 |
| view0/occl_r0c2/both | 0.68 | 0.906 |
| view0/occl_r1c0/both | 1.06 | 0.953 |
| view0/occl_r1c1/both | 1.29 | 0.973 |
| view0/occl_r1c2/both | 0.75 | 0.961 |
| view1/down_x2/both | 0.49 | 0.922 |
| view1/down_x4/both | 0.71 | 0.922 |
| view1/drop/both | 1.19 | 0.945 |
| view1/drop/vae | 1.03 | 1.000 |
| view1/drop/vit | 0.38 | 0.945 |
| view1/occl_r0c0/both | 0.69 | 0.941 |
| view1/occl_r0c1/both | 0.65 | 0.934 |
| view1/occl_r0c2/both | 0.56 | 0.945 |
| view1/occl_r1c0/both | 0.57 | 0.938 |
| view1/occl_r1c1/both | 0.72 | 0.922 |
| view1/occl_r1c2/both | 0.57 | 0.941 |
| view2/down_x2/both | 0.42 | 0.949 |
| view2/down_x4/both | 0.42 | 0.957 |
| view2/drop/both | 1.15 | 0.871 |
| view2/drop/vae | 1.09 | 1.000 |
| view2/drop/vit | 0.41 | 0.871 |
| view2/occl_r0c0/both | 0.36 | 0.945 |
| view2/occl_r0c1/both | 0.38 | 0.910 |
| view2/occl_r0c2/both | 0.49 | 0.973 |
| view2/occl_r1c0/both | 0.57 | 0.918 |
| view2/occl_r1c1/both | 0.41 | 0.934 |
| view2/occl_r1c2/both | 0.44 | 0.977 |

