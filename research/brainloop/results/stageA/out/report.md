# Brainloop E1–E5 report

## Calibration

- anchors: 52 over 2 episodes; median decode 2.99 s
- action chunk aligned to GT offset 1; model h8 arm error 0.0079 rad vs hold-still 0.0161 (ratio 0.49)
- seed floor h8: arm 0.0055 rad, gripper 0.0007; code canvas agreement between seeds 1.000, text similarity 1.000

## E1 — staleness (median h8 arm error, rad, vs fresh decode)

| method | k=0 | k=4 | k=8 | k=12 | k=16 | k=20 | k=24 |
|---|---|---|---|---|---|---|---|
| open_loop | 0.0000 | 0.0107 | 0.0107 | 0.0125 | 0.0117 | 0.0115 | 0.0130 |
| fast1 | 0.0484 | 0.0529 | 0.0575 | 0.0595 | 0.0636 | 0.0671 | 0.0706 |
| fast2 | 0.0105 | 0.0177 | 0.0245 | 0.0269 | 0.0299 | 0.0379 | 0.0376 |
| fast4 | 0.0088 | 0.0157 | 0.0236 | 0.0258 | 0.0292 | 0.0356 | 0.0372 |

seed floor 0.0055. **Verdict: FAIL** (staleness tolerance k = None)

## E2 — warm start

| method | h8 arm | h24 arm | canvas agree | code text sim | video rel-MSE | n |
|---|---|---|---|---|---|---|
| cold1 | 0.0491 | 0.0495 | 0.211 | 0.106 | 0.0373 | 37 |
| cold2 | 0.0118 | 0.0122 | 0.234 | 0.133 | 0.0354 | 37 |
| cold4 | 0.0096 | 0.0103 | 0.227 | 0.206 | 0.0362 | 37 |
| cold8 | 0.0078 | 0.0082 | 0.344 | 0.357 | 0.0416 | 37 |
| warm1_rho0.5 | 0.0080 | 0.0105 | 0.379 | 0.573 | 0.1600 | 48 |
| warm2_rho0.0 | 0.0083 | 0.0102 | 0.391 | 0.530 | 0.1431 | 48 |
| warm2_rho0.5 | 0.0079 | 0.0100 | 0.395 | 0.579 | 0.1425 | 48 |
| warm2_rho1.0 | 0.0087 | 0.0104 | 0.234 | 0.131 | 0.1422 | 48 |
| warm4_rho0.5 | 0.0079 | 0.0097 | 0.414 | 0.571 | 0.1266 | 48 |
| warm8_rho0.5 | 0.0066 | 0.0073 | 0.418 | 0.568 | 0.0820 | 48 |

**Verdict: PASS** (warm2 0.0079 vs cold2 0.0118)

## E3 — surprise

base rates: {"event_within_8": 0.115, "event_last_8": 0.115, "event_within_16": 0.231, "event_last_16": 0.192}

| signal | AUROC event≤8 | AUROC event≤16 | ρ open-loop k8 | ρ fast2 k8 |
|---|---|---|---|---|
| lat_err_pred16 | 0.623 | 0.693 | -0.144 | 0.163 |
| lat_err_pred32 | 0.433 | 0.408 | 0.087 | 0.142 |
| pix_err_pred4 | 0.284 | 0.432 | 0.181 | 0.391 |
| pix_err_pred16 | 0.307 | 0.357 | 0.125 | 0.613 |
| seed_disagreement | 0.413 | 0.306 | 0.382 | 0.218 |
| code_low_conf | 0.580 | 0.635 | -0.161 | -0.011 |
| code_change | 0.269 | 0.408 | 0.095 | 0.210 |

**Verdict: INSUFFICIENT DATA (n=48)** (best AUROC 0.623, best ρ 0.382)

## E5 — latency

device NVIDIA GeForce RTX 5090 (torch.bfloat16), 5090: True

| stage | p50 ms | p95 ms |
|---|---|---|
| vae_encode_obs_3views | 10.4 | 10.4 |
| vit_prefix_3views | 24.5 | 24.9 |
| joint_mot_step | 151.2 | 154.6 |
| lab_decode_eager_nis20_excl_encode | 2961.6 | 2979.4 |
| model_decode_eager_nis20 | 3051.9 | 3097.6 |
| lab_decode_eager_nis6_excl_encode | 900.5 | 908.6 |
| model_decode_eager_nis6 | 945.5 | 960.7 |
| lab_decode_eager_nis4_excl_encode | 604.5 | 612.6 |
| model_decode_eager_nis4 | 648.0 | 653.2 |
| motor_eager_n1 | 49.4 | 49.9 |
| motor_eager_n2 | 97.7 | 98.4 |
| motor_eager_n4 | 195.0 | 196.2 |

motor reaction (p95 + 1 tick) 131.7 ms → PASS ≤150; 4-step deliberation p95 653.2 ms → FAIL ≤400

