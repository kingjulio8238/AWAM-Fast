#!/usr/bin/env python3
"""D2: the measured 5090 latencies (p95, ms) the Stage D sim uses, from e5b_reflex.json + e5_latency_motor.json.
The compiled reflex is the sum of measured pieces (encode + one compiled joint step + the compiled N-step motor);
the eager reflex is measured end to end. Fails loudly if a piece is missing.

    python research/brainloop/env/d2_latency.py /workspace/out_d/d2 > /workspace/out_d/d2/latency_d.json
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
b = json.loads((d / "e5b_reflex.json").read_text())["results"]
m = json.loads((d / "e5_latency_motor.json").read_text())["results"]
if "compiled_error" in b:
    sys.exit(f"compiled path failed: {b['compiled_error']}")
p95 = lambda r, k: float(r[k]["p95"])                                       # noqa: E731
out = {f"cold{n}": p95(b, f"compiled_cold{n}_e2e") for n in (2, 3, 4, 8, 12)}
for n in (2, 4):
    out[f"reflex{n}"] = p95(b, "eager_prepare_static") + p95(b, "compiled_joint_step") + p95(m, f"motor_compiled_n{n}")
    out[f"reflex{n}_eager_e2e"] = p95(b, f"eager_reflex{n}_e2e")
out["pieces"] = {"prepare_static": p95(b, "eager_prepare_static"), "joint_step": p95(b, "compiled_joint_step"),
                 "rebuild_cache_eager": p95(b, "eager_rebuild_cache"),
                 **{f"motor_n{n}": p95(m, f"motor_compiled_n{n}") for n in (2, 4)}}
out["gate_ms"] = 150.0 - 1000.0 / 30                                       # 150 ms including one 30 Hz tick
print(json.dumps(out, indent=1))
