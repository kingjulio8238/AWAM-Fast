"""Turn the lab's JSON lines into summary.json, report.md and report.pdf (light theme, one chart per page).

Pre-registered pass bars (research/brainloop/README.md):
  E1  motor tier with 2 steps, cache <= 8 control steps stale: median h8 arm error <= 1.5x the seed floor AND below
      open-loop playback at the same staleness.
  E2  warm start with 2 steps: median h8 arm error <= 1.5x the seed floor AND below a cold 2-step decode.
  E3  some online surprise signal reaches AUROC >= 0.75 for "gripper event within 8 steps", or Spearman >= 0.4
      with the open-loop error 8 steps later.
  E4  descriptive: per view and pathway, sensitivity = perturbed-vs-base error / seed floor.
  E5  motor tier p95 + one 30 Hz control tick <= 150 ms; 4-step joint decode p95 <= 400 ms (both on an RTX 5090).
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .metrics import auroc, spearman, text_similarity

CONTROL_TICK_MS = 1000.0 / 30.0
MIN_PAIRS = 30          # fewer comparisons than this and a verdict is "insufficient data", not pass/fail
MIN_POS = 10


def _load(out_dir: str) -> Dict[str, List[Dict]]:
    recs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(out_dir, "sweep", "*.jsonl"))) + \
             sorted(glob.glob(os.path.join(out_dir, "e4", "*.jsonl"))) + \
             sorted(glob.glob(os.path.join(out_dir, "chain", "*.jsonl"))):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    # E4 rows written before the fix carry the perturbation type in "kind"
                    if r.get("kind") in ("drop", "down", "occl"):
                        r["pert"] = r["kind"]
                        recs["e4"].append(r)
                        continue
                    if r.get("kind") == "e4":
                        r["kind"] = r.get("pert", r["kind"])
                    recs[r["kind"] if r["kind"] != r.get("pert") else "e4"].append(r)
    return recs


def _med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def _q(xs, q):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.percentile(xs, q)) if xs else float("nan")


def analyze(out_dir: str) -> Dict:
    R = _load(out_dir)
    S: Dict = {"n_episodes": len(R["episode"]), "n_anchors": len(R["anchor"])}
    floors = R["floor"]
    if not floors:
        S["error"] = "no sweep data"
        return S
    # --- calibration: which GT offset the chunk is aligned to, and is the model better than holding still?
    gt = {o: _med([f[f"AvsGT{o}_h8"]["arm"] for f in floors]) for o in (0, 1)}
    o_best = min(gt, key=gt.get)
    hold = _med([f[f"holdvsGT{o_best}_h8"]["arm"] for f in floors])
    floor_h8 = _med([f["AvsB_h8"]["arm"] for f in floors])
    S["calibration"] = {"gt_offset": o_best, "model_vs_gt_h8_arm": gt[o_best], "hold_vs_gt_h8_arm": hold,
                        "model_over_hold": gt[o_best] / hold if hold > 0 else float("nan"),
                        "seed_floor_h8_arm": floor_h8,
                        "seed_floor_h8_grip": _med([f["AvsB_h8"]["grip"] for f in floors]),
                        "code_canvas_agreement_AvsB": _med([f["canvas_AvsB"] for f in floors]),
                        "code_text_sim_AvsB": _med([f["text_AvsB"] for f in floors]),
                        "video_lat_relmse_AvsB": _med([f["lat_AvsB"] for f in floors]),
                        "decode_s_median": _med([a["t_decode_s"] for a in R["anchor"]])}
    cmps = R["cmp"]
    # --- E1
    ks = sorted({c["k"] for c in cmps if c["method"] == "open_loop" or c["method"].startswith("fast")})
    e1 = defaultdict(dict)
    for m in ["open_loop", "fast1", "fast2", "fast4"]:
        for k in ks:
            rows = [c for c in cmps if c["method"] == m and c["k"] == k]
            if not rows:
                continue
            e1[m][k] = {h: _med([c[f"vsA_h{h}"]["arm"] for c in rows]) for h in (4, 8, 16)}
            e1[m][k]["grip_h8"] = _med([c["vsA_h8"]["grip"] for c in rows])
            e1[m][k]["gt_h8"] = _med([c[f"vsGT{o_best}_h8"]["arm"] for c in rows])
            e1[m][k]["n"] = len(rows)
    S["E1"] = {"floor_h8": floor_h8, "by_method": e1}
    tol = [k for k in ks if e1.get("fast2", {}).get(k, {}).get(8, np.inf) <= 1.5 * floor_h8]
    ok8 = all(e1.get("fast2", {}).get(k, {}).get(8, np.inf) <= 1.5 * floor_h8 and
              e1["fast2"][k][8] < e1.get("open_loop", {}).get(k, {}).get(8, -np.inf)
              for k in ks if 0 < k <= 8)
    n8 = min([e1.get("fast2", {}).get(k, {}).get("n", 0) for k in ks if 0 < k <= 8] or [0])
    S["E1"]["verdict"] = {"pass": bool(ok8) if n8 >= MIN_PAIRS else None, "insufficient_data": n8 < MIN_PAIRS,
                          "n_min": n8, "staleness_tolerance_k": max(tol) if tol else None}
    # --- E2
    e2 = {}
    methods = sorted({c["method"] for c in cmps if c["method"].startswith(("warm", "cold"))})
    for m in methods:
        rows = [c for c in cmps if c["method"] == m]
        e2[m] = {"h8": _med([c["vsA_h8"]["arm"] for c in rows]), "h24": _med([c["vsA_h24"]["arm"] for c in rows]),
                 "canvas": _med([c.get("canvas_vsA") for c in rows]), "text": _med([c.get("text_vsA") for c in rows]),
                 "lat": _med([c.get("lat_vsA") for c in rows]), "n": len(rows)}
    S["E2"] = {"by_method": e2}
    w2, c2 = e2.get("warm2_rho0.5", {}).get("h8", np.inf), e2.get("cold2", {}).get("h8", -np.inf)
    n2 = min(e2.get("warm2_rho0.5", {}).get("n", 0), e2.get("cold2", {}).get("n", 0))
    S["E2"]["verdict"] = {"pass": bool(w2 <= 1.5 * floor_h8 and w2 < c2) if n2 >= MIN_PAIRS else None,
                          "insufficient_data": n2 < MIN_PAIRS, "n_min": n2, "warm2_h8": w2, "cold2_h8": c2}
    # --- E2 per task (Stage B bar: holds on >= 5/6 tasks)
    per = {}
    for task in sorted({c["task"] for c in cmps}):
        tw = [c["vsA_h8"]["arm"] for c in cmps if c["task"] == task and c["method"] == "warm2_rho0.5"]
        tc = [c["vsA_h8"]["arm"] for c in cmps if c["task"] == task and c["method"] == "cold2"]
        tf = [f["AvsB_h8"]["arm"] for f in floors if f["task"] == task]
        if tw and tc and tf:
            fw, fc, ff = _med(tw), _med(tc), _med(tf)
            per[task] = {"warm2": fw, "cold2": fc, "floor": ff, "n": len(tw), "pass": bool(fw <= 1.5 * ff and fw < fc)}
    S["E2"]["per_task"] = per
    S["E2"]["tasks_passing"] = f"{sum(v['pass'] for v in per.values())}/{len(per)}"
    # --- E3
    S["E3"] = _e3(R, o_best)
    # --- B3 chained warm starts
    if R["chain"]:
        S["B3"] = _chain(R["chain"])
    # --- E4
    S["E4"] = _e4(R)
    # --- E5
    for mode in ("core", "fastjoint"):
        p5 = os.path.join(out_dir, f"e5b_{mode}.json")
        if os.path.exists(p5):
            S.setdefault("E5b", {})[mode] = json.load(open(p5))["results"]
    if "E5b" in S:
        c = S["E5b"].get("core", {})
        w2 = c.get("compiled_warm2_e2e") or c.get("eager_warm2_e2e")
        js = c.get("compiled_joint_step")
        S["E5b"]["gate"] = {"warm2_e2e_p95_ms": w2["p95"] if w2 else None,
                            "B4_pass_400ms": bool(w2 and w2["p95"] <= 400),
                            "compiled_joint_step_p50_ms": js["p50"] if js else None,
                            "B5_pass_50ms": bool(js and js["p50"] <= 50)}
    for tag in ("bf16", "fp32", "other"):                 # the bf16 run is the headline; fp32 = the sim's setting
        p5 = os.path.join(out_dir, f"e5_latency_{tag}.json")
        if os.path.exists(p5):
            S.setdefault("E5", _e5(json.load(open(p5))))
            S.setdefault("E5_all", {})[tag] = _e5(json.load(open(p5)))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(S, fh, indent=1, default=float)
    _report_md(S, out_dir)
    _report_pdf(S, R, out_dir)
    return S


def _e3(R, o_best) -> Dict:
    by_ep = defaultdict(dict)
    for a in R["anchor"]:
        by_ep[(a["task"], a["ep"])][a["a"]] = a
    floor = {(f["task"], f["ep"], f["a"]): f for f in R["floor"]}
    ol = {(c["task"], c["ep"], c["a"] + c["k"], c["k"]): c for c in R["cmp"] if c["method"] == "open_loop"}
    fast2 = {(c["task"], c["ep"], c["a"] + c["k"], c["k"]): c for c in R["cmp"] if c["method"] == "fast2"}
    events = {(e["task"], e["ep"]): e["events"] for e in R["episode"]}
    sig = defaultdict(list)
    lab = defaultdict(list)
    for key, an in by_ep.items():
        ev_t = np.array([e[0] for e in events.get(key, [])])
        for tau, A in an.items():
            def get(t):
                return an.get(t)
            s = {}
            p16, p32, p4 = get(tau - 16), get(tau - 32), get(tau - 4)
            s["lat_err_pred16"] = float(np.mean([v[0] for v in p16["lat_err"]])) if p16 else np.nan
            s["lat_err_pred32"] = float(np.mean([v[1] for v in p32["lat_err"]])) if p32 and len(p32["lat_err"][0]) > 1 else np.nan
            s["pix_err_pred4"] = float(np.mean([v[0] for v in p4["pix_err"]])) if p4 and p4.get("pix_err") else np.nan
            s["pix_err_pred16"] = (float(np.mean([v[3] for v in p16["pix_err"]]))
                                   if p16 and p16.get("pix_err") and len(p16["pix_err"][0]) > 3 else np.nan)
            f = floor.get((key[0], key[1], tau))
            s["seed_disagreement"] = f["AvsB_h8"]["arm"] if f else np.nan
            conf = [c for c in A["confA"] if c is not None]
            s["code_low_conf"] = 1.0 - float(np.mean(conf)) if conf else np.nan
            s["code_change"] = 1.0 - text_similarity(A["codeA"], p4["codeA"]) if p4 else np.nan
            for name, v in s.items():
                sig[name].append(v)
            for W in (8, 16):
                lab[f"event_within_{W}"].append(int(bool(len(ev_t)) and bool(((ev_t >= tau) & (ev_t < tau + W)).any())))
                lab[f"event_last_{W}"].append(int(bool(len(ev_t)) and bool(((ev_t > tau - W) & (ev_t <= tau)).any())))
            for k in (8, 16):
                c = ol.get((key[0], key[1], tau, k))
                lab[f"openloop_err_k{k}"].append(c["vsA_h8"]["arm"] if c else np.nan)
                c2 = fast2.get((key[0], key[1], tau, k))
                lab[f"fast2_err_k{k}"].append(c2["vsA_h8"]["arm"] if c2 else np.nan)
    out = {"n": len(sig["seed_disagreement"]), "auroc": {}, "spearman": {},
           "base_rate": {k: float(np.mean(v)) for k, v in lab.items() if k.startswith("event")}}
    for name, v in sig.items():
        out["auroc"][name] = {k: auroc(v, lab[k]) for k in lab if k.startswith("event")}
        out["spearman"][name] = {k: spearman(v, lab[k]) for k in lab if k.endswith(("k8", "k16"))}
    best_auc = max([x for d in out["auroc"].values() for k, x in d.items() if k == "event_within_8" and np.isfinite(x)]
                   or [np.nan])
    best_rho = max([x for d in out["spearman"].values() for k, x in d.items() if k == "openloop_err_k8" and np.isfinite(x)]
                   or [np.nan])
    n_pos = int(np.nansum(lab["event_within_8"])) if lab["event_within_8"] else 0
    n_rho = int(np.isfinite(np.asarray(lab["openloop_err_k8"], dtype=float)).sum()) if lab["openloop_err_k8"] else 0
    enough = n_pos >= MIN_POS and n_rho >= MIN_PAIRS
    out["verdict"] = {"pass": bool(enough and ((best_auc >= 0.75) or (best_rho >= 0.4))) if enough else None,
                      "insufficient_data": not enough, "n_event_positives": n_pos, "n_rho_pairs": n_rho,
                      "best_auroc_event8": best_auc, "best_spearman_openloop_k8": best_rho}
    return out


def _chain(rows) -> Dict:
    out = {"by_steps": {}}
    for n in sorted({r["steps"] for r in rows}):
        rr = [r for r in rows if r["steps"] == n]
        hops = sorted({r["hop"] for r in rr})
        by_hop = {h: {"warm": _med([r["warm_h8"]["arm"] for r in rr if r["hop"] == h]),
                      "cold": _med([r["cold_h8"]["arm"] for r in rr if r["hop"] == h]),
                      "floor": _med([r["floor_h8"]["arm"] for r in rr if r["hop"] == h]),
                      "text": _med([r["warm_text"] for r in rr if r["hop"] == h]),
                      "lat": _med([r["warm_lat"] for r in rr if r["hop"] == h])} for h in hops}
        last = by_hop[hops[-1]]
        rho = spearman([r["hop"] for r in rr], [r["warm_h8"]["arm"] for r in rr])
        out["by_steps"][n] = {"by_hop": by_hop, "n": len(rr), "drift_rho": rho,
                              "last_over_floor": last["warm"] / last["floor"] if last["floor"] > 0 else float("nan")}
    two = out["by_steps"].get(2)
    if two:
        n_last = sum(1 for r in rows if r["steps"] == 2 and r["hop"] == max(two["by_hop"]))
        ok = two["last_over_floor"] <= 1.5 and not (two["drift_rho"] > 0.3)
        out["verdict"] = {"pass": bool(ok) if n_last >= 10 else None, "n_last_hop": n_last,
                          "last_over_floor": two["last_over_floor"], "drift_rho": two["drift_rho"]}
    return out


def _e4(R) -> Dict:
    if not R["e4"]:
        return {}
    floor = _med([f["h8"]["arm"] for f in R["e4_floor"]])
    groups = defaultdict(list)
    for r in R["e4"]:
        arg = r["arg"]
        tag = f'{r["kind"]}' + (f'_x{arg}' if r["kind"] == "down" else "") + \
              (f'_r{arg[0]}c{arg[1]}' if r["kind"] == "occl" else "")
        groups[(r["view"], tag, r["path"])].append(r)
    out = {"floor_h8": floor, "floor_canvas": _med([f["canvas"] for f in R["e4_floor"]]), "groups": {}}
    for (v, tag, path), rows in sorted(groups.items()):
        out["groups"][f"view{v}/{tag}/{path}"] = {
            "sens_h8": _med([r["h8"]["arm"] for r in rows]) / floor if floor > 0 else np.nan,
            "canvas": _med([r["canvas"] for r in rows]), "n": len(rows)}
    return out


def _e5(rep) -> Dict:
    r = rep["results"]
    out = {"device": rep["device"], "dtype": rep["dtype"], "ms": {k: v for k, v in r.items() if isinstance(v, dict)
                                                                  and "p50" in v},
           "other": {k: v for k, v in r.items() if not (isinstance(v, dict) and "p50" in v)}}
    motor = r.get("motor_compiled_n2") or r.get("motor_eager_n2")
    dec4 = r.get("fastjoint_compiled_nis4") or r.get("model_decode_eager_nis4")
    out["gate"] = {
        "on_rtx_5090": bool(rep["device"].get("is_rtx_5090")),
        "motor_p95_ms": motor["p95"] if motor else None,
        "motor_reaction_ms": (motor["p95"] + CONTROL_TICK_MS) if motor else None,
        "motor_pass_150ms": bool(motor and motor["p95"] + CONTROL_TICK_MS <= 150),
        "deliberation_p95_ms_nis4": dec4["p95"] if dec4 else None,
        "deliberation_pass_400ms": bool(dec4 and dec4["p95"] <= 400),
    }
    return out


def _verdict(v):
    if v.get("pass") is None:
        n = next((v[k] for k in ("n_min", "n_rho_pairs", "n_last_hop", "n_event_positives") if k in v), None)
        return f"INSUFFICIENT DATA (n={n})"
    return "PASS" if v["pass"] else "FAIL"


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return "nan" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def _report_md(S, out_dir):
    L = ["# Brainloop E1–E5 report", ""]
    c = S["calibration"]
    L += ["## Calibration", "",
          f"- anchors: {S['n_anchors']} over {S['n_episodes']} episodes; median decode {c['decode_s_median']:.2f} s",
          f"- action chunk aligned to GT offset {c['gt_offset']}; model h8 arm error {c['model_vs_gt_h8_arm']:.4f} rad "
          f"vs hold-still {c['hold_vs_gt_h8_arm']:.4f} (ratio {c['model_over_hold']:.2f})",
          f"- seed floor h8: arm {c['seed_floor_h8_arm']:.4f} rad, gripper {c['seed_floor_h8_grip']:.4f}; code canvas "
          f"agreement between seeds {c['code_canvas_agreement_AvsB']:.3f}, text similarity {c['code_text_sim_AvsB']:.3f}", ""]
    e1 = S["E1"]
    L += ["## E1 — staleness (median h8 arm error, rad, vs fresh decode)", "",
          "| method | " + " | ".join(f"k={k}" for k in sorted(next(iter(e1["by_method"].values()), {}))) + " |",
          "|---|" + "---|" * len(next(iter(e1["by_method"].values()), {}))]
    for m, d in e1["by_method"].items():
        L.append(f"| {m} | " + " | ".join(_fmt(d[k][8]) for k in sorted(d)) + " |")
    L += ["", f"seed floor {e1['floor_h8']:.4f}. **Verdict: {_verdict(e1['verdict'])}** "
          f"(staleness tolerance k = {e1['verdict']['staleness_tolerance_k']})", ""]
    L += ["## E2 — warm start", "", "| method | h8 arm | h24 arm | canvas agree | code text sim | video rel-MSE | n |",
          "|---|---|---|---|---|---|---|"]
    for m, d in S["E2"]["by_method"].items():
        L.append(f"| {m} | {_fmt(d['h8'])} | {_fmt(d['h24'])} | {_fmt(d['canvas'], 3)} | {_fmt(d['text'], 3)} | "
                 f"{_fmt(d['lat'], 4)} | {d['n']} |")
    v = S["E2"]["verdict"]
    L += ["", f"**Verdict: {_verdict(v)}** (warm2 {_fmt(v['warm2_h8'])} vs cold2 {_fmt(v['cold2_h8'])})", ""]
    e3 = S["E3"]
    L += ["## E3 — surprise", "", f"base rates: {json.dumps({k: round(x, 3) for k, x in e3['base_rate'].items()})}", "",
          "| signal | AUROC event≤8 | AUROC event≤16 | ρ open-loop k8 | ρ fast2 k8 |", "|---|---|---|---|---|"]
    for name in e3["auroc"]:
        a, s = e3["auroc"][name], e3["spearman"][name]
        L.append(f"| {name} | {_fmt(a.get('event_within_8'), 3)} | {_fmt(a.get('event_within_16'), 3)} | "
                 f"{_fmt(s.get('openloop_err_k8'), 3)} | {_fmt(s.get('fast2_err_k8'), 3)} |")
    v = e3["verdict"]
    L += ["", f"**Verdict: {_verdict(v)}** (best AUROC {_fmt(v['best_auroc_event8'], 3)}, "
          f"best ρ {_fmt(v['best_spearman_openloop_k8'], 3)})", ""]
    if S["E2"].get("per_task"):
        L += [f"E2 per task (warm2 / cold2 / floor, rad): {S['E2']['tasks_passing']} tasks pass", ""]
        for t, d in S["E2"]["per_task"].items():
            L.append(f"- {t}: {d['warm2']:.4f} / {d['cold2']:.4f} / {d['floor']:.4f} → {'PASS' if d['pass'] else 'FAIL'} (n={d['n']})")
        L.append("")
    if S.get("B3"):
        L += ["## B3 — chained warm starts (median h8 arm error vs fresh, rad)", ""]
        for n, d in S["B3"]["by_steps"].items():
            L.append(f"- {n}-step warm: " + ", ".join(f"hop{h}: {v['warm']:.4f} (cold {v['cold']:.4f}, floor {v['floor']:.4f})"
                                                     for h, v in d["by_hop"].items()) + f"; drift ρ={d['drift_rho']:.2f}")
        v = S["B3"].get("verdict", {})
        L += ["", f"**Verdict: {_verdict(v) if v else '—'}** (last hop / floor {_fmt(v.get('last_over_floor'), 2)}, drift ρ {_fmt(v.get('drift_rho'), 2)})", ""]
    if S.get("E5b"):
        L += ["## B4/B5 — end-to-end replan latency", "", "| stage | p50 ms | p95 ms |", "|---|---|---|"]
        for mode in ("core", "fastjoint"):
            for k, d in S["E5b"].get(mode, {}).items():
                if isinstance(d, dict) and "p50" in d:
                    L.append(f"| {mode}/{k} | {d['p50']:.1f} | {d['p95']:.1f} |")
        g = S["E5b"]["gate"]
        L += ["", f"B4 warm-2 replan p95 {_fmt(g['warm2_e2e_p95_ms'], 1)} ms → {'PASS' if g['B4_pass_400ms'] else 'FAIL'} ≤400; "
              f"B5 compiled joint step {_fmt(g['compiled_joint_step_p50_ms'], 1)} ms → {'PASS' if g['B5_pass_50ms'] else 'FAIL'} ≤50", ""]
    if S.get("E4"):
        L += ["## E4 — sensitivity (perturbed error / seed floor; ~1 = no effect)", "", "| perturbation | sensitivity h8 | code canvas agree |", "|---|---|---|"]
        for k, d in S["E4"]["groups"].items():
            L.append(f"| {k} | {_fmt(d['sens_h8'], 2)} | {_fmt(d['canvas'], 3)} |")
        L.append("")
    if S.get("E5"):
        e5 = S["E5"]
        L += ["## E5 — latency", "", f"device {e5['device'].get('device')} ({e5['dtype']}), 5090: {e5['gate']['on_rtx_5090']}", "",
              "| stage | p50 ms | p95 ms |", "|---|---|---|"]
        for k, d in e5["ms"].items():
            L.append(f"| {k} | {d['p50']:.1f} | {d['p95']:.1f} |")
        g = e5["gate"]
        L += ["", f"motor reaction (p95 + 1 tick) {_fmt(g['motor_reaction_ms'], 1)} ms → "
              f"{'PASS' if g['motor_pass_150ms'] else 'FAIL'} ≤150; 4-step deliberation p95 "
              f"{_fmt(g['deliberation_p95_ms_nis4'], 1)} ms → {'PASS' if g['deliberation_pass_400ms'] else 'FAIL'} ≤400", ""]
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def _report_pdf(S, R, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return
    plt.style.use("default")
    with PdfPages(os.path.join(out_dir, "report.pdf")) as pdf:
        e1 = S["E1"]["by_method"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for m, d in e1.items():
            ks = sorted(d)
            ax.plot(ks, [d[k][8] for k in ks], marker="o", label=m)
        ax.axhline(S["E1"]["floor_h8"], color="gray", ls="--", label="seed floor")
        ax.axhline(1.5 * S["E1"]["floor_h8"], color="gray", ls=":", label="1.5× floor (pass bar)")
        ax.set_xlabel("cache staleness k (control steps)"); ax.set_ylabel("median h8 arm error vs fresh (rad)")
        ax.set_title("E1: motor tier with a stale code/video cache"); ax.legend(); ax.grid(alpha=.3)
        pdf.savefig(fig); plt.close(fig)
        e2 = S["E2"]["by_method"]
        if e2:
            fig, ax = plt.subplots(figsize=(8, 5))
            names = list(e2)
            ax.bar(range(len(names)), [e2[n]["h8"] for n in names], color=["#4C72B0" if n.startswith("warm") else "#DD8452" for n in names])
            ax.axhline(S["E1"]["floor_h8"], color="gray", ls="--")
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_ylabel("median h8 arm error vs 20-step fresh (rad)"); ax.set_title("E2: warm vs cold decodes")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        e3 = S["E3"]["auroc"]
        if e3:
            fig, ax = plt.subplots(figsize=(8, 5))
            names = list(e3)
            ax.barh(range(len(names)), [e3[n].get("event_within_8", np.nan) for n in names], color="#55A868")
            ax.axvline(0.5, color="gray", ls="--"); ax.axvline(0.75, color="gray", ls=":")
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.set_xlim(0, 1)
            ax.set_xlabel("AUROC for 'gripper event within 8 steps'"); ax.set_title("E3: surprise signals")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        e4 = S.get("E4", {}).get("groups", {})
        for v in range(3):
            occ = {k: d for k, d in e4.items() if k.startswith(f"view{v}/occl")}
            if not occ:
                continue
            grid = np.full((2, 3), np.nan)
            for k, d in occ.items():
                tag = k.split("/")[1]
                r, c = int(tag.split("_r")[1][0]), int(tag.split("c")[-1][0])
                grid[r, c] = d["sens_h8"]
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(grid, cmap="viridis"); fig.colorbar(im, ax=ax, label="sensitivity (× seed floor)")
            ax.set_title(f"E4: occlusion sensitivity, view {v} ({['head', 'left wrist', 'right wrist'][v]})")
            pdf.savefig(fig); plt.close(fig)
        if S.get("E5"):
            ms = S["E5"]["ms"]
            fig, ax = plt.subplots(figsize=(8, 6))
            names = list(ms)
            ax.barh(range(len(names)), [ms[n]["p95"] for n in names], color="#8172B3")
            ax.axvline(150, color="red", ls="--", label="150 ms gate"); ax.set_xscale("log")
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel("p95 latency (ms, log)"); ax.set_title(f"E5: latency on {S['E5']['device'].get('device')}")
            ax.legend(); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
