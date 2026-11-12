#!/usr/bin/env python3
"""Headless playtest benchmark for Spark Warden quality controllers.

  run      one game run (paced loop, pinned with taskset) ->
           research/frametime/playtest/logs[/<dir>]/<stress>_<ctl>_s<seed>[_<suffix>].sfl
           plus <name>.stdout.txt and <name>.stderr.txt next to it.
           Fails loudly (exit 1) on a nonzero exit, "MODEL REJECTED", or a bad log.
  analyze  every .sfl under research/frametime/playtest/logs (recursive) ->
           research/frametime/playtest/results.csv, plus markdown tables on stdout.

Phase 1 logs live in logs/phase1/ because their names would collide with phase 2
(same stress/ctl/seed, 20 s instead of 40 s). Phase 3 logs carry a _P or _E suffix.
"""
import argparse
import csv
import glob
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
BIN = os.path.join(ROOT, "build/spark_warden")
CKPT = os.path.join(ROOT, "research/frametime/checkpoints")
PLAY = os.path.join(ROOT, "research/frametime/playtest")
LOGS = os.path.join(PLAY, "logs")
TRAIN = os.path.join(ROOT, "research/frametime/train")

# stress name -> (--out-scale, --spawn-mult, --burst-mult)
STRESS = {
    "S3": (2.35, 2.75, 4.0),
    "S4": (2.9, 4.0, 5.5),
    "S5": (3.5, 5.0, 7.0),
    "S6": (4.0, 6.0, 8.0),
}
MODELS = {
    "rls": "20261110-040736_rls_rls-hybrid",
    "gbt": "20261109-180416_gbt_gbt-q",
    "gru": "20261109-180406_gru_gru-stream",
}
CTL_ORDER = ["fixed-L0", "reactive", "model-rls", "model-gbt", "model-gru"]
NAME_RE = re.compile(
    r"^(S\d)_(fixed-L\d|reactive|model-(rls|gbt|gru))(?:_m(-?\d+\.\d+))?_s(\d+)(?:_([PE]))?\.sfl$")
BUDGET_MS = 1000.0 / 60.0


def ctl_args(ctl):
    if ctl.startswith("fixed-L"):
        return ["--ctl", "fixed", "--quality", "L" + ctl[len("fixed-L"):]]
    if ctl == "reactive":
        return ["--ctl", "reactive", "--quality", "L0"]
    if ctl.startswith("model-"):
        model = os.path.join(CKPT, MODELS[ctl[len("model-"):]])
        return ["--ctl", "model", "--model", model, "--quality", "L0"]
    raise SystemExit(f"unknown controller {ctl}")


def check_ac():
    vals = []
    for p in glob.glob("/sys/class/power_supply/A*/online"):
        with open(p) as f:
            vals.append(f.read().strip())
    if not vals or any(v != "1" for v in vals):
        raise SystemExit(f"not on AC power (online={vals}); stopping")


def fail(msg):
    print("PLAYTEST FAILURE: " + msg, file=sys.stderr)
    sys.exit(1)


def cmd_run(a):
    check_ac()
    out_scale, spawn, burst = STRESS[a.stress]
    margin_args = []
    mtag = ""
    if a.margin is not None:
        if not a.ctl.startswith("model-"):
            raise SystemExit("--margin only applies to model controllers")
        margin_args = ["--ctl-margin", f"{a.margin:.2f}"]
        mtag = f"_m{a.margin:.2f}"
    name = f"{a.stress}_{a.ctl}{mtag}_s{a.seed}" + (f"_{a.suffix}" if a.suffix else "")
    logdir = a.logdir or LOGS
    os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, name + ".sfl")
    if os.path.exists(log):
        os.remove(log)
    cmd = ["taskset", "-c", str(a.cpu), BIN,
           "--headless", "--bot", "--seed", str(a.seed),
           "--loop", "paced", "--max-seconds", str(a.seconds),
           *ctl_args(a.ctl), *margin_args,
           "--out-scale", str(out_scale), "--spawn-mult", str(spawn),
           "--burst-mult", str(burst),
           "--log", log]
    out_p = os.path.join(logdir, name + ".stdout.txt")
    err_p = os.path.join(logdir, name + ".stderr.txt")
    t0 = time.time()
    with open(out_p, "w") as fo, open(err_p, "w") as fe:
        try:
            p = subprocess.run(cmd, cwd=ROOT, stdout=fo, stderr=fe,
                               timeout=a.seconds + 120)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            fail(f"{name}: timed out")
    wall = time.time() - t0
    err_text = open(err_p).read()
    out_text = open(out_p).read()
    if "MODEL REJECTED" in err_text or "MODEL REJECTED" in out_text:
        fail(f"{name}: MODEL REJECTED (see {err_p})")
    if rc != 0:
        fail(f"{name}: exit code {rc} (see {err_p})")
    if not os.path.exists(log):
        fail(f"{name}: no log written")
    sys.path.insert(0, TRAIN)
    from framelog import read_log  # reading only
    _, rec = read_log(log)
    if len(rec) == 0:
        fail(f"{name}: empty log")
    sim_s = (int(rec["step0"][-1]) + int(rec["n_steps"][-1]) - int(rec["step0"][0])) / 60
    print(f"{name}: frames={len(rec)} sim_s={sim_s:.2f} wall={wall:.1f}s "
          f"fps={len(rec) / sim_s:.2f} -> {log}")


def metrics(rec, expected_s):
    f = lambda k: rec[k].astype(np.float64)
    n = len(rec)
    step0 = rec["step0"].astype(np.int64)
    sim_s = (int(step0[-1]) + int(rec["n_steps"][-1]) - int(step0[0])) / 60.0
    minutes = sim_s / 60.0
    total_ms = f("t_total_us") / 1000.0
    work = (f("t_total_us") + f("t_pred_us") + f("t_log_us")) / 1000.0
    miss = work > BUDGET_MS
    p99 = float(np.percentile(work, 99))
    lvl = rec["q_level"].astype(np.int64)
    on_lvl = lvl != 255
    pred_us = f("t_pred_us")
    pred_p95 = f("pred_p95")
    sel = pred_p95 > 0
    has_pred = bool(sel.any())
    first5 = sel & ((step0 - step0[0]) < 300)
    rest = sel & ~first5
    cov = lambda m: float(np.mean(total_ms[m] <= pred_p95[m])) if m.any() else float("nan")
    m = {
        "frames": n,
        "sim_s": sim_s,
        "fps": n / sim_s,
        "low1_fps": min(60.0, 1000.0 / p99),
        "p99_ms": p99,
        "miss_pct": 100.0 * float(miss.mean()),
        "misses_per_min": float(miss.sum()) / minutes,
        "mean_level": float(lvl[on_lvl].mean()) if on_lvl.any() else float("nan"),
        "mean_scale": float(rec["q_scale"].astype(np.float64).mean()),
        "mean_glow": float(rec["q_glow"].astype(np.float64).mean()),
        "switches_per_min": float(np.count_nonzero(rec["ctl_action"] != 0)) / minutes,
        "pred_us_mean": float(pred_us.mean()),
        "pred_us_p99": float(np.percentile(pred_us, 99)),
        "cov_p95": cov(sel) if has_pred else float("nan"),
        "cov_p95_first5s": cov(first5) if has_pred else float("nan"),
        "cov_p95_rest": cov(rest) if has_pred else float("nan"),
        "on_ac": bool(np.all(rec["flags"].astype(np.int64) & 8)),
        "expected_s": expected_s,
    }
    return m


def parse_name(path):
    base = os.path.basename(path)
    mt = NAME_RE.match(base)
    if not mt:
        return None
    stress, ctl, _, margin, seed, suffix = mt.groups()
    phase_dir = os.path.basename(os.path.dirname(path))
    if phase_dir == "phase1":
        phase, expected = "1", 20
    elif phase_dir == "phase4":
        phase, expected = "4", 40
    elif suffix:
        phase, expected = "3", 40
    else:
        phase, expected = "2", 40
    # phase 2/3 model runs used the default margin 0.10 (no margin in the name)
    if margin is not None:
        mval = float(margin)
    elif ctl.startswith("model-") and phase in ("2", "3"):
        mval = 0.10
    else:
        mval = float("nan")
    return dict(phase=phase, stress=stress, ctl=ctl, seed=int(seed), margin=mval,
                suffix=suffix or "", expected_s=expected, base=base, path=path)


def cmd_analyze(_a):
    sys.path.insert(0, TRAIN)
    from framelog import read_log
    rows = []
    anomalies = []
    for path in sorted(glob.glob(os.path.join(LOGS, "**", "*.sfl"), recursive=True)):
        info = parse_name(path)
        if info is None:
            anomalies.append(f"unparsed log name: {path}")
            continue
        _, rec = read_log(path)
        m = metrics(rec, info["expected_s"])
        row = {k: info[k] for k in ("base", "phase", "stress", "ctl", "margin", "seed", "suffix")}
        row.update(m)
        stderr_p = path[:-4] + ".stderr.txt"
        if os.path.exists(stderr_p) and "MODEL REJECTED" in open(stderr_p).read():
            anomalies.append(f"{info['base']}: MODEL REJECTED in stderr")
        if not m["on_ac"]:
            anomalies.append(f"{info['base']}: not on AC power (flags)")
        if abs(m["sim_s"] - info["expected_s"]) > 1.0:
            anomalies.append(f"{info['base']}: sim_s={m['sim_s']:.2f}, expected {info['expected_s']}")
        rows.append(row)
    if not rows:
        raise SystemExit("no logs found")

    cols = list(rows[0].keys())
    with open(os.path.join(PLAY, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and math.isnan(v)) else v)
                        for k, v in r.items()})

    def sel(**kw):
        out = []
        for r in rows:
            if all(r[k] == v for k, v in kw.items()):
                out.append(r)
        return out

    def mean(rs, k):
        vals = [r[k] for r in rs if not (isinstance(r[k], float) and math.isnan(r[k]))]
        return float(np.mean(vals)) if vals else float("nan")

    def fmt(v, d=2):
        if isinstance(v, float) and math.isnan(v):
            return "n/a"
        return f"{v:.{d}f}"

    print(f"# results: {len(rows)} runs written to {os.path.join(PLAY, 'results.csv')}\n")
    print("## Phase 1 (20 s, seed 1)\n")
    print("| stress | ctl | fps | miss_pct | low1_fps | p99_ms | mean_level |")
    print("|---|---|---|---|---|---|---|")
    for st in STRESS:
        for c in ("fixed-L0", "fixed-L8"):
            for r in sel(phase="1", stress=st, ctl=c):
                print(f"| {st} | {c} | {fmt(r['fps'])} | {fmt(r['miss_pct'])} | "
                      f"{fmt(r['low1_fps'])} | {fmt(r['p99_ms'])} | {fmt(r['mean_level'])} |")

    print("\n## Phase 2 (40 s, seeds 1-3; mean over seeds, fps min-max over seeds)\n")
    stresses = [s for s in STRESS if sel(phase="2", stress=s)]
    for st in stresses:
        print(f"### {st}\n")
        print("| ctl | n | fps (mean) | fps min-max | low1_fps | miss_pct | misses_per_min | "
              "p99_ms | mean_level | mean q_scale | mean q_glow | switches_per_min | "
              "pred_us_mean | pred_us_p99 | cov_p95 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in CTL_ORDER:
            rs = sel(phase="2", stress=st, ctl=c)
            if not rs:
                continue
            fps = [r["fps"] for r in rs]
            print(f"| {c} | {len(rs)} | {fmt(mean(rs,'fps'))} | "
                  f"{min(fps):.2f}-{max(fps):.2f} | {fmt(mean(rs,'low1_fps'))} | "
                  f"{fmt(mean(rs,'miss_pct'))} | {fmt(mean(rs,'misses_per_min'))} | "
                  f"{fmt(mean(rs,'p99_ms'))} | {fmt(mean(rs,'mean_level'))} | "
                  f"{fmt(mean(rs,'mean_scale'),3)} | {fmt(mean(rs,'mean_glow'),3)} | "
                  f"{fmt(mean(rs,'switches_per_min'))} | {fmt(mean(rs,'pred_us_mean'),1)} | "
                  f"{fmt(mean(rs,'pred_us_p99'),1)} | {fmt(mean(rs,'cov_p95'),4)} |")
        print()

    print("## Phase 3 (S4, 40 s, seeds 1-3; core P = taskset -c 2, E = taskset -c 10)\n")
    print("| ctl | core | n | fps (mean) | fps min-max | miss_pct | p99_ms | pred_us_mean | "
          "pred_us_p99 | cov_p95 | cov_p95 first 5 s | cov_p95 rest |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in ("model-rls", "model-gbt"):
        for core in ("P", "E"):
            rs = sel(phase="3", stress="S4", ctl=c, suffix=core)
            if not rs:
                continue
            fps = [r["fps"] for r in rs]
            print(f"| {c} | {core} | {len(rs)} | {fmt(mean(rs,'fps'))} | "
                  f"{min(fps):.2f}-{max(fps):.2f} | {fmt(mean(rs,'miss_pct'))} | "
                  f"{fmt(mean(rs,'p99_ms'))} | {fmt(mean(rs,'pred_us_mean'),1)} | "
                  f"{fmt(mean(rs,'pred_us_p99'),1)} | {fmt(mean(rs,'cov_p95'),4)} | "
                  f"{fmt(mean(rs,'cov_p95_first5s'),4)} | {fmt(mean(rs,'cov_p95_rest'),4)} |")

    def near(r, m):
        return (not math.isnan(r["margin"])) and abs(r["margin"] - m) < 1e-9

    def grp(st, ctl, m):
        if ctl == "reactive":
            return sel(phase="4", stress=st, ctl="reactive")
        if m == 0.10:
            return sel(phase="2", stress=st, ctl=ctl)
        return [r for r in sel(phase="4", stress=st, ctl=ctl) if near(r, m)]

    def interp(pts, x):
        if len(pts) < 2 or x < pts[0][0] or x > pts[-1][0]:
            return "out of range"
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                if x1 == x0:
                    return f"{y0:.2f}"
                return f"{y0 + (y1 - y0) * (x - x0) / (x1 - x0):.2f}"
        return "out of range"

    if sel(phase="4"):
        print("\n## Phase 4: equal-quality comparison (40 s, seeds 1-3)\n")
        print("margin 0.10 rows are the phase 2 runs (same seeds). Reactive rows are the phase 4 reactive runs.\n")
        for st in ("S4", "S5"):
            print(f"### {st}\n")
            print("| controller | margin | n | mean_level | misses_per_min (mean) | misses_per_min min-max | "
                  "miss_pct | fps | low1_fps | switches_per_min | pred_us_mean | cov_p95 |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|")
            entries = [("reactive", "n/a", grp(st, "reactive", None))]
            for c in ("model-rls", "model-gru", "model-gbt"):
                for m in (-0.10, 0.00, 0.05, 0.10):
                    entries.append((c, f"{m:.2f}", grp(st, c, m)))
            for name, mtxt, rs in entries:
                if not rs:
                    continue
                mp = [r["misses_per_min"] for r in rs]
                print(f"| {name} | {mtxt} | {len(rs)} | {fmt(mean(rs, 'mean_level'))} | "
                      f"{fmt(mean(rs, 'misses_per_min'))} | {min(mp):.2f}-{max(mp):.2f} | "
                      f"{fmt(mean(rs, 'miss_pct'))} | {fmt(mean(rs, 'fps'))} | "
                      f"{fmt(mean(rs, 'low1_fps'))} | {fmt(mean(rs, 'switches_per_min'))} | "
                      f"{fmt(mean(rs, 'pred_us_mean'), 1)} | {fmt(mean(rs, 'cov_p95'), 4)} |")
            print()

        print("### Interpolation: model misses/min at the phase 4 reactive mean_level\n")
        print("Linear interpolation over the model's (mean_level, misses_per_min) points at margins "
              "-0.10, 0.00, 0.05, 0.10 (seed means). 'out of range' if the reactive level is outside "
              "the model's level range.\n")
        print("| model | stress | reactive level | reactive misses/min | model level range | "
              "model misses/min at that level |")
        print("|---|---|---|---|---|---|")
        for st in ("S4", "S5"):
            rr = grp(st, "reactive", None)
            if not rr:
                continue
            rl = mean(rr, "mean_level")
            rm = mean(rr, "misses_per_min")
            for c in ("model-rls", "model-gru", "model-gbt"):
                pts = []
                for m in (-0.10, 0.00, 0.05, 0.10):
                    rs = grp(st, c, m)
                    if rs:
                        pts.append((mean(rs, "mean_level"), mean(rs, "misses_per_min")))
                pts.sort()
                if pts:
                    rng = f"{pts[0][0]:.3f}-{pts[-1][0]:.3f}"
                else:
                    rng = "n/a"
                print(f"| {c} | {st} | {fmt(rl, 3)} | {fmt(rm, 2)} | {rng} | {interp(pts, rl)} |")
        print()

    print("\n## Per-run anomalies\n")
    if anomalies:
        for a in anomalies:
            print("- " + a)
    else:
        print("none")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one configuration")
    r.add_argument("--stress", required=True, choices=list(STRESS))
    r.add_argument("--ctl", required=True, choices=CTL_ORDER + ["fixed-L8"])
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--seconds", type=float, required=True)
    r.add_argument("--cpu", type=int, default=2, help="taskset CPU (2 = P-core, 10 = E-core)")
    r.add_argument("--suffix", choices=["P", "E"], default=None)
    r.add_argument("--margin", type=float, default=None,
                   help="model controllers only: --ctl-margin (phase 4)")
    r.add_argument("--logdir", default=None, help="default: research/frametime/playtest/logs")
    r.set_defaults(func=cmd_run)
    an = sub.add_parser("analyze", help="aggregate all logs")
    an.set_defaults(func=cmd_analyze)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
