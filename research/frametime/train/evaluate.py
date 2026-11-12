#!/usr/bin/env python3
"""Evaluation entry points (metrics per LIT_REVIEW section 4.5).

    evaluate.py controller [--runs RUN_DIR ...]     offline closed-loop controller simulation on the test
                                                    sessions: misses vs quality level vs switches, model
                                                    controllers vs the reactive baseline (margin sweeps,
                                                    comparison at equal mean quality), fixed levels, oracle
    evaluate.py checkpoint RUN_DIR [...]            open-loop test metrics of existing checkpoints
    evaluate.py playtest LOG.sfl [...]              in-game logs: the logged predictions (pred_*) vs t_total

Predictor cost uses the per-family in-game numbers of MODEL_CONTRACT section 7, scaled by size
(cost.py); the real number is measured in the game (t_pred_us).
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import common
from common import Timer, log, make_run_id, rel, write_json
import controllers as C
import cost
import data as Dt
import metrics as M
import runlib
import spkm

_BENCH = None  # shared with forked workers


def _task(t):
    label, mode, budget, margin, run_dir, level = t
    model = common.freeze_online(spkm.read(os.path.join(run_dir, "model.spkm"))) if run_dir else None  # dry: no RLS update
    tm = Timer()
    r = C.run(_BENCH, mode, budget, margin, model=model, decision=_BENCH.cfg["controller"]["decision_output"],
              fixed_level=level)
    r["label"], r["run_dir"], r["sim_seconds"] = label, (rel(run_dir) if run_dir else None), tm.s()
    if model is not None:
        c = cost.estimate(model, evals=max(1.0, r["mean_evals"]), worst_evals=_BENCH.L)
        r["est_pred_us_mean"] = c["ingame_us_est"]
        r["est_pred_us_worst"] = c["ingame_us_est_worst"]
    return r


def auto_runs(res_dir, include_dry=False, per_family=1):
    """Best validated checkpoint per family (val pinball p95), plus the exported naive/max8 baselines."""
    best = {}
    for f in glob.glob(str(Path(res_dir) / "runs" / "*.json")):
        r = json.loads(Path(f).read_text())
        ck = r.get("checkpoint")
        if not ck or ck.get("validator_errors") or (r.get("dry_run") and not include_dry):
            continue
        path = ck["path"]
        path = path if os.path.isabs(path) else str(common.PROJECT_ROOT / path)
        if not os.path.exists(os.path.join(path, "DONE")):
            continue
        score = r.get("headline", {}).get("val", {}).get("pinball_p95")
        if score is None:
            continue
        key = r["experiment"] if r["experiment"].startswith("baseline-") else r["family"]
        if r["experiment"].startswith("baseline-") and r["experiment"] not in ("baseline-naive", "baseline-max8"):
            continue
        best.setdefault(key, []).append((score, path, r["experiment"]))
    out = []
    for key, lst in sorted(best.items()):
        for score, path, exp in sorted(lst)[:per_family]:
            out.append((exp, path))
    return out


def cmd_controller(args):
    global _BENCH
    cfg = common.load_config(args.config)
    ckpt_dir, res_dir = common.output_dirs(args)
    ds = Dt.Dataset(cfg, args.data, args.cache_dir, holdout=args.holdout, splits=("test",), max_sessions=args.max_sessions,
                    workers=args.workers, dry_run=args.dry_run, dry_frames=args.dry_frames)
    runs = [(Path(r).name, str(Path(r).resolve())) for r in args.runs] if args.runs else \
        auto_runs(res_dir, include_dry=args.dry_run)
    log(f"controller simulation: {len(runs)} models: {[r[0] for r in runs]}")
    t = Timer()
    _BENCH = C.Bench(ds, cfg, "test")
    log(f"bench ready ({_BENCH.S} sessions, {_BENCH.sd.N:,} frames, {_BENCH.L} ladder levels) in {t.s():.1f}s")
    sim_val = C.validate_simulator(ds, cfg, "test")
    cc = cfg["controller"]
    budgets = [ds.budget_ms] + [b for b in cc["budgets_ms"] if abs(b - ds.budget_ms) > 1e-3]
    margins = cc["margins_sweep"] if not args.dry_run else [cc["margin"]]
    tasks = []
    for b in budgets:
        for lvl in range(_BENCH.L):
            tasks.append((f"fixed-L{lvl}", "fixed", b, 0.0, None, lvl))
        tasks.append(("oracle", "oracle", b, cc["margin"], None, 0))
        for mg in margins:
            tasks.append(("reactive", "reactive", b, mg, None, 0))
            for name, path in runs:
                tasks.append((name, "model", b, mg, path, 0))
    log(f"{len(tasks)} simulations on {args.workers} workers")
    if args.workers > 1:
        with get_context("fork").Pool(args.workers) as pool:
            results = list(pool.imap(_task, tasks))
    else:
        results = [_task(x) for x in tasks]
    summary = []
    for b in budgets:
        rb = [r for r in results if abs(r["budget_ms"] - b) < 1e-6]
        react = [r for r in rb if r["mode"] == "reactive"]
        react_def = min(react, key=lambda r: abs(r["margin"] - cc["margin"]))
        for label in sorted({r["label"] for r in rb if r["mode"] == "model"}):
            curve = [r for r in rb if r["label"] == label]
            dflt = min(curve, key=lambda r: abs(r["margin"] - cc["margin"]))
            eq = C.equal_quality(curve, react_def["mean_level"])
            summary.append({"budget_ms": b, "model": label, "run_dir": dflt["run_dir"],
                            "misses_per_min": dflt["misses_per_min"], "mean_level": dflt["mean_level"],
                            "switches_per_min": dflt["switches_per_min"], "p99_ms": dflt["p99_ms"],
                            "mean_evals": dflt["mean_evals"], "est_pred_us_mean": dflt.get("est_pred_us_mean"),
                            "misses_per_min_at_reactive_level": eq,
                            "reactive_misses_per_min": react_def["misses_per_min"],
                            "reactive_mean_level": react_def["mean_level"],
                            "reactive_switches_per_min": react_def["switches_per_min"]})
    run_id = make_run_id("controller", "sim")
    out = {"schema": "spark-ft-controller-v1", "run_id": run_id, "created_utc": common.utc_now(),
           "dry_run": bool(args.dry_run), "dataset": ds.info(), "ladder": [list(q) for q in _BENCH.ladder],
           "budgets_ms": budgets, "margins": margins, "decision_output": cc["decision_output"],
           "simulator": {"kind": "analytic", "validation_pairs": sim_val}, "summary": summary, "runs": results,
           "code_version": common.code_version(), "hardware": common.hardware_info(), "time_s": t.s()}
    write_json(res_dir / "controller" / f"{run_id}.json", out)
    (res_dir / "controller" / f"{run_id}.md").write_text(controller_md(out))
    log(f"wrote {rel(res_dir / 'controller' / (run_id + '.json'))}")
    return 0


def _f(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def controller_md(out):
    L = ["# Controller simulation " + out["run_id"], "",
         f"Test sessions of {out['dataset']['name']} ({out['dataset']['num_frames']:,} frames), analytic counterfactual "
         f"simulator, start at L0, decision output {out['decision_output']}." +
         (" DRY RUN (untrained models)." if out["dry_run"] else ""), ""]
    for b in out["budgets_ms"]:
        L += [f"## Budget {b:.2f} ms", "",
              "| controller | margin | misses/min | miss rate | mean level | switches/min | p99 ms | evals | est µs |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted([r for r in out["runs"] if abs(r["budget_ms"] - b) < 1e-6],
                        key=lambda r: (r["mode"] != "fixed", r["mode"], r["label"], r["margin"])):
            L.append(f"| {r['label']} | {r['margin']:.2f} | {_f(r['misses_per_min'])} | {_f(r['miss_rate'], 4)} | "
                     f"{_f(r['mean_level'])} | {_f(r['switches_per_min'])} | {_f(r['p99_ms'])} | {_f(r['mean_evals'])} | "
                     f"{_f(r.get('est_pred_us_mean'))} |")
        L += ["", "**At equal quality** (model controller's misses/min interpolated at the reactive controller's mean "
              "level, default margin):", "", "| model | misses/min @ reactive level | reactive misses/min | reactive level |",
              "|---|---|---|---|"]
        for s in out["summary"]:
            if abs(s["budget_ms"] - b) < 1e-6:
                L.append(f"| {s['model']} | {_f(s['misses_per_min_at_reactive_level'])} | "
                         f"{_f(s['reactive_misses_per_min'])} | {_f(s['reactive_mean_level'])} |")
        L.append("")
    L += ["## Simulator check on paired sessions", "", "| from → to | frames | actual mean | simulated mean | ratio | "
          "median rel. err | corr |", "|---|---|---|---|---|---|---|"]
    for p in out["simulator"]["validation_pairs"]:
        L.append(f"| {p['from']} {p['q_from']} → {p['to']} {p['q_to']} | {p['frames']:,} | {_f(p['mean_actual_ms'])} | "
                 f"{_f(p['mean_sim_ms'])} | {_f(p['ratio_of_means'], 3)} | {_f(p['median_abs_rel_err'], 3)} | "
                 f"{_f(p['corr'], 3)} |")
    return "\n".join(L) + "\n"


def cmd_checkpoint(args):
    cfg = common.load_config(args.config)
    ds = Dt.Dataset(cfg, args.data, args.cache_dir, holdout=args.holdout, splits=("test",), max_sessions=args.max_sessions,
                    workers=args.workers, dry_run=args.dry_run, dry_frames=args.dry_frames)
    te = ds.split("test")
    for r in args.runs:
        m = spkm.read(os.path.join(r, "model.spkm"))
        pred = runlib.predict_model(m, te)
        res = M.evaluate(pred, te.y, ds.budget_ms, te)
        print(json.dumps({"run": r, "headline": M.headline(res)}, indent=1, default=float))
    return 0


def cmd_playtest(args):
    """Score the predictions a model made in the game (logged pred_* fields) against the actual t_total."""
    import framelog
    for path in args.logs:
        header, rec = framelog.read_log(path)
        valid = (rec["flags"] & 4) != 0
        y = rec["t_total_us"][valid].astype(np.float64) / 1000
        budget = float(header["session"].get("budget_ms", 1000 / 60))
        pred = {}
        for f, n in (("pred_p50", "t_p50_ms"), ("pred_p95", "t_p95_ms"), ("pred_p99", "t_p99_ms"), ("pred_pmiss", "p_miss")):
            v = rec[f][valid].astype(np.float64)
            if np.all(v >= 0):
                pred[n] = v
        res = M.evaluate(pred, y, budget)
        tp = rec["t_pred_us"][valid]
        out = {"log": path, "model": header["session"].get("model"), "frames_with_prediction": int(valid.sum()),
               "t_pred_us_p50": float(np.median(tp)) if len(tp) else None,
               "t_pred_us_p99": float(np.percentile(tp, 99)) if len(tp) else None, "headline": M.headline(res)}
        print(json.dumps(out, indent=1, default=float))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = common.add_common_args(sub.add_parser("controller", help="closed-loop controller simulation"))
    c.add_argument("--runs", nargs="*", help="checkpoint folders (default: best validated run per family)")
    k = common.add_common_args(sub.add_parser("checkpoint", help="open-loop test metrics of checkpoints"))
    k.add_argument("runs", nargs="+")
    p = sub.add_parser("playtest", help="score logged in-game predictions")
    p.add_argument("logs", nargs="+")
    a = sub.add_parser("auto-runs", help="print the checkpoint folders the controller simulation would pick")
    a.add_argument("--out-root", default=None)
    a.add_argument("--families", nargs="*", default=None, help="only these families (e.g. mlp gbt linear)")
    a.add_argument("--include-dry", action="store_true")
    args = ap.parse_args(argv)
    return {"controller": cmd_controller, "checkpoint": cmd_checkpoint, "playtest": cmd_playtest,
            "auto-runs": cmd_auto_runs}[args.cmd](args)


def cmd_auto_runs(args):
    root = Path(args.out_root) if args.out_root else common.FT_DIR
    for exp, path in auto_runs(root / "results", include_dry=args.include_dry):
        fam = spkm.read(os.path.join(path, "model.spkm")).family
        if args.families is None or fam in args.families:
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
