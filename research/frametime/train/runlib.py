"""Finishing a run: evaluate, write the checkpoint (model.spkm + manifest.json + REPORT.md + native/),
validate it, write DONE last, and write the results record that aggregate.py reads.

Every trained model reports: exact parameter count, architecture (one line + details), dataset
(name, frames, bytes, split), training time, hardware, seed, code version and its metrics.
"""
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np

import cost
import fastpred
import metrics as M
import spkm
from common import FT_DIR, PROJECT_ROOT, code_version, hardware_info, log, rel, utc_now, write_json

RESULT_SCHEMA = "spark-ft-result-v1"


def _validator():
    spec = importlib.util.spec_from_file_location("validate_checkpoint", FT_DIR / "tools" / "validate_checkpoint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def spkm_eval_bin():
    for p in (os.environ.get("SPKM_EVAL"), PROJECT_ROOT / "build" / "spkm_eval"):
        if p and Path(p).is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def spkm_eval_check(path):
    """Ask the engine's own loader (build/spkm_eval check) whether it accepts the file, if built."""
    b = spkm_eval_bin()
    if b is None:
        return {"ran": False, "reason": "build/spkm_eval not built (make parity)"}
    p = subprocess.run([b, "check", str(path)], capture_output=True, text=True, timeout=60)
    return {"ran": True, "ok": p.returncode == 0, "output": (p.stdout + p.stderr).strip()[:500]}


def outputs_for(target, quantiles, pmiss=True):
    """OUTS rows for quantile outputs in the target's export form (+ p_miss as a sigmoid logit)."""
    ex = target.export() if target is not None else ("identity", "-")
    tr, base = ex if ex is not None else ("identity", "-")
    outs = []
    for q in quantiles:
        add = target.add_feature(q) if (target is not None and ex is not None) else None
        outs.append((M.qname(q), tr, base, add) if add else (M.qname(q), tr, base))
    if pmiss:
        outs.append(("p_miss", "sigmoid", "-"))
    return outs


def predict_model(m, sd, target=None, decode=False):
    """Predictions (ms) for every row of split `sd` through the exported model (fastpred = engine maths).
    decode=True: the model's quantile outputs are in training space (target not exportable)."""
    Y = fastpred.predict_split(m, sd.X, sd.H, sd.y, sd.offsets)
    pred = {o[0]: Y[:, i] for i, o in enumerate(m.outputs)}
    if decode:
        ref = target.ref(sd.X, sd)
        for k in list(pred):
            if k.startswith("t_p") or k == "t_mean_ms":
                pred[k] = target.decode(pred[k], ref)
                a = target.add_values(sd.X, M.qlevel(k)) if k.startswith("t_p") else None
                if a is not None:
                    pred[k] = pred[k] + a
    return pred


def check_close(a, b, what, atol=1e-3, rtol=1e-3):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.abs(a - b)
    tol = atol + rtol * np.abs(b)
    bad = float(np.mean(err > tol))
    worst = float(err.max()) if err.size else 0.0
    log(f"  parity {what}: max |diff| {worst:.2e}, frac over tol {bad:.2e}")
    if bad > 1e-3:
        raise RuntimeError(f"export parity failed for {what}: max |diff| {worst:.3g} ({bad:.2%} rows over tolerance)")
    return {"max_abs_diff": worst, "frac_over_tol": bad}


def runner_parity(m, sd, n=300):
    """spkm.Runner (the reference the C++ engine is tested against) vs fastpred on the first frames of
    the first session of `sd`."""
    a = int(sd.offsets[0])
    b = int(min(sd.offsets[1], a + n))
    r = spkm.Runner(m)
    hist, ref = [], []
    for i in range(a, b):
        r.begin_frame(hist[-spkm.MAX_HISTORY:])
        ref.append(r.predict(sd.X[i]))
        hist.append(sd.H[i])
        r.end_frame(sd.X[i], sd.H[i], sd.y[i])
    fp = fastpred.predict_split(m, sd.X[a:b], sd.H[a:b], sd.y[a:b], np.array([0, b - a]))
    return check_close(fp, np.array(ref), "fastpred vs spkm.Runner", atol=1e-4, rtol=1e-4)


def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "-"
        return f"{v:.{nd}f}"
    return str(v)


def report_md(rec):
    m, d, t = rec["model"], rec["dataset"], rec["training"]
    lines = [f"# {rec['run_id']}", "",
             f"- **Experiment:** `{rec['experiment']}` (group `{rec['group']}`){' — DRY RUN, untrained weights' if rec.get('dry_run') else ''}",
             f"- **Family / architecture:** {m['family']} — {m['architecture']}",
             f"- **Details:** {m.get('details', '-')}",
             f"- **Parameters (exact, as the engine counts them):** {m['num_parameters']:,}",
             f"- **Inputs:** {rec['features']['feature_set']} ({rec['features']['n_used']}/{rec['features']['n_features']}"
             f" features used, mask `{rec['features']['mask']}`), history_len {m.get('history_len', 0)}",
             f"- **Target:** {json.dumps(rec['target'])}; outputs {', '.join(m['outputs'])}",
             f"- **Estimated in-game cost:** {m['cost'].get('ingame_us_est')} µs/frame at 2 evals, "
             f"{m['cost'].get('ingame_us_est_worst')} µs with one eval per ladder level (budget 50 µs; estimate, measure in game)",
             f"- **Dataset:** {d['name']} — {d['num_frames']:,} frames, {d['size_bytes'] / 1e6:.1f} MB; "
             + ", ".join(f"{k} {v['sessions']} sessions / {v['frames']:,} frames"
                         for k, v in d['split']['detail'].items()) + " (split by session)",
             f"- **Training:** {t['time_s']:.1f} s on {t['hardware'].get('hostname')} "
             f"({t['hardware'].get('cpu_model')}; device {t['hardware'].get('device')}"
             f"{', ' + t['hardware']['gpu'] if t['hardware'].get('gpu') else ''}); seed {t['seed']}; "
             f"code {t['code_version']}",
             f"- **Exportable to the engine:** {m['exportable']}" + (f" ({m.get('export_note')})" if m.get('export_note') else ""),
             ""]
    keys = ["mae_ms", "pinball_p50", "pinball_p95", "pinball_p99", "coverage_p50", "coverage_p95", "coverage_p99",
            "crossing_rate", "pmiss_pr_auc", "pmiss_derived_pr_auc", "pmiss_brier", "dec_miss_precision",
            "dec_miss_recall"]
    lines += ["## Metrics", "", "| metric | val | test |", "|---|---|---|"]
    for k in keys:
        v1 = rec["metrics"].get("val", {}).get("overall", {}).get(k)
        v2 = rec["metrics"].get("test", {}).get("overall", {}).get(k)
        if v1 is not None or v2 is not None:
            lines.append(f"| {k} | {_fmt(v1)} | {_fmt(v2)} |")
    bys = rec["metrics"].get("test", {}).get("by_stress", {})
    if bys:
        lines += ["", "## Test by stress level", "", "| stress | frames | miss rate | MAE | pinball p95 | cov p95 | cov p99 |",
                  "|---|---|---|---|---|---|---|"]
        for s, v in bys.items():
            lines.append(f"| {s} | {v['frames']:,} | {_fmt(v.get('miss_rate'))} | {_fmt(v.get('mae_ms'))} | "
                         f"{_fmt(v.get('pinball_p95'))} | {_fmt(v.get('coverage_p95'))} | {_fmt(v.get('coverage_p99'))} |")
    for k in ("transitions", "spikes", "over_budget"):
        v = rec["metrics"].get("test", {}).get(k)
        if v and v.get("frames"):
            lines.append(f"\n**Test, {k}:** {v['frames']:,} frames, pinball p95 {_fmt(v.get('pinball_p95'))}, "
                         f"coverage p95 {_fmt(v.get('coverage_p95'))}, MAE {_fmt(v.get('mae_ms'))}")
    lines += ["", "Full metrics: the results record " + f"`results/runs/{rec['run_id']}.json`.", ""]
    return "\n".join(lines)


def finish(args, ds, cfg, *, group, experiment, family, run_id, model, architecture, details, feature_mask,
           target, train_time_s, train_info, preds, hp, device=None, native=None, export_note=None, ckpt_dir=None,
           res_dir=None, extra=None, worst_evals=8):
    """preds: {'val': {name: array}, 'test': {...}} in ms. model: spkm.Spkm or None (not exportable)."""
    budget = ds.budget_ms
    mets = {}
    for sp in ("val", "test"):
        if sp in preds:
            mets[sp] = M.evaluate(preds[sp], ds.split(sp).y, budget, ds.split(sp),
                                  extra_budgets=[b for b in cfg["controller"]["budgets_ms"] if abs(b - budget) > 1e-3])
    exportable = model is not None and export_note is None
    n_used = int(np.sum(feature_mask)) if feature_mask is not None else spkm.F
    rec = {
        "schema": RESULT_SCHEMA, "run_id": run_id, "group": group, "experiment": experiment, "family": family,
        "created_utc": utc_now(), "dry_run": bool(args.dry_run),
        "model": {"family": family, "architecture": architecture, "details": details,
                  "num_parameters": int(model.num_parameters()) if model is not None else int(train_info.get("params", 0)),
                  "history_len": model.history_len if model is not None else 0,
                  "outputs": [o[0] for o in model.outputs] if model is not None else sorted(preds.get("test", {})),
                  "exportable": exportable, "export_note": export_note,
                  "cost": cost.estimate(model, worst_evals=worst_evals) if model is not None else {}},
        "features": {"feature_set": spkm.FEATURE_SET, "feature_hash": spkm.feature_hash(), "n_features": spkm.F,
                     "n_used": n_used, "mask": hp.get("features", "all"),
                     "used": [n for n, k in zip(spkm.FEATURE_NAMES, feature_mask if feature_mask is not None
                                                else np.ones(spkm.F, bool)) if k]},
        "target": target.describe() if target is not None else {"name": "none", "type": "none"},
        "dataset": ds.info(),
        "training": {"seed": args.seed, "time_s": float(train_time_s), "hardware": hardware_info(device),
                     "code_version": code_version(), "hyperparams": hp,
                     **{k: v for k, v in train_info.items() if k not in ("history",)}},
        "metrics": mets,
        "headline": {sp: M.headline(v) for sp, v in mets.items()},
        "checkpoint": None,
    }
    if extra:
        rec.update(extra)
    if "history" in train_info:
        rec["training"]["curve"] = train_info["history"]

    ckpt_dir = Path(ckpt_dir)
    res_dir = Path(res_dir)
    if exportable:
        run_dir = ckpt_dir / run_id
        flat = {f"test_{k}": v for k, v in rec["headline"].get("test", {}).items()}
        flat.update({f"val_{k}": v for k, v in rec["headline"].get("val", {}).items()})
        flat["est_ingame_us"] = rec["model"]["cost"].get("ingame_us_est")
        model.info.update({"experiment": experiment, "code_version": rec["training"]["code_version"]})
        manifest = spkm.write_checkpoint(str(run_dir), model, dataset=_manifest_dataset(rec["dataset"]),
                                         metrics=_finite(flat), architecture=architecture)
        manifest["model"]["details"] = details
        manifest["experiment"] = {"name": experiment, "group": group, "dry_run": bool(args.dry_run)}
        manifest["training"] = {k: rec["training"][k] for k in ("seed", "time_s", "hardware", "code_version",
                                                                 "hyperparams")}
        manifest["target"] = rec["target"]
        manifest["cost_estimate"] = rec["model"]["cost"]
        write_json(run_dir / "manifest.json", manifest)
        if native:
            (run_dir / "native").mkdir(exist_ok=True)
            for name, writer in native.items():
                writer(run_dir / "native" / name)
        (run_dir / "REPORT.md").write_text(report_md(rec))
        errs, _ = _validator().check_run(run_dir, require_done=False)
        rec["checkpoint"] = {"path": rel(run_dir), "validator_errors": errs,
                             "spkm_eval": spkm_eval_check(run_dir / "model.spkm")}
        if errs:
            log(f"VALIDATION FAILED for {run_id}: {errs}")
        else:
            (run_dir / "DONE").write_text("")
            log(f"checkpoint {rel(run_dir)} validated, DONE written")
    else:
        log(f"{run_id}: not exported ({export_note or 'no engine model'}); results only")
    write_json(res_dir / "runs" / f"{run_id}.json", rec)
    if "test" in preds:
        (res_dir / "preds").mkdir(parents=True, exist_ok=True)
        np.savez(res_dir / "preds" / f"{run_id}.npz", **{k: np.asarray(v, np.float32) for k, v in preds["test"].items()})
    if exportable and rec["checkpoint"]["validator_errors"]:
        raise RuntimeError(f"checkpoint {run_id} failed validation: {rec['checkpoint']['validator_errors']}")
    return rec


def _finite(d):
    return {k: v for k, v in d.items() if isinstance(v, (int, float)) and v is not None and np.isfinite(v)}


def _manifest_dataset(info):
    d = dict(info)
    d["num_frames"] = int(d["num_frames"])
    d["size_bytes"] = max(1, int(d["size_bytes"]))
    return d
