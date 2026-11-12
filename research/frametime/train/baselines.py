#!/usr/bin/env python3
"""Non-ML baselines (LIT_REVIEW section 4.3), evaluated open-loop like every model.

Point predictors v (all computed from frames < k, i.e. what the engine knows):
    last      t_{k-1}                       (feature t_prev_ms)
    ema01/03/05, ema02 (feature t_ema_ms)   EMA of t_total
    max8      max of the last 8 frames      (feature t_max8_ms)
    mean3     mean of the last 3 frames     (ENTIRE's OLP baseline)
    nlms8     online normalised-LMS AR(8) on the last 8 frame times (per session)
    naive     steps-aware analytic estimate (feature naive_ms, counterfactual over quality)
    tref      fingerprint reference t_ref_ms (feature set v2; skipped when the logs have no fingerprint)

Quantiles: v + c_q (additive) or v * r_q (multiplicative), with c_q / r_q the train-set residual
quantiles, whichever is better on val (conformal-style calibration; no model fitting beyond that).
p_miss: derived from those quantiles, plus a 1-D logistic calibration on v for the exported form.
Baselines that are features are exported as `linear` checkpoints so they can run in the game.
The reactive (closed-loop) baseline lives in controllers.py.
"""
import argparse
import sys

import numpy as np

import common
from common import Timer, log, make_run_id
import data as Dt
import metrics as M
import runlib
import spkm

FEATURE_BASELINES = {"last": "t_prev_ms", "ema02": "t_ema_ms", "max8": "t_max8_ms", "naive": "naive_ms",
                     "tref": "t_ref_ms"}


def nlms(y, offsets, order=8, mu=0.05, eps=1e-3):
    """Online NLMS AR(order) per session: predict from the previous `order` frame times, then update."""
    out = np.zeros(len(y))
    y = np.asarray(y, np.float64)
    for a, b in zip(offsets[:-1], offsets[1:]):
        w = np.zeros(order + 1)
        w[0] = 1.0  # start as "last frame"
        buf = np.zeros(order)
        for i in range(a, b):
            u = np.r_[buf, 1.0]
            p = w @ u
            out[i] = p if i > a else 0.0
            e = y[i] - p
            w += mu * e * u / (eps + u @ u)
            buf = np.r_[y[i], buf[:-1]]
    return out


def mean_last(y, offsets, k=3):
    s = np.zeros(len(y))
    c = np.zeros(len(y))
    for j in range(1, k + 1):
        lj = Dt.lag(y, offsets, j, fill=np.nan)
        ok = np.isfinite(lj)
        s += np.where(ok, lj, 0)
        c += ok
    return np.where(c > 0, s / np.maximum(c, 1), 0.0)


def point(name, sd, alpha=None):
    if name in FEATURE_BASELINES:
        return sd.X[:, spkm.FEATURE_NAMES.index(FEATURE_BASELINES[name])].astype(np.float64)
    if name.startswith("ema"):
        return Dt.ema_prev(sd.y, sd.offsets, int(name[3:]) / 10)
    if name == "mean3":
        return mean_last(sd.y, sd.offsets, 3)
    if name == "nlms8":
        return nlms(sd.y, sd.offsets, 8)
    raise ValueError(name)


def calibrate(y, v, quantiles, mode):
    if mode == "add":
        return np.quantile(np.asarray(y, np.float64) - v, quantiles)
    ok = v > 1e-3
    return np.quantile(np.asarray(y, np.float64)[ok] / v[ok], quantiles)


def apply(v, c, mode):
    return v[:, None] + c[None, :] if mode == "add" else v[:, None] * c[None, :]


def logistic_1d(v, lab, iters=30):
    """P(miss) = sigmoid(a*v + b) by Newton's method (2 parameters)."""
    common.require_fitting("baselines.logistic_1d")
    Xd = np.stack([v, np.ones_like(v)], 1)
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Xd @ w)))
        g = Xd.T @ (p - lab)
        Hm = (Xd * (p * (1 - p))[:, None]).T @ Xd + 1e-6 * np.eye(2)
        w -= np.linalg.solve(Hm, g)
    return w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    common.add_common_args(ap)
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    ckpt_dir, res_dir = common.output_dirs(args)
    ds = Dt.Dataset(cfg, args.data, args.cache_dir, holdout=args.holdout, max_sessions=args.max_sessions, workers=args.workers,
                    dry_run=args.dry_run, dry_frames=args.dry_frames)
    qs = np.array(cfg["quantiles"])
    budget = ds.budget_ms
    names = args.only or ["last", "ema01", "ema02", "ema03", "ema05", "max8", "mean3", "nlms8", "naive", "tref"]
    names = [n for n in names if n not in FEATURE_BASELINES or FEATURE_BASELINES[n] in spkm.FEATURE_NAMES]
    tr, va, te = ds.split("train"), ds.split("val"), ds.split("test")
    for name in names:
        t = Timer()
        v = {sp: point(name, sd) for sp, sd in (("train", tr), ("val", va), ("test", te))}
        if np.mean(v["train"] > 1e-3) < 0.5:
            log(f"skip {name}: zero on most rows (e.g. no fingerprint in these logs)")
            continue
        best = None
        for mode in ("add", "mul"):
            c = calibrate(tr.y, v["train"], qs, mode) if not args.dry_run else (np.zeros(len(qs)) if mode == "add"
                                                                                 else np.ones(len(qs)))
            Qv = apply(v["val"], c, mode)
            score = sum(M.pinball(va.y, Qv[:, i], q) for i, q in enumerate(qs))
            if best is None or score < best[0]:
                best = (score, mode, c)
        _, mode, c = best
        ab = logistic_1d(v["train"], (tr.y > budget).astype(np.float64)) if not args.dry_run else np.zeros(2)
        tt = t.s()
        fam = "linear"
        label = f"baseline-{name}" + (f"@{args.holdout}" if args.holdout else "")
        run_id = make_run_id(fam if name in FEATURE_BASELINES else "baseline", label)
        preds = {}
        model, note = None, "not a feature of the engine's set: evaluated offline only"
        if name in FEATURE_BASELINES:
            fi = spkm.FEATURE_NAMES.index(FEATURE_BASELINES[name])
            outs, b = [], []
            if mode == "add":
                outs = [(M.qname(q), "identity", FEATURE_BASELINES[name]) for q in qs]
                b = list(c)
            elif "mulexp" in getattr(spkm, "TRANSFORMS", ()):
                outs = [(M.qname(q), "mulexp", FEATURE_BASELINES[name]) for q in qs]
                b = list(np.log(np.maximum(c, 1e-6)))
            if outs:
                W = np.zeros((len(outs) + 1, spkm.F))
                W[-1, fi] = ab[0]
                outs.append(("p_miss", "sigmoid", "-"))
                b.append(ab[1])
                model = spkm.linear(W, np.array(b), outs, budget_ms=budget,
                                    info={"run_id": run_id, "experiment": label})
                note = None
                for sp, sd in (("val", va), ("test", te)):
                    preds[sp] = runlib.predict_model(model, sd)
                Qt = apply(v["val"], c, mode)
                runlib.check_close(preds["val"][M.qname(qs[1])], Qt[:, 1], f"baseline {name} export")
        if not preds:
            for sp, vv in (("val", v["val"]), ("test", v["test"])):
                Q = apply(vv, c, mode)
                preds[sp] = {M.qname(q): Q[:, i] for i, q in enumerate(qs)}
                preds[sp]["p_miss"] = 1 / (1 + np.exp(-(ab[0] * vv + ab[1])))
        mask = np.zeros(spkm.F, bool)
        if name in FEATURE_BASELINES:
            mask[spkm.FEATURE_NAMES.index(FEATURE_BASELINES[name])] = True
        info = {"calibration": mode, "offsets": c, "logistic": ab, "params": len(qs) + 2,
                "effective_params": len(qs) + 2}
        arch = f"baseline {name}: quantiles = v {'+' if mode == 'add' else 'x'} train residual quantiles; " \
               f"p_miss = logistic(v)"
        runlib.finish(args, ds, cfg, group="baselines" + ("-transfer" if args.holdout else ""), experiment=label,
                      family=fam if model else
                      "baseline", run_id=run_id, model=model, architecture=arch,
                      details=f"point predictor {name}; calibration {mode}; offsets {np.round(c, 4).tolist()}",
                      feature_mask=mask, target=None, train_time_s=tt, train_info=info, preds=preds,
                      hp={"features": FEATURE_BASELINES.get(name, "-"), "calibration": mode}, export_note=note,
                      ckpt_dir=ckpt_dir, res_dir=res_dir, worst_evals=len(ds.ladder))
        h = M.headline(M.evaluate(preds["test"], te.y, budget))
        log(f"{name:6s} ({mode}): test MAE {h.get('mae_ms', float('nan')):.4f} pinball p95 "
            f"{h.get('pinball_p95', float('nan')):.4f} cov95 {h.get('coverage_p95', float('nan')):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
