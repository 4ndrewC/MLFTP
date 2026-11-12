"""Open-loop metrics (LIT_REVIEW section 4.5). Pure numpy, no fitting.

    evaluate(pred, y, budget_ms, split=sd)   -> nested dict: overall, by_stress, by_loop, by_core,
                                                transitions, spikes, over_budget
where pred maps output names (t_pNN_ms, t_mean_ms, p_miss) to arrays of length N.
"""
import numpy as np

import data as D


def qname(q):
    return f"t_p{int(round(q * 100))}_ms"


def qlevel(name):
    return int(name[3:-3]) / 100


def pinball(y, q_pred, tau):
    d = np.asarray(y, np.float64) - np.asarray(q_pred, np.float64)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def coverage(y, q_pred):
    return float(np.mean(np.asarray(y) <= np.asarray(q_pred)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, np.float64) - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, np.float64) - p) ** 2)))


def ape(y, p):
    y = np.asarray(y, np.float64)
    a = np.abs(p - y) / np.maximum(y, 1e-6)
    return float(np.median(a)), float(np.mean(a))


def crossing_rate(Q):
    """Fraction of rows where the quantile columns (ascending tau) are not non-decreasing."""
    Q = np.asarray(Q)
    if Q.shape[1] < 2:
        return 0.0
    return float(np.mean(np.any(np.diff(Q, axis=1) < -1e-9, axis=1)))


def average_precision(labels, scores):
    """PR-AUC as average precision (step function, same definition as sklearn)."""
    labels = np.asarray(labels, bool)
    P = labels.sum()
    if P == 0 or P == len(labels):
        return float("nan")
    order = np.argsort(-np.asarray(scores, np.float64), kind="mergesort")
    s = np.asarray(scores, np.float64)[order]
    l = labels[order]
    tp = np.cumsum(l)
    fp = np.cumsum(~l)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # threshold boundaries (ties grouped)
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp)
    rec = tp / P
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def prf(labels, pred_pos):
    labels = np.asarray(labels, bool)
    pred_pos = np.asarray(pred_pos, bool)
    tp = int(np.sum(labels & pred_pos))
    fp = int(np.sum(~labels & pred_pos))
    fn = int(np.sum(labels & ~pred_pos))
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if tp else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def brier(labels, prob):
    return float(np.mean((np.asarray(prob, np.float64) - np.asarray(labels, np.float64)) ** 2))


def logloss(labels, prob, eps=1e-7):
    p = np.clip(np.asarray(prob, np.float64), eps, 1 - eps)
    l = np.asarray(labels, np.float64)
    return float(-np.mean(l * np.log(p) + (1 - l) * np.log(1 - p)))


def pmiss_from_quantiles(Q, taus, budget):
    """P(t > budget) from predicted quantiles: piecewise-linear CDF between the quantile points
    (from (0, 0) up to the lowest one) and an exponential tail through the top two quantiles.
    Crossing quantiles are sorted first. Q: (N, k) ascending taus."""
    Q = np.sort(np.asarray(Q, np.float64), axis=1)
    taus = np.asarray(taus, np.float64)
    N, k = Q.shape
    B = float(budget)
    F = np.empty(N)
    # below the first quantile: linear from (0, 0)
    lo = B <= Q[:, 0]
    F[lo] = taus[0] * B / np.maximum(Q[lo, 0], 1e-9)
    done = lo.copy()
    for i in range(k - 1):
        m = ~done & (B <= Q[:, i + 1])
        w = (B - Q[m, i]) / np.maximum(Q[m, i + 1] - Q[m, i], 1e-9)
        F[m] = taus[i] + w * (taus[i + 1] - taus[i])
        done |= m
    m = ~done  # above the top quantile: exponential tail
    if k >= 2:
        scale = (Q[m, -1] - Q[m, -2]) / np.log((1 - taus[-2]) / (1 - taus[-1]))
    else:
        scale = np.maximum(Q[m, -1] * 0.1, 1e-3)
    F[m] = 1 - (1 - taus[-1]) * np.exp(-(B - Q[m, -1]) / np.maximum(scale, 1e-6))
    return np.clip(1 - F, 0.0, 1.0)


def _quantile_cols(pred):
    names = sorted([n for n in pred if n.startswith("t_p")], key=qlevel)
    return names


def core_metrics(pred, y, budget, idx=None):
    """Metrics for one subset of rows."""
    y = np.asarray(y, np.float64)
    sel = slice(None) if idx is None else idx
    yy = y[sel]
    out = {"frames": int(len(yy))}
    if len(yy) == 0:
        return out
    miss = yy > budget
    out["miss_rate"] = float(miss.mean())
    qn = _quantile_cols(pred)
    for n in qn:
        tau = qlevel(n)
        p = np.asarray(pred[n], np.float64)[sel]
        key = n[2:-3]  # p50
        out[f"pinball_{key}"] = pinball(yy, p, tau)
        out[f"coverage_{key}"] = coverage(yy, p)
    point = pred.get("t_p50_ms", pred.get("t_mean_ms"))
    if point is not None:
        p = np.asarray(point, np.float64)[sel]
        out["mae_ms"] = mae(yy, p)
        out["rmse_ms"] = rmse(yy, p)
        out["median_ape"], out["mean_ape"] = ape(yy, p)
        out["bias_ms"] = float(np.mean(p - yy))
    if len(qn) >= 2:
        Q = np.stack([np.asarray(pred[n])[sel] for n in qn], 1)
        out["crossing_rate"] = crossing_rate(Q)
    for src, key in (("p_miss", "pmiss"), ("_pmiss_derived", "pmiss_derived")):
        if src in pred:
            pm = np.asarray(pred[src], np.float64)[sel]
            out[f"{key}_pr_auc"] = average_precision(miss, pm)
            out[f"{key}_brier"] = brier(miss, pm)
            out[f"{key}_logloss"] = logloss(miss, pm)
            r = prf(miss, pm >= 0.5)
            out[f"{key}_precision"], out[f"{key}_recall"], out[f"{key}_f1"] = r["precision"], r["recall"], r["f1"]
    # the controller's view: does the decision output flag the misses?
    dec = pred.get("t_p95_ms", point)
    if dec is not None:
        r = prf(miss, np.asarray(dec)[sel] > budget)
        out["dec_miss_precision"], out["dec_miss_recall"] = r["precision"], r["recall"]
    return out


def evaluate(pred, y, budget, split=None, extra_budgets=(), groups=True):
    """Full metric set. `split` (data.SplitData) enables the breakdowns."""
    pred = dict(pred)
    qn = _quantile_cols(pred)
    if len(qn) >= 1:
        Q = np.stack([pred[n] for n in qn], 1)
        pred["_pmiss_derived"] = pmiss_from_quantiles(Q, [qlevel(n) for n in qn], budget)
    res = {"budget_ms": budget, "overall": core_metrics(pred, y, budget)}
    for b in extra_budgets:
        sub = {k: v for k, v in pred.items() if k != "p_miss" and k != "_pmiss_derived"}
        if qn:
            sub["_pmiss_derived"] = pmiss_from_quantiles(np.stack([pred[n] for n in qn], 1),
                                                         [qlevel(n) for n in qn], b)
        res[f"budget_{b:.2f}"] = core_metrics(sub, y, b)
    if split is None or not groups:
        return res
    for key, codes, names in (("by_stress", split.stress, D.STRESS), ("by_loop", split.loop, D.LOOPS),
                              ("by_core", split.core, D.CORES)):
        res[key] = {}
        for c, n in enumerate(names):
            idx = np.nonzero(codes == c)[0]
            if len(idx):
                res[key][n] = core_metrics(pred, y, budget, idx)
    tm = D.transition_mask(split.aux["flags"], split.offsets, radius=5)
    res["transitions"] = core_metrics(pred, y, budget, np.nonzero(tm)[0])
    sm = D.spike_mask(y, split.offsets)
    res["spikes"] = core_metrics(pred, y, budget, np.nonzero(sm)[0])
    res["over_budget"] = core_metrics(pred, y, budget, np.nonzero(np.asarray(y) > budget)[0])
    return res


def headline(res):
    """Flat summary of the most important numbers (for manifests and tables)."""
    o = res["overall"]
    keys = ["mae_ms", "rmse_ms", "median_ape", "pinball_p50", "pinball_p95", "pinball_p99", "coverage_p50",
            "coverage_p95", "coverage_p99", "crossing_rate", "pmiss_pr_auc", "pmiss_brier", "pmiss_precision",
            "pmiss_recall", "pmiss_derived_pr_auc", "pmiss_derived_brier", "dec_miss_precision", "dec_miss_recall"]
    h = {k: o[k] for k in keys if k in o}
    for k in ("transitions", "spikes", "over_budget"):
        if k in res and "pinball_p95" in res[k]:
            h[f"{k}_pinball_p95"] = res[k]["pinball_p95"]
            h[f"{k}_coverage_p95"] = res[k].get("coverage_p95")
    return h
