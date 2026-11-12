#!/usr/bin/env python3
"""Which inputs and which quality levers to keep ("PCA or something").

    importance.py [--checkpoints RUN_DIR ...] [--dry-run --out-root SCRATCH]

1. Redundancy: PCA on standardised train features (explained variance, loadings) and Spearman
   correlation clustering (|rho| >= 0.9 -> same cluster).
2. Importance: permutation importance on test (single features and whole correlation clusters,
   pinball p95/p50 increase) for a reference LightGBM quantile model and any stateless checkpoints
   passed in; TreeSHAP (LightGBM pred_contrib) mean |contribution|.
3. Ablations: retrain the reference GBT without each feature group (config ablation_groups).
4. Levers: predicted cost of every level of every lever (others at best quality) from counterfactual
   rows, next to the analytic simulator; all lever combinations on a small bench -> a suggested ladder.
5. A recommendation table (keep / redundant / drop candidate) for features and levers.

Outputs results/importance/<run_id>.{json,md}.
"""
import argparse
import itertools
import sys

import numpy as np

import common
from common import Timer, log, make_run_id, rel, write_json
import controllers as C
import data as Dt
import fastpred
import metrics as M
import spkm


# ---------------------------------------------------------------- redundancy

def pca(Xs):
    U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    var = S ** 2
    ratio = var / var.sum()
    cum = np.cumsum(ratio)
    comps = {f"n_for_{int(p * 100)}pct": int(np.searchsorted(cum, p) + 1) for p in (0.9, 0.95, 0.99)}
    top = []
    for k in range(min(6, len(S))):
        order = np.argsort(-np.abs(Vt[k]))[:6]
        top.append({"pc": k + 1, "explained": float(ratio[k]),
                    "top_loadings": [(spkm.FEATURE_NAMES[i], float(Vt[k, i])) for i in order]})
    main_pc = np.argmax(np.abs(Vt[:max(1, comps["n_for_95pct"])]) * np.sqrt(var[:max(1, comps["n_for_95pct"])])[:, None],
                        axis=0) + 1
    return {"explained_ratio": ratio.tolist(), "cumulative": cum.tolist(), **comps, "top": top,
            "feature_main_pc": {n: int(main_pc[i]) for i, n in enumerate(spkm.FEATURE_NAMES)}}


def spearman(X):
    from scipy.stats import rankdata
    R = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    sd = R.std(0)
    const = sd < 1e-12
    Rz = (R - R.mean(0)) / np.where(const, 1, sd)
    rho = (Rz.T @ Rz) / len(R)
    rho[const, :] = 0
    rho[:, const] = 0
    np.fill_diagonal(rho, 1)
    return np.clip(rho, -1, 1), const


def clusters(rho, threshold=0.9):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    Dm = 1 - np.abs(rho)
    np.fill_diagonal(Dm, 0)
    Z = linkage(squareform(Dm, checks=False), "average")
    return fcluster(Z, t=1 - threshold, criterion="distance")


# ---------------------------------------------------------------- reference model + importance

class RefModel:
    """p50/p95 predictor: LightGBM boosters (trained here) or an exported stateless .spkm."""

    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def __call__(self, X):
        return self.fn(X)


def lgb_reference(tr, va, keep, hp, seed, threads, subsample=None, rng=None):
    import trees as TR
    rows = np.arange(tr.N)
    if subsample and tr.N > subsample:
        rows = np.sort(rng.choice(tr.N, subsample, replace=False))
    Xtr = np.ascontiguousarray(tr.X[rows][:, keep])
    Xva = np.ascontiguousarray(va.X[:, keep])
    bs = []
    for q in (0.5, 0.95):
        b, _ = TR.fit_lgb(Xtr, tr.y[rows], Xva, va.y, "quantile", q, hp, seed, threads, hp["n_rounds"])
        bs.append(b)
    return bs


def lgb_fn(bs, keep):
    return lambda X: np.stack([b.predict(X[:, keep], num_iteration=b.best_iteration or None) for b in bs], 1)


def spkm_fn(m):
    names = [o[0] for o in m.outputs]
    i50 = names.index("t_p50_ms") if "t_p50_ms" in names else 0
    i95 = names.index("t_p95_ms") if "t_p95_ms" in names else i50
    P = fastpred.StepPredictor(m, 1)

    def f(X):
        Y = np.concatenate([P.predict(X[a:a + 65536][:, None, :])[:, 0] for a in range(0, len(X), 65536)])
        return Y[:, [i50, i95]]
    return f


def score(y, Q):
    return M.pinball(y, Q[:, 0], 0.5), M.pinball(y, Q[:, 1], 0.95)


def permutation(model, X, y, groups, rng, repeats=2):
    """groups: {label: [feature indices]} -> {label: (d_pinball_p50, d_pinball_p95)}."""
    b50, b95 = score(y, model(X))
    out = {}
    for label, idx in groups.items():
        d50 = d95 = 0.0
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, idx] = X[perm][:, idx]
            s50, s95 = score(y, model(Xp))
            d50 += (s50 - b50) / repeats
            d95 += (s95 - b95) / repeats
        out[label] = (d50, d95)
    return out, (b50, b95)


# ---------------------------------------------------------------- levers

def lever_costs(ds, cfg, model, rows_step=20, split="test"):
    """Predicted (p50, p95) and simulated mean cost per level of each lever, others at level 0."""
    sd = ds.split(split)
    levers = cfg["levers"]
    sel = np.arange(0, sd.N, rows_step)
    out = []
    for i, lv in enumerate(levers):
        per = []
        for v in range(lv["levels"]):
            q = tuple(v if j == i else 0 for j in range(len(levers)))
            X = ds.counterfactual_split(split, q)[sel]
            Q = model(X)
            tsim = np.concatenate([C.simulate_costs(s, cfg, [q])[0] for s in sd.sessions])[sel]
            by = {}
            for c, name in enumerate(Dt.STRESS):
                m = sd.stress[sel] == c
                if m.any():
                    by[name] = float(Q[m, 0].mean())
            per.append({"level": v, "value": lv["values"][v] if v < len(lv.get("values", [])) else None,
                        "pred_p50_ms": float(Q[:, 0].mean()), "pred_p95_ms": float(Q[:, 1].mean()),
                        "sim_ms": float(tsim.mean()), "pred_p50_by_stress": by})
        base = per[0]["pred_p50_ms"]
        rng_ms = base - min(p["pred_p50_ms"] for p in per)
        share = rng_ms / max(base, 1e-9)
        sim_share = (per[0]["sim_ms"] - min(p["sim_ms"] for p in per)) / max(per[0]["sim_ms"], 1e-9)
        rec = ("strong lever: keep" if share >= 0.2 else "medium lever: keep" if share >= 0.05
               else "weak lever: drop, merge into another step, or replace")
        out.append({"lever": lv["name"], "field": lv["field"], "levels": per, "pred_range_ms": rng_ms,
                    "pred_share_of_best_quality_cost": share, "sim_share": sim_share, "recommendation": rec})
    return out


def paired_lever_truth(ds, cfg, model, splits=("val", "test")):
    """Ground truth from sessions that play the same game at two fixed qualities (laptop-v1 pairs,
    laptop-v2 lever sweeps): measured cost change vs the model's counterfactual prediction vs the simulator."""
    rows = []
    for sp in splits:
        if sp not in ds.splits:
            continue
        for key, ss in C._comparable_groups(ds.split(sp)).items():
            base = min(ss, key=lambda s: sum(C._quality(s)))
            for B in ss:
                qA, qB = C._quality(base), C._quality(B)
                if B is base or qA == qB:
                    continue
                ia, jb = C.align(base, B)
                if len(ia) < 50:
                    continue
                XA = ds.counterfactual(base, qA)[ia]
                XB = ds.counterfactual(base, qB)[ia]
                pa, pb = model(XA)[:, 0], model(XB)[:, 0]
                ya, yb = base.a["y"][ia].astype(np.float64), B.a["y"][jb].astype(np.float64)
                sim = C.simulate_costs(base, cfg, [qB])[0][ia]
                rows.append({"split": sp, "group": str(key[0]), "from": base.id, "to": B.id, "q_from": list(qA),
                             "q_to": list(qB), "frames": int(len(ia)), "actual_from_ms": float(ya.mean()),
                             "actual_delta_ms": float(yb.mean() - ya.mean()),
                             "pred_delta_ms": float(pb.mean() - pa.mean()), "sim_delta_ms": float(sim.mean() - ya.mean())})
    return rows


def ladder_search(ds, cfg, model, n_sessions=6, frames=4000, split="test", weights=None):
    """Predicted mean p50 for every lever combination on a small bench (shortened test sessions), then a
    greedy ladder: from best quality, each rung lowers one lever by one level, choosing the largest predicted
    saving per unit of quality lost (weights per lever level, default 1)."""
    import features
    import framelog
    sd = ds.split(split)
    sess = sorted(sd.sessions, key=lambda s: -s.n)[:n_sessions]
    levers = cfg["levers"]
    weights = weights or [1.0] * len(levers)
    recs = []
    for s in sess:
        h, r = framelog.read_log(s.path)
        recs.append((h, r[:frames]))
    combos = list(itertools.product(*[range(lv["levels"]) for lv in levers]))
    cost = {}
    for q in combos:
        X = np.concatenate([features.compute_features(r, h, quality=q) for h, r in recs])
        cost[q] = float(model(X)[:, 0].mean())
    cur = tuple(0 for _ in levers)
    ladder = [{"q": list(cur), "pred_p50_ms": cost[cur]}]
    while True:
        moves = []
        for i, lv in enumerate(levers):
            if cur[i] + 1 < lv["levels"]:
                nq = tuple(v + (1 if j == i else 0) for j, v in enumerate(cur))
                moves.append(((cost[cur] - cost[nq]) / weights[i], nq))
        if not moves:
            break
        _, cur = max(moves)
        ladder.append({"q": list(cur), "pred_p50_ms": cost[cur]})
    current = [{"q": list(q), "pred_p50_ms": cost.get(tuple(q))} for q in ds.ladder]
    return {"bench_sessions": [s.id for s in sess], "frames_per_session": frames, "n_combinations": len(combos),
            "suggested_ladder": ladder, "current_ladder": current,
            "all": [{"q": list(q), "pred_p50_ms": v} for q, v in sorted(cost.items(), key=lambda kv: kv[1])]}


# ---------------------------------------------------------------- recommendation

def recommend(names, std0, clus, perm, shap, groups_of, quality_feats):
    top95 = max([v[1] for v in perm.values()] + [1e-12])
    topshap = max(list(shap.values()) + [1e-12]) if shap else 1e-12
    score = {n: max(perm.get(n, (0, 0))[1] / top95, (shap.get(n, 0) / topshap) if shap else 0) for n in names}
    rep = {}
    for n in names:
        c = clus[n]
        if c not in rep or score[n] > score[rep[c]]:
            rep[c] = n
    rows = []
    for i, n in enumerate(names):
        c = clus[n]
        if std0[i]:
            r = "drop on this dataset (constant); re-check with data where it varies"
        elif n in quality_feats:
            r = "keep (the controller's counterfactual input)"
        elif rep[c] != n and score[n] < 0.25 * score[rep[c]]:
            r = f"redundant with {rep[c]} (same correlation cluster)"
        elif score[n] < 0.01:
            r = "drop candidate (no measurable importance)"
        else:
            r = "keep"
        rows.append({"feature": n, "group": groups_of.get(n, "-"), "cluster": int(c), "cluster_rep": rep[c],
                     "score": score[n], "perm_d_pinball_p95": perm.get(n, (0, 0))[1],
                     "perm_d_pinball_p50": perm.get(n, (0, 0))[0], "shap_mean_abs": shap.get(n) if shap else None,
                     "recommendation": r})
    return rows


def to_md(out):
    f = lambda v, nd=4: "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
    L = [f"# Feature and lever importance {out['run_id']}", "",
         f"Dataset {out['dataset']['name']}, reference model: {out['reference']}."
         + (" DRY RUN (untrained reference)." if out["dry_run"] else ""), "",
         f"PCA: {out['pca']['n_for_90pct']} / {out['pca']['n_for_95pct']} / {out['pca']['n_for_99pct']} components "
         f"explain 90 / 95 / 99% of the standardised feature variance ({len(spkm.FEATURE_NAMES)} features).", "",
         "## Features", "", "| feature | group | cluster (rep) | perm Δpinball p95 | perm Δpinball p50 | SHAP mean abs | "
         "recommendation |", "|---|---|---|---|---|---|---|"]
    for r in sorted(out["features"], key=lambda r: -r["score"]):
        L.append(f"| {r['feature']} | {r['group']} | {r['cluster']} ({r['cluster_rep']}) | {f(r['perm_d_pinball_p95'])} | "
                 f"{f(r['perm_d_pinball_p50'])} | {f(r['shap_mean_abs'])} | {r['recommendation']} |")
    L += ["", "## Correlation clusters (|Spearman rho| >= 0.9) — grouped permutation", "",
          "| cluster | features | Δpinball p95 | Δpinball p50 |", "|---|---|---|---|"]
    for c in out["cluster_perm"]:
        L.append(f"| {c['cluster']} | {', '.join(c['features'])} | {f(c['d_p95'])} | {f(c['d_p50'])} |")
    L += ["", "## Group ablations (reference GBT retrained without the group; test pinball increase)", "",
          "| dropped group | n features | Δpinball p95 | Δpinball p50 |", "|---|---|---|---|"]
    for a in out["ablations"]:
        L.append(f"| {a['group']} | {a['n_features']} | {f(a['d_pinball_p95'])} | {f(a['d_pinball_p50'])} |")
    L += ["", "## Quality levers (others at best quality; mean over test frames)", "",
          "| lever | level: predicted p50 ms (simulator ms) | range | share of best-quality cost | recommendation |",
          "|---|---|---|---|---|"]
    for lv in out["levers"]:
        lev = "; ".join(f"{p['level']}: {p['pred_p50_ms']:.3f} ({p['sim_ms']:.3f})" for p in lv["levels"])
        L.append(f"| {lv['lever']} | {lev} | {lv['pred_range_ms']:.3f} ms | {lv['pred_share_of_best_quality_cost']:.1%} "
                 f"(sim {lv['sim_share']:.1%}) | {lv['recommendation']} |")
    if out.get("paired_truth"):
        L += ["", "### Check against paired sessions (same game, two fixed qualities; val/test)", "",
              "| from → to | frames | actual from ms | actual Δ ms | predicted Δ ms | simulator Δ ms |", "|---|---|---|---|---|---|"]
        for p in out["paired_truth"]:
            L.append(f"| {p['q_from']} → {p['q_to']} ({p['to']}) | {p['frames']:,} | {p['actual_from_ms']:.3f} | "
                     f"{p['actual_delta_ms']:+.3f} | {p['pred_delta_ms']:+.3f} | {p['sim_delta_ms']:+.3f} |")
    ls = out.get("ladder")
    if ls:
        L += ["", "## Ladder", "", f"Bench: {len(ls['bench_sessions'])} test sessions x {ls['frames_per_session']} frames, "
              f"{ls['n_combinations']} lever combinations.", "", "| rung | current ladder (pred p50 ms) | suggested (pred p50 ms) |",
              "|---|---|---|"]
        for i in range(max(len(ls["current_ladder"]), len(ls["suggested_ladder"]))):
            a = ls["current_ladder"][i] if i < len(ls["current_ladder"]) else None
            b = ls["suggested_ladder"][i] if i < len(ls["suggested_ladder"]) else None
            L.append(f"| L{i} | {a['q'] if a else '-'} ({f(a['pred_p50_ms'], 3) if a else '-'}) | "
                     f"{b['q'] if b else '-'} ({f(b['pred_p50_ms'], 3) if b else '-'}) |")
    L += ["", "## PCA top components", ""]
    for t in out["pca"]["top"]:
        L.append(f"- PC{t['pc']} ({t['explained']:.1%}): " + ", ".join(f"{n} {v:+.2f}" for n, v in t["top_loadings"]))
    for extra in out.get("checkpoint_perm", []):
        L += ["", f"## Permutation importance for checkpoint {extra['run']} (top 15 by Δpinball p95)", ""]
        for n, (d50, d95) in sorted(extra["perm"].items(), key=lambda kv: -kv[1][1])[:15]:
            L.append(f"- {n}: Δp95 {d95:.4f}, Δp50 {d50:.4f}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="*", default=[], help="stateless checkpoints to also permute")
    ap.add_argument("--perm-rows", type=int, default=50000)
    ap.add_argument("--ablation-subsample", type=int, default=300000)
    ap.add_argument("--no-ladder", action="store_true")
    common.add_common_args(ap)
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    _, res_dir = common.output_dirs(args)
    rng = np.random.default_rng(args.seed)
    t = Timer()
    ds = Dt.Dataset(cfg, args.data, args.cache_dir, holdout=args.holdout, max_sessions=args.max_sessions, workers=args.workers,
                    dry_run=args.dry_run, dry_frames=args.dry_frames)
    tr, va, te = ds.split("train"), ds.split("val"), ds.split("test")
    names = spkm.FEATURE_NAMES
    sub = rng.choice(tr.N, min(tr.N, 200000), replace=False)
    mean, std = Dt.norm_stats(tr.X)
    Xs = Dt.normalise(tr.X[sub], mean, std).astype(np.float64)
    std0 = tr.X.astype(np.float64).std(0) < 1e-9
    log("PCA + correlation clusters")
    pc = pca(Xs)
    rho, const = spearman(tr.X[sub[:100000]].astype(np.float64))
    cl = clusters(rho)
    clus = {n: int(c) for n, c in zip(names, cl)}
    groups_of = {}
    for g, fs in cfg.get("ablation_groups", {}).items():
        for n in fs:
            groups_of.setdefault(n, g)

    hp = dict(n_rounds=200, learning_rate=0.1, num_leaves=15, max_depth=4, min_data_in_leaf=200, early_stopping=30)
    keep = np.arange(spkm.F)
    if args.dry_run:
        import trees as TR
        dumps = [TR.random_dump(rng, 5, 4, spkm.F) for _ in range(2)]
        m = TR.export_gbt(dumps, [("t_p50_ms", "identity", "-"), ("t_p95_ms", "identity", "-")], ds.budget_ms, {})
        ref = RefModel("random trees (dry run)", spkm_fn(m))
        boosters = None
    else:
        log("reference LightGBM p50/p95")
        boosters = lgb_reference(tr, va, keep, hp, args.seed, args.workers)
        ref = RefModel("LightGBM quantile p50/p95, 200 rounds, depth 4", lgb_fn(boosters, keep))
    prow = rng.choice(te.N, min(te.N, args.perm_rows), replace=False)
    Xp, yp = te.X[prow], te.y[prow].astype(np.float64)
    log("permutation importance")
    perm, base = permutation(ref, Xp, yp, {n: [i] for i, n in enumerate(names)}, rng)
    cgroups = {}
    for i, n in enumerate(names):
        cgroups.setdefault(clus[n], []).append(i)
    cperm, _ = permutation(ref, Xp, yp, {f"c{c}": idx for c, idx in cgroups.items() if len(idx) > 1}, rng)
    cluster_perm = [{"cluster": int(k[1:]), "features": [names[i] for i in cgroups[int(k[1:])]], "d_p50": v[0],
                     "d_p95": v[1]} for k, v in sorted(cperm.items(), key=lambda kv: -kv[1][1])]
    shap = {}
    if boosters is not None:
        log("TreeSHAP (LightGBM pred_contrib)")
        contrib = boosters[1].predict(Xp[:20000], pred_contrib=True, num_iteration=boosters[1].best_iteration or None)
        shap = {n: float(v) for n, v in zip(names, np.abs(contrib[:, :-1]).mean(0))}
    ablations = []
    if boosters is not None:
        full = score(te.y, lgb_fn(lgb_reference(tr, va, keep, hp, args.seed, args.workers,
                                                args.ablation_subsample, np.random.default_rng(args.seed)), keep)(te.X))
        for g, fs in cfg.get("ablation_groups", {}).items():
            drop = set(Dt.name_index(fs))
            if not drop:
                continue
            kp = np.array([i for i in range(spkm.F) if i not in drop])
            log(f"ablation: without {g} ({len(drop)} features)")
            bs = lgb_reference(tr, va, kp, hp, args.seed, args.workers, args.ablation_subsample,
                               np.random.default_rng(args.seed))
            s = score(te.y, lgb_fn(bs, kp)(te.X))
            ablations.append({"group": g, "n_features": len(drop), "features": [names[i] for i in sorted(drop)],
                              "d_pinball_p50": s[0] - full[0], "d_pinball_p95": s[1] - full[1]})
    log("lever costs (counterfactual rows)")
    levers = lever_costs(ds, cfg, ref, rows_step=20 if not args.dry_run else 5)
    truth = paired_lever_truth(ds, cfg, ref)
    ladder = None if args.no_ladder else ladder_search(ds, cfg, ref, n_sessions=6 if not args.dry_run else 1,
                                                       frames=4000 if not args.dry_run else 300)
    ck_perm = []
    for r in args.checkpoints:
        m = spkm.read(f"{r}/model.spkm")
        if fastpred.is_stateful(m):
            log(f"skip {r}: stateful family {m.family} (permutation needs the stateless path)")
            continue
        p, _ = permutation(RefModel(r, spkm_fn(m)), Xp, yp, {n: [i] for i, n in enumerate(names)}, rng, repeats=1)
        ck_perm.append({"run": rel(r), "perm": p})
    qfeats = set(n for n in names if n.startswith("q_"))
    feats = recommend(names, std0, clus, perm, shap, groups_of, qfeats)
    run_id = make_run_id("importance", "study")
    out = {"schema": "spark-ft-importance-v1", "run_id": run_id, "created_utc": common.utc_now(),
           "dry_run": bool(args.dry_run), "dataset": ds.info(), "reference": ref.name,
           "reference_test_pinball": {"p50": base[0], "p95": base[1]}, "pca": pc,
           "spearman_constant_features": [n for n, c in zip(names, const) if c], "features": feats,
           "cluster_perm": cluster_perm, "ablations": ablations, "levers": levers, "paired_truth": truth,
           "ladder": ladder,
           "checkpoint_perm": ck_perm, "code_version": common.code_version(), "hardware": common.hardware_info(),
           "time_s": t.s(), "seed": args.seed}
    write_json(res_dir / "importance" / f"{run_id}.json", out)
    (res_dir / "importance" / f"{run_id}.md").write_text(to_md(out))
    log(f"wrote {rel(res_dir / 'importance' / (run_id + '.md'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
