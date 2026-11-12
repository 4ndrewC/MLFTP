"""Gradient-boosted quantile trees (LightGBM) and their export to the engine's `gbt` family.

One LightGBM booster per quantile (objective=quantile) plus a binary booster for p_miss, all
packed into one GBTE chunk (MODEL_CONTRACT section 5). Trees are trained on the RAW features
and the model is exported without a NORM chunk, so the engine's xn equals x exactly and the
thresholds need no rescaling; each double threshold is rounded DOWN to float32, which keeps
`x <= t` identical for every float32 x.

Engine-format limits (engine/predictor.cpp): <= 100,000 trees, <= 1,000,000 nodes per tree,
numeric splits only, children after parents. The cost budget is the tighter limit:
cost.estimate() scales the measured 7.3 us for 300 depth-4 trees.
"""
import numpy as np

import spkm


def f32_floor(t):
    """Largest float32 <= t (so float32(x) <= f32_floor(t)  <=>  x <= t for float32 x)."""
    v = np.float32(t)
    if float(v) > t:
        v = np.nextafter(v, np.float32(-np.inf), dtype=np.float32)
    return float(v)


def tree_from_dump(struct, feat_map=None):
    """LightGBM dump_model() tree_structure -> engine nodes (pre-order: children after parents).
    feat_map: column index used in training -> index in the full feature vector."""
    nodes = []

    def walk(n):
        i = len(nodes)
        nodes.append(None)
        if "leaf_value" in n and "split_feature" not in n:
            nodes[i] = (-1, 0.0, 0, 0, float(n["leaf_value"]), False)
            return i
        if n.get("decision_type", "<=") != "<=":
            raise ValueError(f"unsupported split {n.get('decision_type')} (categorical splits are not exportable)")
        if str(n.get("missing_type", "None")) not in ("None", "NaN"):
            raise ValueError(f"missing_type {n.get('missing_type')} not exportable (train with zero_as_missing=False)")
        f = int(n["split_feature"])
        f = feat_map[f] if feat_map is not None else f
        thr = f32_floor(float(n["threshold"]))
        dl = bool(n.get("default_left", True))
        l = walk(n["left_child"])
        r = walk(n["right_child"])
        nodes[i] = (f, thr, l, r, 0.0, dl)
        return i

    walk(struct)
    return nodes


def eval_dump_tree(struct, x):
    """Reference evaluation of a dump tree on one raw row (for tests)."""
    n = struct
    while "split_feature" in n:
        v = x[int(n["split_feature"])]
        n = n["left_child"] if (np.isnan(v) and n.get("default_left", True)) or v <= n["threshold"] \
            else n["right_child"]
    return n["leaf_value"]


def export_gbt(dumps, outputs, budget_ms, info, feat_map=None):
    """dumps: list (one per output, in output order) of booster.dump_model() dicts."""
    trees = []
    for o, d in enumerate(dumps):
        for t in d["tree_info"]:
            trees.append((o, tree_from_dump(t["tree_structure"], feat_map)))
    return spkm.gbt(np.zeros(len(dumps)), trees, outputs, budget_ms=budget_ms, info=info)


def random_dump(rng, n_trees, depth, n_feat, leaf_scale=0.3):
    """A LightGBM-format dump with random trees (dry runs/tests only: no fitting)."""
    def node(d):
        if d == 0 or rng.random() < 0.15:
            return {"leaf_value": float(rng.normal(0, leaf_scale))}
        return {"split_feature": int(rng.integers(n_feat)), "threshold": float(rng.normal(0, 2)),
                "decision_type": "<=", "default_left": bool(rng.random() < 0.5), "missing_type": "None",
                "left_child": node(d - 1), "right_child": node(d - 1)}
    return {"tree_info": [{"tree_structure": node(depth)} for _ in range(n_trees)]}


def lgb_params(hp, objective, alpha=None, seed=0, threads=1):
    p = {
        "objective": objective, "learning_rate": hp.get("learning_rate", 0.05), "num_leaves": hp.get("num_leaves", 15),
        "max_depth": hp.get("max_depth", 4), "min_data_in_leaf": hp.get("min_data_in_leaf", 200),
        "feature_fraction": hp.get("feature_fraction", 0.9), "bagging_fraction": hp.get("bagging_fraction", 0.8),
        "bagging_freq": 1, "lambda_l2": hp.get("lambda_l2", 1.0), "max_bin": hp.get("max_bin", 255),
        "zero_as_missing": False, "use_missing": False, "deterministic": True, "force_row_wise": True,
        "seed": seed, "num_threads": threads, "verbosity": -1,
    }
    if alpha is not None:
        p["alpha"] = alpha
    return p


def fit_lgb(X, y, Xv, yv, objective, alpha, hp, seed, threads, n_rounds):
    """Train one booster with early stopping on the validation set. Returns (booster, info)."""
    from common import require_fitting
    require_fitting("trees.fit_lgb")
    import lightgbm as lgb
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    params = lgb_params(hp, objective, alpha, seed, threads)
    params["metric"] = "quantile" if objective == "quantile" else "binary_logloss"
    cb = [lgb.early_stopping(hp.get("early_stopping", 50), verbose=False), lgb.log_evaluation(100)]
    b = lgb.train(params, dtr, num_boost_round=n_rounds, valid_sets=[dva], valid_names=["val"], callbacks=cb)
    return b, {"best_iteration": int(b.best_iteration or b.current_iteration()), "params": params}


def booster_dump(b):
    return b.dump_model(num_iteration=b.best_iteration or None)
