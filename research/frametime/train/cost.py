"""Estimated in-game predictor cost per frame, scaled from the measured laptop numbers in
MODEL_CONTRACT.md section 7 / PHASE1_REPORT.md section 4 (i5-13500H P-core, random weights,
`make bench-predictor` + in-game t_pred_us). These are ESTIMATES by multiply-accumulate count;
the real number comes from running the checkpoint in the game (`--ctl model`, t_pred_us).

Per frame: feature computation (~0.8 us) + evals x predict + the end-of-frame work
(GRU step, TCN encode, RLS update). The model controller typically makes 2 evals (max 8).
"""
import numpy as np

import spkm

BUDGET_US = 50.0
FEATURES_US = 0.8
# reference measurements: (cost_us, reference size in MACs or comparable units)
REF = {
    "linear_predict": (0.13, 47 * 4),
    "mlp_predict": (4.3, 47 * 64 + 64 * 64 + 64 * 4),
    "gbt_predict": (5.1, 300 * 4),            # trees x depth
    "gru_head": (1.6, (32 + 47) * 32 + 32 * 4),
    "gru_step": (4.8, 3 * 32 * (50 + 32)),
    "tcn_head": (1.5, (32 + 47) * 32 + 32 * 4),
    "tcn_encode": (25.0, None),               # filled below: needed-position MACs of 3x32, k3, d1/2/4, L32
    "rls_predict": (0.07, 17),
    "rls_update": (0.54, 17 * 17),
}


def tcn_needed_macs(cin0, channels, k, dilations, L, residual=None):
    """MACs the engine spends in TCN encode: only positions the last output depends on (engine/predictor.cpp)."""
    need = [None] * len(dilations)
    mark = np.zeros(L, bool)
    mark[L - 1] = True
    for l in range(len(dilations) - 1, -1, -1):
        need[l] = np.nonzero(mark)[0]
        prev = np.zeros(L, bool)
        for t in need[l]:
            for j in range(k):
                ts = t - (k - 1 - j) * dilations[l]
                if ts >= 0:
                    prev[ts] = True
            if (residual[l] if residual is not None else l > 0):
                prev[t] = True  # a residual layer reads its input at t
        mark = prev
    macs, cin = 0, cin0
    for l, cout in enumerate(channels):
        macs += len(need[l]) * cout * cin * k
        cin = cout
    return macs


REF["tcn_encode"] = (25.0, tcn_needed_macs(50, [32, 32, 32], 3, [1, 2, 4], 32))


def _scale(key, units):
    us, ref = REF[key]
    return us * units / ref


def _dense_macs(layers):
    return sum(W.shape[0] * W.shape[1] for W, _, _ in layers)


def estimate(m: spkm.Spkm, evals=2, worst_evals=8):
    """{'predict_us', 'per_frame_extra_us', 'ingame_us_est' (typical: 2 evals), 'ingame_us_est_worst' (one eval
    per ladder level: the controller's worst case when dropping), 'within_budget' (worst case <= 50 us)}."""
    p = m.params
    extra = 0.0
    if m.family == "linear":
        pred = _scale("linear_predict", p["W"].size)
    elif m.family == "mlp":
        pred = _scale("mlp_predict", _dense_macs(p["layers"]))
    elif m.family == "gbt":
        depth = []
        for _, nodes in p["trees"]:
            d = np.zeros(len(nodes), int)
            for i, (f, _, l, r, _, _) in enumerate(nodes):
                if f >= 0:
                    d[l] = d[r] = d[i] + 1
            depth.append(d.max() if len(d) else 0)
        pred = _scale("gbt_predict", float(np.sum(depth)))
    elif m.family == "rls":
        n = len(p["idx"]) + 1
        pred = _scale("rls_predict", n)
        extra = _scale("rls_update", n * n)
    elif m.family == "gru":
        H, Dh = p["Whh"].shape[1], p["Wih"].shape[1]
        pred = _scale("gru_head", _dense_macs(p["head"]))
        extra = _scale("gru_step", 3 * H * (Dh + H))
        if not p["stream"]:
            extra *= m.history_len
    elif m.family == "tcn":
        cs = p["convs"]
        macs = tcn_needed_macs(cs[0]["W"].shape[1], [c["W"].shape[0] for c in cs], cs[0]["W"].shape[2],
                               [c["dilation"] for c in cs], m.history_len, [c["residual"] for c in cs])
        pred = _scale("tcn_head", _dense_macs(p["head"]))
        extra = _scale("tcn_encode", macs)
    else:
        raise ValueError(m.family)
    total = FEATURES_US + evals * pred + extra
    worst = FEATURES_US + worst_evals * pred + extra
    return {"predict_us": round(pred, 3), "per_frame_extra_us": round(extra, 3), "ingame_us_est": round(total, 2),
            "ingame_us_est_worst": round(worst, 2), "worst_evals": worst_evals,
            "within_budget": bool(worst <= BUDGET_US),
            "method": "scaled by MACs from MODEL_CONTRACT section 7 (laptop P-core); measure in game"}
