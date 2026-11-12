"""Reference implementation of feature set spark-ft-v2 (MODEL_CONTRACT.md) from frame logs.

    from framelog import read_log
    from features import compute_features, history_vectors, targets
    header, rec = read_log(path)
    X = compute_features(rec, header)          # (frames, 64) float32, exactly what the engine computed
    Hv = history_vectors(rec, X)               # (frames, 67) float32, sequence-model inputs
    X_cf = compute_features(rec, header, quality=(3, 3, 3, 1))   # counterfactual: other quality (s, p, g, aa)

The fingerprint terms come from header["machine"]["fingerprint"] (spark-hw-v1). v1 logs (phase-1
dataset) also work: AA was always on and there is no fingerprint, so the fp/t_ref features are 0.
compute_features_v1() reproduces the frozen v1 feature set (47 features) for analysis.

Row k of X is the input for predicting frame k (computed before frame k ran,
from frames < k). The engine's C++ (engine/features.cpp) is the other half;
tests/test_features_parity.py checks they agree on real logs.
"""
import math

import numpy as np

from spkm import FEATURE_NAMES, FEATURE_NAMES_V1, F

F_V1 = len(FEATURE_NAMES_V1)

K_SCALE = np.array([1.0, 0.8, 0.65, 0.5], np.float32)
K_PARTICLES = np.array([1.0, 0.6, 0.35, 0.15], np.float32)
K_GLOW = np.array([1.0, 0.75, 0.5, 0.0], np.float32)
K_AA = np.array([1.0, 0.0], np.float32)


def scaled_size(logical, s):
    """Canvas::scaledSize in float32: max(1, int(logical * s + 0.5))."""
    v = np.float32(logical) * np.asarray(s, np.float32) + np.float32(0.5)
    return np.maximum(1, v.astype(np.int64))


def _state(rec):
    """FeatureState after each frame, shifted so row k holds the state BEFORE frame k."""
    n = len(rec)
    t = rec["t_total_us"].astype(np.float64) / 1000.0
    t_steps = rec["t_steps_us"].astype(np.float64) / 1000.0
    t_render = rec["t_render_us"].astype(np.float64) / 1000.0
    steps = rec["n_steps"].astype(np.int64)
    s2 = rec["render_scale"].astype(np.float64) ** 2
    raster_px = sum(rec[k].astype(np.float64) for k in ("px_rect", "px_circle", "px_ring", "px_line", "px_poly"))
    g = K_GLOW[np.minimum(rec["q_glow"], 3)].astype(np.float64)
    work = (rec["px_clear"].astype(np.float64) + raster_px + rec["px_glow_req"].astype(np.float64) * g * g) / 1e6

    st = {k: np.zeros(n + 1) for k in ("t_prev", "t_prev2", "t_ema", "t_max8", "t_steps_prev", "t_render_prev",
                                        "steps_prev", "ema_step", "ema_rpw", "raster_lpx", "glow_req_lpx")}
    ema = ema_step = ema_rpw = 0.0
    ema_init = step_init = rpw_init = False
    with np.errstate(divide="ignore", invalid="ignore"):
        step_v = np.where(steps >= 1, t_steps / np.maximum(steps, 1), 0)
        rpw_v = np.where(work > 0, t_render / np.where(work > 0, work, 1), 0)
    for k in range(n):  # sequential EMAs (exactly as the engine; double precision)
        ema = 0.8 * ema + 0.2 * t[k] if ema_init else t[k]
        ema_init = True
        if steps[k] >= 1:
            ema_step = 0.9 * ema_step + 0.1 * step_v[k] if step_init else step_v[k]
            step_init = True
        if work[k] > 0:
            ema_rpw = 0.9 * ema_rpw + 0.1 * rpw_v[k] if rpw_init else rpw_v[k]
            rpw_init = True
        st["t_ema"][k + 1] = ema
        st["ema_step"][k + 1] = ema_step
        st["ema_rpw"][k + 1] = ema_rpw
    st["t_prev"][1:] = t
    st["t_prev2"][2:] = t[:-1]
    win = np.full((n + 8,), 0.0)
    win[8:] = t
    st["t_max8"][1:] = np.max(np.stack([win[i:i + n] for i in range(1, 9)]), axis=0)
    st["t_steps_prev"][1:] = t_steps
    st["t_render_prev"][1:] = t_render
    st["steps_prev"][1:] = steps
    st["raster_lpx"][1:] = raster_px / s2
    st["glow_req_lpx"][1:] = rec["px_glow_req"].astype(np.float64) / s2
    return {k: v[:n] for k, v in st.items()}


def compute_features_v1(rec, header, quality=None):
    """Frozen spark-ft-v1 features (frames, 47). quality=(scale, particles, glow) for counterfactuals."""
    sess = header["session"]
    lw, lh = int(sess["logical_w"]), int(sess["logical_h"])
    out_scale = np.float32(sess["out_scale"])
    n = len(rec)
    st = _state(rec)
    X = np.zeros((n, F_V1), np.float64)
    if quality is None:
        qs, qp, qg = rec["q_scale"], rec["q_particles"], rec["q_glow"]
        s = rec["render_scale"].astype(np.float32)
    else:
        qs, qp, qg = (np.full(n, v, np.int64) for v in quality[:3])
        s = np.full(n, out_scale * K_SCALE[quality[0]], np.float32)
    s2 = s.astype(np.float64) ** 2
    g = K_GLOW[qg].astype(np.float64)
    dens = K_PARTICLES[qp].astype(np.float64)
    render_mpx = scaled_size(lw, s).astype(np.float64) * scaled_size(lh, s).astype(np.float64) / 1e6
    est_raster = st["raster_lpx"] * s2 / 1e6
    est_glow = st["glow_req_lpx"] * s2 * g * g / 1e6

    col = {name: i for i, name in enumerate(FEATURE_NAMES_V1)}
    X[:, 0] = rec["n_steps"]
    X[:, 1] = rec["n_steps"].astype(np.int64) - rec["n_frozen"]
    X[:, 2] = rec["acc_ms"]
    X[:, 3] = rec["hitstop_ms"]
    X[:, 4] = rec["trauma"]
    X[:, 5] = rec["scene"] == 1
    X[:, 6] = rec["scene"] == 2
    X[:, 7] = rec["scene"] == 3
    for name in ("entities", "enemies", "enemy_k0", "enemy_k1", "enemy_k2", "enemy_k3", "enemy_k4", "projectiles",
                 "enemy_projectiles", "pickups", "telegraphs", "effects", "popups", "tele_due_now", "tele_due_6",
                 "tele_due_30", "burst_due", "particles_alive", "particles_streak", "particles_expiring", "wave",
                 "wave_to_spawn"):
        X[:, col[name]] = rec[name]
    X[:, 30] = s
    X[:, 31] = render_mpx
    X[:, 32] = dens
    X[:, 33] = g
    X[:, 34] = est_raster
    X[:, 35] = est_glow
    X[:, 36] = rec["particles_alive"].astype(np.float64) + rec["burst_due"].astype(np.float64) * dens
    X[:, 37] = rec["n_steps"].astype(np.float64) * st["ema_step"] + st["ema_rpw"] * (render_mpx + est_raster + est_glow)
    X[:, 38] = st["t_prev"]
    X[:, 39] = st["t_prev2"]
    X[:, 40] = st["t_ema"]
    X[:, 41] = st["t_max8"]
    X[:, 42] = st["t_steps_prev"]
    X[:, 43] = st["t_render_prev"]
    X[:, 44] = st["steps_prev"]
    X[:, 45] = rec["cpu_mhz"].astype(np.float64) / 1000.0
    X[:, 46] = rec["ctxsw"]
    return X.astype(np.float32)


def history_vectors(rec, X):
    """Sequence-model inputs: [x(j), t_total_ms, t_steps_ms, t_render_ms] per frame (float32, as the engine)."""
    ms = lambda k: rec[k].astype(np.float32) / np.float32(1000)
    return np.concatenate([X, ms("t_total_us")[:, None], ms("t_steps_us")[:, None], ms("t_render_us")[:, None]],
                          axis=1).astype(np.float32)


def targets(rec, budget_ms=1000 / 60):
    """t_total in ms (the regression target) and the miss indicator for a budget."""
    t = rec["t_total_us"].astype(np.float64) / 1000.0
    return t, t > budget_ms


# ---------------------------------------------------------------- v2

def fp_coef(fingerprint):
    """engine/features.cpp FpCoef::from: constants from a spark-hw-v1 record (dict) or None."""
    c = dict(valid=False, stepBase=0.0, stepSlope=0.0, stepNormal=0.0, nsClear=0.0, nsRect=0.0, nsPoly=0.0,
             nsGlow=0.0, nsCircle=(0.0, 0.0), nsRing=(0.0, 0.0), nsLine=(0.0, 0.0), calib=1.0, cliff=1.0, px1=1.0,
             px32=1.0, noiseP99Ms=0.0, noiseP999Ms=0.0, sleepP99Ms=0.0, logs=[0.0] * 9)
    if not fingerprint:
        return c
    f = fingerprint["fp"]
    g = lambda k, d=0.0: float(f.get(k, d))
    lg = lambda v: math.log10(v) if v > 0 else 0.0
    c.update(valid=True, stepBase=g("step_ms_base"), stepSlope=g("step_ms_per_entity"), stepNormal=g("step_ms_normal"),
             nsClear=g("ns_px_clear"), nsRect=g("ns_px_rect"), nsPoly=g("ns_px_poly"), nsGlow=g("ns_px_glow"),
             nsCircle=(g("ns_px_circle"), g("ns_px_circle_aliased")), nsRing=(g("ns_px_ring"), g("ns_px_ring_aliased")),
             nsLine=(g("ns_px_line"), g("ns_px_line_aliased")), calib=g("raster_calib", 1.0), cliff=g("cliff_ratio", 1.0))
    c["px1"] = max(1.0, g("frame_px_s1", 1.0))
    c["px32"] = max(c["px1"] + 1, g("frame_px_s32", c["px1"] + 1))
    c["noiseP99Ms"] = g("noise_excess_p99_us") / 1000.0
    c["noiseP999Ms"] = g("noise_excess_p999_us") / 1000.0
    c["sleepP99Ms"] = g("sleep_over_p99_us") / 1000.0
    c["logs"] = [lg(c["stepNormal"]), lg(g("frame_ms_s1") * 1e6 / c["px1"]), lg(c["cliff"]), lg(c["nsGlow"]),
                 lg(g("part_update_ns")), lg(g("ghz_start")), lg(g("lat_l3_ns")), lg(g("lat_dram_ns")),
                 lg(g("bw_dram_gbs"))]
    return c


def compute_features(rec, header, quality=None, fingerprint="header"):
    """spark-ft-v2 feature matrix (frames, 64) float32, bit-exact with engine/features.cpp.

    quality=None: each frame's actual levels. quality=(scale, particles, glow[, aa]) levels: counterfactual
    rows for that setting (aa defaults to 0 = on). fingerprint="header" uses the one the session ran with
    (machine.fingerprint, None in v1 logs); pass a dict to evaluate t_ref under another fingerprint."""
    n = len(rec)
    Xv1 = compute_features_v1(rec, header, quality)
    if fingerprint == "header":
        fingerprint = header.get("machine", {}).get("fingerprint")
    c = fp_coef(fingerprint)
    names = rec.dtype.names
    if quality is None:
        aa = rec["q_aa"].astype(np.int64) if "q_aa" in names and header.get("format_version", 1) >= 2 \
            else np.zeros(n, np.int64)
        s = rec["render_scale"].astype(np.float32)
        qg = rec["q_glow"]
    else:
        aa = np.full(n, quality[3] if len(quality) > 3 else 0, np.int64)
        s = np.full(n, np.float32(header["session"]["out_scale"]) * K_SCALE[quality[0]], np.float32)
        qg = np.full(n, quality[2], np.int64)
    s2 = s.astype(np.float64) ** 2
    g = K_GLOW[qg].astype(np.float64)
    sess = header["session"]
    lw, lh = int(sess["logical_w"]), int(sess["logical_h"])
    render_mpx = scaled_size(lw, s).astype(np.float64) * scaled_size(lh, s).astype(np.float64) / 1e6

    # Previous frame's per-class logical pixel work (state before frame k).
    ps2 = rec["render_scale"].astype(np.float64) ** 2
    prev = lambda field: np.concatenate([[0.0], rec[field].astype(np.float64)[:-1] / ps2[:-1]])
    lpx = {k: prev("px_" + k) for k in ("rect", "circle", "ring", "line", "poly")}
    glow_req = prev("px_glow_req")

    X = np.zeros((n, F), np.float64)
    X[:, :F_V1] = Xv1
    X[:, 47] = np.where(aa != 0, 0.0, 1.0)
    X[:, 48] = (lpx["circle"] + lpx["ring"] + lpx["line"]) * s2 / 1e6
    if c["valid"]:
        per_step = np.maximum(c["stepBase"] + c["stepSlope"] * rec["entities"].astype(np.float64), 0.25 * c["stepNormal"])
        t_steps = rec["n_steps"].astype(np.float64) * per_step
        buf_px = render_mpx * 1e6
        t = np.clip((buf_px - c["px1"]) / (c["px32"] - c["px1"]), 0.0, 1.0)
        cliff = 1 + (c["cliff"] - 1) * t
        pick = lambda pair: np.where(aa != 0, pair[1], pair[0])
        ns = c["nsClear"] * buf_px + s2 * (c["nsRect"] * lpx["rect"] + c["nsPoly"] * lpx["poly"] +
                                           pick(c["nsCircle"]) * lpx["circle"] + pick(c["nsRing"]) * lpx["ring"] +
                                           pick(c["nsLine"]) * lpx["line"] + c["nsGlow"] * glow_req * g * g)
        t_render = c["calib"] * cliff * ns / 1e6
        X[:, 49] = t_steps
        X[:, 50] = t_render
        X[:, 51] = t_steps + t_render
    X[:, 52] = c["noiseP99Ms"]
    X[:, 53] = c["noiseP999Ms"]
    X[:, 54] = c["sleepP99Ms"]
    X[:, 55:64] = np.array(c["logs"])[None, :]
    return X.astype(np.float32)
