"""Offline closed-loop controller simulation on test sessions (counterfactual features).

Ground truth at a quality the session did not run at is not observed, so the simulator rescales
each logged frame analytically (the "analytic counterfactual simulator"):

    t_sim(q) = t_steps + (t_render - t_r_particles) * W(q) / W(actual) + t_r_particles * r^2 * dens(q) / dens(actual)
    W(q)     = buffer_px(q) + r^2 * (raster_px - aa_px + aa_px * aa(q)/aa(actual)) + r^2 * glow_req_px * g(q)^2
    r        = render_scale(q) / render_scale(actual)

which keeps each frame's real noise, spikes and sim cost (render is ~99.7% of frame time, and render
cost tracks pixel work; PHASE1_REPORT section 2). `validate_simulator` checks it on the paired
sessions (same game at two qualities). Lever kinds come from the config (`levers[].sim`).

Controllers are ports of engine/controller.cpp (model: predict-then-optimise with hysteresis;
reactive: EMA + rate limit + panic; fixed; plus an oracle that knows t_sim, as an upper bound).
History features the model sees (t_prev_ms, t_ema_ms, ...) are recomputed from the SIMULATED
frame times, so a controller lives with the consequences of its own choices.
"""
import numpy as np

import features
import fastpred
import spkm


# ---------------------------------------------------------------- simulator

def _lever_values(cfg, kind):
    for i, lv in enumerate(cfg["levers"]):
        if lv.get("sim") == kind:
            return i, np.asarray(lv["values"], np.float64)
    return None, None


def simulate_costs(session, cfg, ladder):
    """(L, n) simulated t_total (ms) of every frame of `session` at every ladder level."""
    a = session.a
    hdr = session.header["session"]
    lw, lh = int(hdr["logical_w"]), int(hdr["logical_h"])
    out_scale = np.float32(hdr["out_scale"])
    q = a["q"].astype(np.int64)
    si, sv = _lever_values(cfg, "scale")
    di, dv = _lever_values(cfg, "density")
    gi, gv = _lever_values(cfg, "glow")
    ai, av = _lever_values(cfg, "aa")
    fac = [(i, np.asarray(lv["values"], np.float64)) for i, lv in enumerate(cfg["levers"]) if lv.get("sim") == "factor"]
    s_a = a["render_scale"].astype(np.float64)
    g_a = gv[q[:, gi]] if gi is not None else np.ones(session.n)
    aa_a = av[q[:, ai]] if ai is not None else np.ones(session.n)
    d_a = dv[q[:, di]] if di is not None else np.ones(session.n)
    W_a = a["px_clear"] + a["raster_px"] + a["px_glow_req"] * g_a ** 2
    t_part = np.minimum(a["t_r_particles"].astype(np.float64), a["t_render"])
    t_np = a["t_render"].astype(np.float64) - t_part
    rest = np.maximum(0.0, session.a["y"].astype(np.float64) - a["t_steps"] - a["t_render"])
    out = np.empty((len(ladder), session.n))
    for l, lev in enumerate(ladder):
        s_l = float(np.float32(out_scale * np.float32(sv[lev[si]]))) if si is not None else s_a
        buf = (features.scaled_size(lw, np.float32(s_l)) * features.scaled_size(lh, np.float32(s_l))).astype(np.float64)
        r2 = (s_l / np.maximum(s_a, 1e-9)) ** 2
        g_l = gv[lev[gi]] if gi is not None else g_a
        aa_l = av[lev[ai]] if ai is not None else aa_a
        d_l = dv[lev[di]] if di is not None else d_a
        W_l = buf + r2 * (a["raster_px"] - a["px_aa"] + a["px_aa"] * aa_l / np.maximum(aa_a, 1e-9)) \
            + r2 * a["px_glow_req"] * g_l ** 2
        ratio = np.where(W_a > 0, W_l / np.maximum(W_a, 1), 1.0)
        f = np.ones(session.n)
        for i, vals in fac:
            f = f * vals[lev[i]] / vals[q[:, i]]
        out[l] = a["t_steps"] + (t_np * ratio + t_part * r2 * d_l / np.maximum(d_a, 1e-9)) * f + rest
    return out


def _comparable_groups(sd):
    """Sessions that play the same game at different fixed qualities on the same hardware condition:
    MANIFEST `pair` (laptop-v1) or `group` (laptop-v2: lever sweeps, matched, vhw)."""
    groups = {}
    for s in sd.sessions:
        g = s.meta.get("pair") or s.meta.get("group")
        if g and s.meta.get("policy", "fixed") == "fixed":
            key = (g, s.meta.get("loop"), s.meta.get("core_type"), s.meta.get("corunner"), s.meta.get("condition"),
                   s.meta.get("cpu"))
            groups.setdefault(key, []).append(s)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _quality(s):
    return tuple(int(v) for v in s.a["q"][len(s.a["q"]) // 2])  # fixed-quality runs


def align(A, B):
    """Row pairs (i in A, j in B) for the same game step (step0) and the same number of sim steps."""
    ka = {(int(s0), int(n)): i for i, (s0, n) in enumerate(zip(A.a["step0"], A.a["n_steps"]))}
    pairs = [(ka[(int(s0), int(n))], j) for j, (s0, n) in enumerate(zip(B.a["step0"], B.a["n_steps"]))
             if (int(s0), int(n)) in ka]
    return (np.array(pairs).T if pairs else (np.zeros(0, int), np.zeros(0, int)))


def validate_simulator(ds, cfg, split="test"):
    """Paired sessions: predict session B's frame times from session A's frames at B's quality and compare
    (frames aligned by step0/n_steps). Every ordered pair with different qualities in a comparable group."""
    sd = ds.split(split)
    rows = []
    for key, ss in _comparable_groups(sd).items():
        for A in ss:
            for B in ss:
                qA, qB = _quality(A), _quality(B)
                if A is B or qA == qB:
                    continue
                ia, jb = align(A, B)
                if len(ia) < 50:
                    continue
                tA = simulate_costs(A, cfg, [qB])[0]
                pred, act, same = tA[ia], B.a["y"][jb].astype(np.float64), A.a["y"][ia].astype(np.float64)
                rows.append({"group": str(key[0]), "from": A.id, "to": B.id, "q_from": list(qA), "q_to": list(qB),
                             "frames": int(len(ia)), "mean_actual_ms": float(act.mean()),
                             "mean_sim_ms": float(pred.mean()), "mean_source_ms": float(same.mean()),
                             "ratio_of_means": float(pred.mean() / max(act.mean(), 1e-9)),
                             "median_abs_rel_err": float(np.median(np.abs(pred - act) / np.maximum(act, 1e-6))),
                             "corr": float(np.corrcoef(pred, act)[0, 1]) if len(ia) > 2 else None})
    return rows


# ---------------------------------------------------------------- controllers (ports of engine/controller.cpp)

class CtlState:
    def __init__(self, S):
        self.level = np.zeros(S, np.int64)
        self.up = np.zeros(S, np.int64)
        self.miss_run = np.zeros(S, np.int64)
        self.since_change = np.full(S, 1 << 20, np.int64)
        self.since_drop = np.full(S, 1 << 20, np.int64)
        self.ema = np.zeros(S)
        self.ema_init = np.zeros(S, bool)


def decide_model(st, s, v, target, p, L):
    """v: (L,) decision values for session s at every ladder level. Returns (new_level, action, evals)."""
    c = st.level[s]
    evals = 1
    if v[c] > target:
        pick = L - 1
        for l in range(c + 1, L):
            evals += 1
            if v[l] <= target:
                pick = l
                break
        st.up[s] = 0
        return pick, (1 if pick != c else 0), evals
    new, action = c, 0
    if c > 0 and st.since_drop[s] >= p["up_cooldown"]:
        evals += 1
        st.up[s] = st.up[s] + 1 if v[c - 1] <= target * p["up_frac"] else 0
        if st.up[s] >= p["up_hold"]:
            st.up[s] = 0
            new, action = c - 1, 2
    else:
        st.up[s] = 0
    return new, action, evals


def decide_reactive(st, s, last_ms, budget, target, p, L):
    c = st.level[s]
    if last_ms <= 0:
        return c, 0
    st.ema[s] = last_ms if not st.ema_init[s] else (1 - p["react_ema_alpha"]) * st.ema[s] + p["react_ema_alpha"] * last_ms
    st.ema_init[s] = True
    st.miss_run[s] = st.miss_run[s] + 1 if last_ms > budget else 0
    pick = c
    if st.miss_run[s] >= p["react_panic_frames"]:
        pick = min(L - 1, c + 2)
        st.miss_run[s] = 0
        st.up[s] = 0
    elif st.ema[s] > target and st.since_change[s] >= p["react_down_every"]:
        pick = min(L - 1, c + 1)
        st.up[s] = 0
    elif st.ema[s] < target * p["react_up_frac"]:
        st.up[s] += 1
        if st.up[s] >= p["react_up_hold"] and c > 0:
            pick = c - 1
            st.up[s] = 0
    else:
        st.up[s] = 0
    return pick, (1 if pick > c else 2 if pick < c else 0)


DECISION_FALLBACK = ["t_p99_ms", "t_p95_ms", "t_p90_ms", "t_mean_ms", "t_p50_ms"]


def decision_index(model, wanted):
    names = [o[0] for o in model.outputs]
    if wanted in names:
        return names.index(wanted)
    for n in DECISION_FALLBACK:
        if n in names:
            return names.index(n)
    raise ValueError(f"model has no time output ({names})")


class Bench:
    """Test sessions prepared for simulation: CF features per ladder level, simulated costs."""

    def __init__(self, ds, cfg, split="test"):
        self.ds, self.cfg = ds, cfg
        self.sd = sd = ds.split(split)
        self.ladder = ds.ladder
        self.L = len(self.ladder)
        self.starts, self.lens = sd.offsets[:-1], np.diff(sd.offsets)
        self.S = len(self.starts)
        self.CF = np.stack([ds.counterfactual_split(split, q) for q in self.ladder])  # (L, N, F) float32
        self.tsim = np.concatenate([simulate_costs(s, cfg, self.ladder) for s in sd.sessions], axis=1)  # (L, N)
        self.t_steps = sd.aux["t_steps"].astype(np.float64)
        self.minutes = np.array([sd.aux["n_steps"][a:b].sum() * cfg.get("step_s", 1 / 60) / 60
                                 for a, b in zip(sd.offsets[:-1], sd.offsets[1:])])
        names = spkm.FEATURE_NAMES
        self.hcols = {kind: names.index(f) for f, kind in cfg.get("history_overrides", {}).items() if f in names}
        self.ema_alpha = cfg.get("t_ema_alpha", 0.2)


def run(bench, mode, budget, margin, model=None, decision="t_p95_ms", fixed_level=0, p=None):
    """Simulate one controller over all sessions in lockstep. Returns per-run metrics."""
    B, L, S = bench, bench.L, bench.S
    p = dict(bench.cfg["controller"], **(p or {}))
    target = budget * (1 - margin)
    st = CtlState(S)
    if mode == "fixed":
        st.level[:] = fixed_level
    pred = fastpred.StepPredictor(model, S) if model is not None else None
    di = decision_index(model, decision) if model is not None else None
    hist = {k: np.zeros(S) for k in ("t_prev", "t_prev2", "t_ema", "t_steps_prev", "t_render_prev")}
    ema_init = np.zeros(S, bool)
    win = np.zeros((S, 8))
    chosen_t = np.zeros(B.sd.N)
    chosen_l = np.zeros(B.sd.N, np.int16)
    evals = np.zeros(B.sd.N, np.int8)
    switches = np.zeros(S, np.int64)
    for k in range(int(B.lens.max())):
        act = np.nonzero(B.lens > k)[0]
        rows = B.starts[act] + k
        last = hist["t_prev"][act]
        if mode == "model":
            Xc = B.CF[:, rows].transpose(1, 0, 2).copy()  # (S', L, F)
            for kind, col in B.hcols.items():
                Xc[:, :, col] = (np.max(win[act], 1) if kind == "t_max8" else hist[kind][act])[:, None]
            pred.begin_frame(act)
            V = pred.predict(Xc, act)  # (S', L, n_out)
        for j, s in enumerate(act):
            st.since_change[s] += 1
            st.since_drop[s] += 1
            c = st.level[s]
            if mode == "model":
                new, action, ev = decide_model(st, s, V[j, :, di], target, p, L)
            elif mode == "reactive":
                new, action = decide_reactive(st, s, last[j], budget, target, p, L)
                ev = 0
            elif mode == "oracle":
                ok = np.nonzero(B.tsim[:, rows[j]] <= target)[0]
                new = int(ok[0]) if len(ok) else L - 1
                action, ev = (1 if new > c else 2 if new < c else 0), 0
            else:
                new, action, ev = c, 0, 0
            if new != c:
                st.since_change[s] = 0
                switches[s] += 1
                if action == 1:
                    st.since_drop[s] = 0
            st.level[s] = new
            evals[rows[j]] = ev
        lv = st.level[act]
        t = B.tsim[lv, rows]
        chosen_t[rows], chosen_l[rows] = t, lv
        if mode == "model":
            x = Xc[np.arange(len(act)), lv]
            ts = B.t_steps[rows]
            h = np.concatenate([x, np.stack([t, ts, t - ts], 1)], 1)
            pred.end_frame(x, h, t, act)
        hist["t_prev2"][act] = hist["t_prev"][act]
        hist["t_prev"][act] = t
        hist["t_ema"][act] = np.where(ema_init[act], (1 - B.ema_alpha) * hist["t_ema"][act] + B.ema_alpha * t, t)
        ema_init[act] = True
        hist["t_steps_prev"][act] = B.t_steps[rows]
        hist["t_render_prev"][act] = t - B.t_steps[rows]
        win[act] = np.concatenate([win[act][:, 1:], t[:, None]], 1)
    miss = chosen_t > budget
    mins = B.minutes.sum()
    res = {"mode": mode, "budget_ms": budget, "margin": margin, "frames": int(B.sd.N), "game_minutes": float(mins),
           "misses": int(miss.sum()), "misses_per_min": float(miss.sum() / max(mins, 1e-9)),
           "miss_rate": float(miss.mean()), "p99_ms": float(np.percentile(chosen_t, 99)),
           "max_ms": float(chosen_t.max()), "mean_ms": float(chosen_t.mean()),
           "mean_level": float(chosen_l.mean()), "level_hist": np.bincount(chosen_l, minlength=L).tolist(),
           "switches": int(switches.sum()), "switches_per_min": float(switches.sum() / max(mins, 1e-9)),
           "mean_evals": float(evals.mean())}
    by = {}
    for c, name in enumerate(["S0", "S1", "S2", "S3", "S4"]):
        idx = B.sd.stress == c
        if idx.any():
            by[name] = {"miss_rate": float(miss[idx].mean()), "mean_level": float(chosen_l[idx].mean())}
    res["by_stress"] = by
    if mode == "fixed":
        res["fixed_level"] = fixed_level
    return res


def equal_quality(curve, ref_level):
    """Interpolate misses/min of a (mean_level, misses_per_min) curve at the reference mean level."""
    pts = sorted((r["mean_level"], r["misses_per_min"]) for r in curve)
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    if len(xs) == 0 or ref_level < xs.min() or ref_level > xs.max():
        return None
    return float(np.interp(ref_level, xs, ys))
