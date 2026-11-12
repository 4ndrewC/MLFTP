"""Vectorised numpy evaluation of .spkm models (train/spkm.py Spkm objects).

spkm.Runner is the per-frame reference; this module computes the same thing for many rows
or many sessions at once, so we can evaluate exported models on whole test splits and run
the closed-loop controller simulation. Tested against spkm.Runner in tests/.

    P = StepPredictor(model, n_streams=S)     one stream per session (state for rls/gru/tcn)
    P.begin_frame(active)                     sequence models encode their history
    Y = P.predict(Xc, active)                 Xc: (S, C, F) candidate rows -> (S, C, n_out)
    P.end_frame(x, h, y, active)              chosen row, history vector, measured ms

    predict_split(model, X, H, y, offsets)    open-loop predictions for a whole split (N, n_out)
"""
from math import sqrt

import numpy as np
from scipy.special import erfc

import spkm

F, D = spkm.F, spkm.D


def _act(a, v):
    if a == "relu":
        return np.maximum(v, 0)
    if a == "tanh":
        return np.tanh(v)
    if a == "sigmoid":
        return 1 / (1 + np.exp(-v))
    return v


def _dense(layers, v):
    for W, b, a in layers:
        v = _act(a, v @ W.astype(np.float64).T + b.astype(np.float64))
    return v


def _sig(v):
    return 1 / (1 + np.exp(-v))


class _Trees:
    """All trees padded into arrays for simultaneous traversal."""

    def __init__(self, base, trees):
        self.base = np.asarray(base, np.float64)
        T = len(trees)
        maxn = max((len(n) for _, n in trees), default=1)
        self.out = np.array([o for o, _ in trees], np.int64)
        self.feat = np.full((T, maxn), -1, np.int64)
        self.thr = np.zeros((T, maxn), np.float32)
        self.left = np.zeros((T, maxn), np.int64)
        self.right = np.zeros((T, maxn), np.int64)
        self.val = np.zeros((T, maxn), np.float64)
        self.dl = np.zeros((T, maxn), bool)
        depth = 0
        for t, (_, nodes) in enumerate(trees):
            d = np.zeros(len(nodes), np.int64)
            for i, (f, th, l, r, v, dl) in enumerate(nodes):
                self.feat[t, i], self.thr[t, i], self.val[t, i], self.dl[t, i] = f, np.float32(th), v, dl
                if f >= 0:
                    self.left[t, i], self.right[t, i] = l, r
                    d[l] = d[r] = d[i] + 1
            depth = max(depth, int(d.max()) if len(d) else 0)
        self.depth = depth
        self.T = T

    def raw(self, xn32, n_out):
        """xn32: (R, F) float32 -> (R, n_out) float64."""
        R = len(xn32)
        out = np.tile(self.base, (R, 1))
        if self.T == 0:
            return out
        tix = np.arange(self.T)[None, :]
        node = np.zeros((R, self.T), np.int64)
        rows = np.arange(R)[:, None]
        for _ in range(self.depth):
            f = self.feat[tix, node]
            leaf = f < 0
            v = xn32[rows, np.maximum(f, 0)]
            thr = self.thr[tix, node]
            nan = np.isnan(v)
            go_left = np.where(nan, self.dl[tix, node], v <= thr)
            nxt = np.where(go_left, self.left[tix, node], self.right[tix, node])
            node = np.where(leaf, node, nxt)
        leafv = self.val[tix, node]
        for o in range(n_out):
            m = self.out == o
            if m.any():
                out[:, o] += leafv[:, m].sum(1)
        return out


class StepPredictor:
    def __init__(self, m: spkm.Spkm, n_streams=1):
        self.m = m
        self.S = n_streams
        p = m.params
        mean, std = m.norm if m.norm is not None else (np.zeros(F, np.float32), np.ones(F, np.float32))
        self.mean, self.inv = mean.astype(np.float32), (np.float32(1) / std.astype(np.float32))
        hm, hs = m.hnorm if m.hnorm is not None else (np.zeros(D, np.float32), np.ones(D, np.float32))
        self.hmean, self.hinv = hm.astype(np.float32), (np.float32(1) / hs.astype(np.float32))
        self.n_out = len(m.outputs)
        idx = lambda b: -1 if b == "-" else spkm.FEATURE_NAMES.index(b)
        self.base_idx = [idx(o[2]) for o in m.outputs]
        self.add_idx = [idx(o[3]) if len(o) > 3 else -1 for o in m.outputs]
        self.trans = [o[1] for o in m.outputs]
        if m.family == "gbt":
            self.trees = _Trees(p["base"], p["trees"])
        if m.family == "gru" and not p["stream"]:
            raise NotImplementedError("windowed GRU is over the per-frame budget; only stream mode is supported here")
        self.reset()

    def reset(self):
        m, p, S = self.m, self.m.params, self.S
        if m.family == "rls":
            n = len(p["idx"]) + 1
            self.w = np.tile(p["w"].astype(np.float64), (S, 1))
            P0 = p["P"].copy() if p["P"] is not None else np.eye(n) * p["delta"]
            self.P = np.tile(P0[None], (S, 1, 1))
            self.qoff = np.tile(p["q_offsets"].astype(np.float64), (S, 1))
            self.var = np.full(S, p["var0"], np.float64)
            self.qmap = []
            for o in m.outputs:
                name = o[0]
                if name.startswith("t_p"):
                    q = int(name[3:-3]) / 100
                    self.qmap.append(int(np.argmin(np.abs(p["q_levels"] - q))))
                else:
                    self.qmap.append(-1)
        if m.family == "gru":
            self.h = np.zeros((S, p["Whh"].shape[1]))
        if m.family == "tcn":
            self.ring = np.zeros((S, m.history_len, D))
            self.enc = None

    # ---- helpers
    def xn(self, x):
        return (np.asarray(x, np.float32) - self.mean) * self.inv  # float32, like the engine

    def hn(self, h):
        return ((np.asarray(h, np.float32) - self.hmean) * self.hinv).astype(np.float64)

    def _gru_step(self, xin, h):
        p = self.m.params
        H = h.shape[1]
        gi = xin @ p["Wih"].astype(np.float64).T + p["bih"]
        gh = h @ p["Whh"].astype(np.float64).T + p["bhh"]
        r = _sig(gi[:, :H] + gh[:, :H])
        z = _sig(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        return (1 - z) * n + z * h

    def _tcn_encode(self, seq):
        """seq: (B, L, D) normalised history -> (B, C) last-step channels."""
        L = seq.shape[1]
        for c in self.m.params["convs"]:
            W = c["W"].astype(np.float64)
            cout, cin, k = W.shape
            out = np.broadcast_to(c["b"].astype(np.float64), (seq.shape[0], L, cout)).copy()
            for j in range(k):
                shift = (k - 1 - j) * c["dilation"]
                if shift < L:
                    out[:, shift:] += seq[:, :L - shift] @ W[:, :, j].T
            out = _act(c["act"], out)
            if c["residual"]:
                out = out + seq
            seq = out
        return seq[:, -1]

    def _finish(self, raw, x):
        """raw (R, n_out) -> outputs with base + transform (x: (R, F) raw features)."""
        out = np.empty_like(raw)
        xf = lambda i: np.asarray(x[:, i], np.float32).astype(np.float64)
        for o in range(self.n_out):
            v = raw[:, o]
            t = self.trans[o]
            if t == "mulexp":  # .spkm v2: y = x[base] * exp(raw)
                v = xf(self.base_idx[o]) * np.exp(v)
            else:
                if self.base_idx[o] >= 0:
                    v = v + xf(self.base_idx[o])
                if t == "sigmoid":
                    v = _sig(v)
                elif t == "exp":
                    v = np.exp(v)
                elif t == "softplus":
                    v = np.where(v > 20, v, np.log1p(np.exp(np.minimum(v, 20))))
            if self.add_idx[o] >= 0:
                v = v + xf(self.add_idx[o])
            out[:, o] = v
        return out

    # ---- protocol
    def begin_frame(self, active=None):
        if self.m.family == "tcn":
            ring = self.ring if active is None else self.ring[active]
            self.enc_rows = active
            self.enc = self._tcn_encode(ring)

    def predict(self, Xc, active=None):
        """Xc: (S', C, F) raw candidate rows for the active streams -> (S', C, n_out)."""
        m, p = self.m, self.m.params
        Xc = np.asarray(Xc)
        Sa, C, _ = Xc.shape
        x = Xc.reshape(Sa * C, F)
        xn32 = self.xn(x)
        xn = xn32.astype(np.float64)
        st = np.arange(self.S) if active is None else np.asarray(active)
        rep = np.repeat(st, C)
        if m.family == "rls":
            phi = np.concatenate([xn[:, p["idx"]], np.ones((len(xn), 1))], 1)
            mean = np.einsum("rn,rn->r", phi, self.w[rep])
            if p["base_idx"] >= 0:
                mean = mean + np.asarray(x[:, p["base_idx"]], np.float32).astype(np.float64)
            out = np.empty((len(x), self.n_out))
            for o, name in enumerate(oo[0] for oo in m.outputs):
                if name == "t_mean_ms":
                    out[:, o] = mean
                elif name == "p_miss":
                    sd = np.sqrt(np.maximum(self.var[rep], 1e-12))
                    out[:, o] = 0.5 * erfc((m.budget_ms - mean) / (sd * sqrt(2.0)))
                else:
                    out[:, o] = mean + self.qoff[rep, self.qmap[o]]
            return out.reshape(Sa, C, self.n_out)
        if m.family == "linear":
            raw = xn @ p["W"].astype(np.float64).T + p["b"]
        elif m.family == "mlp":
            raw = _dense(p["layers"], xn)
        elif m.family == "gbt":
            raw = self.trees.raw(xn32, self.n_out)
        elif m.family == "gru":
            raw = _dense(p["head"], np.concatenate([self.h[rep], xn], 1))
        else:  # tcn: enc rows follow the active order used in begin_frame
            enc = np.repeat(self.enc, C, axis=0)
            raw = _dense(p["head"], np.concatenate([enc, xn], 1))
        return self._finish(raw, x).reshape(Sa, C, self.n_out)

    def end_frame(self, x, h, y, active=None):
        m, p = self.m, self.m.params
        st = np.arange(self.S) if active is None else np.asarray(active)
        if m.family == "gru":
            self.h[st] = self._gru_step(self.hn(h), self.h[st])
        elif m.family == "tcn":
            r = self.ring[st]
            self.ring[st] = np.concatenate([r[:, 1:], self.hn(h)[:, None, :]], 1)
        elif m.family == "rls" and p["online"]:
            xn = self.xn(x).astype(np.float64)
            phi = np.concatenate([xn[:, p["idx"]], np.ones((len(xn), 1))], 1)
            mean = np.einsum("rn,rn->r", phi, self.w[st])
            if p["base_idx"] >= 0:
                mean = mean + np.asarray(x[:, p["base_idx"]], np.float32).astype(np.float64)
            e = np.asarray(y, np.float32).astype(np.float64) - mean
            ql = p["q_levels"][None, :]
            self.qoff[st] += p["gamma"] * ((e[:, None] > self.qoff[st]).astype(np.float64) - (1 - ql))
            self.var[st] = (1 - p["var_alpha"]) * self.var[st] + p["var_alpha"] * e * e
            P = self.P[st]
            Pphi = np.einsum("sij,sj->si", P, phi)
            denom = p["lam"] + np.einsum("si,si->s", phi, Pphi)
            k = Pphi / denom[:, None]
            self.w[st] += k * e[:, None]
            P = (P - k[:, :, None] * Pphi[:, None, :]) / p["lam"]
            tr = np.einsum("sii->s", P)
            sc = np.where(tr > p["max_trace"], p["max_trace"] / np.maximum(tr, 1e-300), 1.0)
            self.P[st] = P * sc[:, None, None]


def is_stateful(m):
    return m.family in ("rls", "gru", "tcn")


def predict_split(m, X, H, y, offsets, chunk=65536):
    """Open-loop predictions for every row of a split, following the engine's per-frame protocol
    (sequence/online state per session, reset at session start). Returns (N, n_out) float64."""
    N = len(X)
    if not is_stateful(m):
        P = StepPredictor(m, 1)
        out = np.empty((N, len(m.outputs)))
        for a in range(0, N, chunk):
            out[a:a + chunk] = P.predict(X[a:a + chunk][:, None, :])[:, 0]
        return out
    starts, ends = offsets[:-1], offsets[1:]
    S = len(starts)
    lens = ends - starts
    out = np.empty((N, len(m.outputs)))
    if m.family == "tcn":  # vectorised over all rows: window of the L previous history vectors
        L = m.history_len
        P = StepPredictor(m, 1)
        Hn = P.hn(H)
        row_start = np.repeat(starts, lens)
        for a in range(0, N, chunk // max(1, L // 4)):
            rows = np.arange(a, min(N, a + chunk // max(1, L // 4)))
            idx = rows[:, None] - L + np.arange(L)[None, :]
            valid = idx >= row_start[rows][:, None]
            seq = np.where(valid[:, :, None], Hn[np.maximum(idx, 0)], 0.0)
            P.enc = P._tcn_encode(seq)
            out[rows] = P.predict(X[rows][:, None, :], active=np.zeros(len(rows), np.int64))[:, 0]
        return out
    P = StepPredictor(m, S)  # rls / gru stream: lockstep over sessions
    for t in range(int(lens.max())):
        act = np.nonzero(lens > t)[0]
        rows = starts[act] + t
        P.begin_frame(act)
        out[rows] = P.predict(X[rows][:, None, :], act)[:, 0]
        P.end_frame(X[rows], H[rows], y[rows], act)
    return out


def predict_dict(m, X, H, y, offsets):
    Y = predict_split(m, X, H, y, offsets)
    return {o[0]: Y[:, i] for i, o in enumerate(m.outputs)}
