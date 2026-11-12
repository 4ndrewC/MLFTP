"""The .spkm weight format: writer, reader and numpy reference implementation.

research/frametime/MODEL_CONTRACT.md is the specification. This module is the
Python half of it; engine/predictor.cpp is the C++ half, and
research/frametime/tests/test_spkm_parity.py checks they agree.

Typical export (phase 2, after training on the cluster):

    import spkm
    m = spkm.mlp([(W1, b1, "relu"), (W2, b2, "relu"), (W3, b3, "identity")],
                 outputs=[("t_p50_ms", "identity", "-"), ("t_p95_ms", "identity", "-"),
                          ("p_miss", "sigmoid", "-")],
                 norm=(mean, std))
    spkm.write(m, "checkpoints/<run_id>/model.spkm")

Weights are stored as float32 (RLS state as float64). numpy only; no ML
framework is needed to write, read or evaluate a model.
"""
import struct
import zlib

import numpy as np

SPKM_VERSION = 2  # v2: feature set spark-ft-v2; OUTS transform "mulexp" and an optional 4th "add" column
FEATURE_SET = "spark-ft-v2"
FEATURE_SET_V1 = "spark-ft-v1"
# v1 feature names (phase-1 dataset, analysis only; the engine runs v2 models only).
FEATURE_NAMES_V1 = [
    "n_steps", "n_steps_active", "acc_ms", "hitstop_ms", "trauma",
    "scene_menu", "scene_play", "scene_overlay",
    "entities", "enemies", "enemy_k0", "enemy_k1", "enemy_k2", "enemy_k3", "enemy_k4",
    "projectiles", "enemy_projectiles", "pickups", "telegraphs", "effects", "popups",
    "tele_due_now", "tele_due_6", "tele_due_30", "burst_due",
    "particles_alive", "particles_streak", "particles_expiring", "wave", "wave_to_spawn",
    "q_render_scale", "q_render_mpx", "q_particle_density", "q_glow_scale",
    "est_raster_mpx", "est_glow_mpx", "est_particles", "naive_ms",
    "t_prev_ms", "t_prev2_ms", "t_ema_ms", "t_max8_ms", "t_steps_prev_ms", "t_render_prev_ms",
    "steps_prev", "cpu_ghz", "ctxsw_prev",
]
# Must match engine/features.cpp kFeatureNames (checked by the parity tests and the engine loader).
# v2 = the 47 v1 features (same order and meaning) + 17 new ones.
FEATURE_NAMES = FEATURE_NAMES_V1 + [
    "q_aa", "est_aa_mpx", "t_ref_steps_ms", "t_ref_render_ms", "t_ref_ms",
    "fp_noise_p99_ms", "fp_noise_p999_ms", "fp_sleep_p99_ms",
    "fp_log_k_step", "fp_log_k_rpx", "fp_log_cliff", "fp_log_k_glow", "fp_log_k_part", "fp_log_ghz",
    "fp_log_lat_l3", "fp_log_lat_dram", "fp_log_bw_dram",
]
HISTORY_EXTRA = ["h_t_total_ms", "h_t_steps_ms", "h_t_render_ms"]
F = len(FEATURE_NAMES)
D = F + len(HISTORY_EXTRA)
MAX_HISTORY = 64
FAMILIES = ("linear", "rls", "mlp", "gbt", "gru", "tcn")
ACTS = {"identity": 0, "relu": 1, "tanh": 2, "sigmoid": 3}
ACT_NAMES = {v: k for k, v in ACTS.items()}
TRANSFORMS = ("identity", "sigmoid", "exp", "softplus", "mulexp")


def feature_hash(feature_set=FEATURE_SET, names=FEATURE_NAMES, history=HISTORY_EXTRA):
    """FNV-1a 64 over the feature-set id and names, '\\n'-joined; 16 lowercase hex digits."""
    s = "\n".join([feature_set] + list(names) + list(history)).encode()
    h = 1469598103934665603
    for c in s:
        h ^= c
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


# ---------------------------------------------------------------- model spec

class Spkm:
    """In-memory model. `params` holds the family-specific arrays (see the builders below)."""

    def __init__(self, family, outputs, params, history_len=0, budget_ms=1000 / 60, norm=None,
                 hnorm=None, info=None):
        assert family in FAMILIES, family
        self.family = family
        # (name, transform, base) or (name, transform, base, add); base/add are feature names or "-".
        # transform "mulexp": y = x[base] * exp(raw) (base required); add: y += x[add] after the transform.
        self.outputs = [tuple(o) for o in outputs]
        for o in self.outputs:
            assert len(o) in (3, 4) and o[1] in TRANSFORMS, o
            assert o[1] != "mulexp" or o[2] != "-", "mulexp needs a base feature"
        self.params = params
        self.history_len = int(history_len)
        self.budget_ms = float(budget_ms)
        self.norm = None if norm is None else (np.asarray(norm[0], np.float32), np.asarray(norm[1], np.float32))
        self.hnorm = None if hnorm is None else (np.asarray(hnorm[0], np.float32), np.asarray(hnorm[1], np.float32))
        self.info = dict(info or {})

    def num_parameters(self):
        p = self.params
        f = self.family
        if f == "linear":
            return p["W"].size + p["b"].size
        if f == "mlp":
            return sum(W.size + b.size for W, b, _ in p["layers"])
        if f == "gbt":
            return len(p["base"]) + sum(2 * len(nodes) for _, nodes in p["trees"])
        if f == "rls":
            n = len(p["idx"]) + 1
            return n + n * n + 2 * len(p["q_levels"])
        head = sum(W.size + b.size for W, b, _ in p["head"])
        if f == "gru":
            return p["Wih"].size + p["Whh"].size + p["bih"].size + p["bhh"].size + head
        return sum(c["W"].size + c["b"].size for c in p["convs"]) + head


def _layers(layers):
    return [(np.asarray(W, np.float32), np.asarray(b, np.float32), a) for W, b, a in layers]


def linear(W, b, outputs, **kw):
    """y = W @ xn + b, W shape (n_out, F)."""
    return Spkm("linear", outputs, {"W": np.asarray(W, np.float32), "b": np.asarray(b, np.float32)}, **kw)


def mlp(layers, outputs, **kw):
    """layers: [(W (out,in), b (out,), act), ...]; first in = F."""
    return Spkm("mlp", outputs, {"layers": _layers(layers)}, **kw)


def gbt(base, trees, outputs, **kw):
    """base: (n_out,) start values; trees: [(out_index, nodes)], nodes: list of
    (feature, threshold, left, right, value, default_left). feature = -1 marks a leaf.
    Node 0 is the root; children must have larger indices than their parent.
    Go left when xn[feature] <= threshold (NaN goes left if default_left)."""
    return Spkm("gbt", outputs, {"base": np.asarray(base, np.float32), "trees": list(trees)}, **kw)


def rls(idx, w, outputs, lam=0.999, delta=100.0, max_trace=1e6, online=True, P=None, base_idx=-1,
        q_levels=(), q_offsets=(), gamma=0.01, var0=1.0, var_alpha=0.01, **kw):
    """Online RLS mean model over xn[idx] plus a bias, with online quantile offsets and a
    Gaussian p_miss (see MODEL_CONTRACT.md). w has len(idx)+1 entries (bias last)."""
    n = len(idx) + 1
    return Spkm("rls", outputs, {
        "idx": np.asarray(idx, np.uint32), "base_idx": int(base_idx), "lam": float(lam), "delta": float(delta),
        "max_trace": float(max_trace), "online": bool(online), "w": np.asarray(w, np.float64).reshape(n),
        "P": None if P is None else np.asarray(P, np.float64).reshape(n, n),
        "q_levels": np.asarray(q_levels, np.float64), "q_offsets": np.asarray(q_offsets, np.float64),
        "gamma": float(gamma), "var0": float(var0), "var_alpha": float(var_alpha)}, **kw)


def gru(Wih, Whh, bih, bhh, head, outputs, history_len, stream=False, **kw):
    """PyTorch nn.GRU (1 layer) weights: Wih (3H, D), Whh (3H, H), gate order r, z, n.
    head: dense layers on concat(h, xn) -> outputs."""
    return Spkm("gru", outputs, {"Wih": np.asarray(Wih, np.float32), "Whh": np.asarray(Whh, np.float32),
                                 "bih": np.asarray(bih, np.float32), "bhh": np.asarray(bhh, np.float32),
                                 "stream": bool(stream), "head": _layers(head)}, history_len=history_len, **kw)


def tcn(convs, head, outputs, history_len, **kw):
    """convs: [dict(W=(cout,cin,k), b=(cout,), dilation=d, act="relu", residual=False)], causal,
    PyTorch Conv1d weight layout. head: dense layers on concat(last-step channels, xn)."""
    cs = [dict(W=np.asarray(c["W"], np.float32), b=np.asarray(c["b"], np.float32), dilation=int(c.get("dilation", 1)),
               act=c.get("act", "relu"), residual=bool(c.get("residual", False))) for c in convs]
    return Spkm("tcn", outputs, {"convs": cs, "head": _layers(head)}, history_len=history_len, **kw)


# ---------------------------------------------------------------- writing

def _chunk(tag, payload):
    assert len(tag) == 4
    pad = (-len(payload)) % 4
    return tag.encode() + struct.pack("<I", len(payload)) + payload + b"\0" * pad


def _dense_chunk(W, b, act):
    W = np.asarray(W, np.float32)
    n_out, n_in = W.shape
    return _chunk("DENS", struct.pack("<3I", n_in, n_out, ACTS[act]) + W.tobytes() + np.asarray(b, np.float32).tobytes())


def to_bytes(m: Spkm):
    info = {"family": m.family, "feature_set": FEATURE_SET, "feature_hash": feature_hash(), "n_features": F,
            "history_len": m.history_len, "history_dim": D, "budget_ms": repr(m.budget_ms)}
    info.update({k: v for k, v in m.info.items() if k not in info})
    for k, v in info.items():
        assert "\n" not in str(k) + str(v) and "=" not in str(k), (k, v)
    chunks = [_chunk("INFO", "".join(f"{k}={v}\n" for k, v in info.items()).encode()),
              _chunk("FEAT", ("\n".join(FEATURE_NAMES + HISTORY_EXTRA) + "\n").encode()),
              _chunk("OUTS", "".join(" ".join(o) + "\n" for o in m.outputs).encode())]
    if m.norm is not None:
        chunks.append(_chunk("NORM", m.norm[0].astype(np.float32).tobytes() + m.norm[1].astype(np.float32).tobytes()))
    if m.hnorm is not None:
        chunks.append(_chunk("HNRM", m.hnorm[0].astype(np.float32).tobytes() + m.hnorm[1].astype(np.float32).tobytes()))
    p = m.params
    if m.family == "linear":
        W = p["W"]
        chunks.append(_chunk("LINR", struct.pack("<2I", W.shape[0], W.shape[1]) + W.tobytes() + p["b"].tobytes()))
    elif m.family == "mlp":
        chunks += [_dense_chunk(*l) for l in p["layers"]]
    elif m.family == "gbt":
        body = struct.pack("<I", len(p["base"])) + p["base"].tobytes() + struct.pack("<I", len(p["trees"]))
        node_dt = np.dtype([("f", "<i4"), ("t", "<f4"), ("l", "<i4"), ("r", "<i4"), ("v", "<f4"), ("fl", "<u4")])
        for out, nodes in p["trees"]:
            arr = np.array([(int(f), t, int(l), int(r), v, 1 if dl else 0) for f, t, l, r, v, dl in nodes], node_dt)
            body += struct.pack("<2I", out, len(arr)) + arr.tobytes()
        chunks.append(_chunk("GBTE", body))
    elif m.family == "rls":
        n = len(p["idx"]) + 1
        body = struct.pack("<I", len(p["idx"])) + p["idx"].astype("<u4").tobytes()
        body += struct.pack("<i3d2I", p["base_idx"], p["lam"], p["delta"], p["max_trace"], int(p["online"]),
                            0 if p["P"] is None else 1)
        body += p["w"].astype("<f8").tobytes()
        if p["P"] is not None:
            body += p["P"].astype("<f8").reshape(n * n).tobytes()
        body += struct.pack("<I", len(p["q_levels"])) + p["q_levels"].astype("<f8").tobytes() + \
            p["q_offsets"].astype("<f8").tobytes()
        body += struct.pack("<3d", p["gamma"], p["var0"], p["var_alpha"])
        chunks.append(_chunk("RLS_", body))
    elif m.family == "gru":
        H = p["Whh"].shape[1]
        chunks.append(_chunk("GRU_", struct.pack("<3I", p["Wih"].shape[1], H, int(p["stream"])) + p["Wih"].tobytes() +
                             p["Whh"].tobytes() + p["bih"].tobytes() + p["bhh"].tobytes()))
        chunks += [_dense_chunk(*l) for l in p["head"]]
    elif m.family == "tcn":
        for c in p["convs"]:
            cout, cin, k = c["W"].shape
            chunks.append(_chunk("TCNL", struct.pack("<6I", cin, cout, k, c["dilation"], ACTS[c["act"]],
                                                     int(c["residual"])) + c["W"].tobytes() + c["b"].tobytes()))
        chunks += [_dense_chunk(*l) for l in p["head"]]
    body = b"".join(chunks)
    return b"SPKM" + struct.pack("<3I", SPKM_VERSION, len(chunks), zlib.crc32(body) & 0xFFFFFFFF) + body


def write(m: Spkm, path):
    data = to_bytes(m)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


# ---------------------------------------------------------------- reading

def from_bytes(data):
    if data[:4] != b"SPKM":
        raise ValueError("not a .spkm file")
    version, n_chunks, crc = struct.unpack_from("<3I", data, 4)
    if version != SPKM_VERSION:
        raise ValueError(f"unsupported .spkm version {version}")
    if zlib.crc32(data[16:]) & 0xFFFFFFFF != crc:
        raise ValueError("CRC mismatch: truncated or corrupt")
    pos, chunks = 16, []
    for _ in range(n_chunks):
        tag = data[pos:pos + 4].decode()
        (size,) = struct.unpack_from("<I", data, pos + 4)
        chunks.append((tag, data[pos + 8:pos + 8 + size]))
        pos += 8 + size + (-size) % 4
    get = lambda t: next((c for tg, c in chunks if tg == t), None)
    info = dict(l.split("=", 1) for l in get("INFO").decode().splitlines() if "=" in l)
    names = [l for l in get("FEAT").decode().splitlines() if l]
    if names != FEATURE_NAMES + HISTORY_EXTRA or info.get("feature_hash") != feature_hash():
        raise ValueError("feature list/hash does not match this spkm.py")
    outputs = [tuple(l.split()) for l in get("OUTS").decode().splitlines() if l]
    f32 = lambda b, off, n: (np.frombuffer(b, "<f4", n, off).copy(), off + 4 * n)

    def norm(tag, n):
        c = get(tag)
        return None if c is None else (np.frombuffer(c, "<f4", n, 0).copy(), np.frombuffer(c, "<f4", n, 4 * n).copy())

    def dense_layers():
        out = []
        for tg, c in chunks:
            if tg == "DENS":
                n_in, n_out, act = struct.unpack_from("<3I", c, 0)
                W, o = f32(c, 12, n_in * n_out)
                b, _ = f32(c, o, n_out)
                out.append((W.reshape(n_out, n_in), b, ACT_NAMES[act]))
        return out

    fam = info["family"]
    kw = dict(history_len=int(info.get("history_len", 0)), budget_ms=float(info.get("budget_ms", 1000 / 60)),
              norm=norm("NORM", F), hnorm=norm("HNRM", D),
              info={k: v for k, v in info.items() if k not in ("family", "feature_set", "feature_hash", "n_features",
                                                                "history_len", "history_dim", "budget_ms")})
    if fam == "linear":
        c = get("LINR")
        n_out, n_in = struct.unpack_from("<2I", c, 0)
        W, o = f32(c, 8, n_out * n_in)
        b, _ = f32(c, o, n_out)
        return linear(W.reshape(n_out, n_in), b, outputs, **kw)
    if fam == "mlp":
        return mlp(dense_layers(), outputs, **kw)
    if fam == "gbt":
        c = get("GBTE")
        (n_out,) = struct.unpack_from("<I", c, 0)
        base, o = f32(c, 4, n_out)
        (n_trees,) = struct.unpack_from("<I", c, o)
        o += 4
        trees = []
        node_dt = np.dtype([("f", "<i4"), ("t", "<f4"), ("l", "<i4"), ("r", "<i4"), ("v", "<f4"), ("fl", "<u4")])
        for _ in range(n_trees):
            out, nn = struct.unpack_from("<2I", c, o)
            arr = np.frombuffer(c, node_dt, nn, o + 8)
            o += 8 + 24 * nn
            trees.append((out, [(int(a["f"]), float(a["t"]), int(a["l"]), int(a["r"]), float(a["v"]), bool(a["fl"] & 1))
                                for a in arr]))
        return gbt(base, trees, outputs, **kw)
    if fam == "rls":
        c = get("RLS_")
        (ni,) = struct.unpack_from("<I", c, 0)
        idx = np.frombuffer(c, "<u4", ni, 4).copy()
        o = 4 + 4 * ni
        base_idx, lam, delta, max_trace, online, has_p = struct.unpack_from("<i3d2I", c, o)
        o += struct.calcsize("<i3d2I")
        n = ni + 1
        w = np.frombuffer(c, "<f8", n, o).copy()
        o += 8 * n
        P = None
        if has_p:
            P = np.frombuffer(c, "<f8", n * n, o).copy().reshape(n, n)
            o += 8 * n * n
        (nq,) = struct.unpack_from("<I", c, o)
        o += 4
        ql = np.frombuffer(c, "<f8", nq, o).copy()
        qo = np.frombuffer(c, "<f8", nq, o + 8 * nq).copy()
        o += 16 * nq
        gamma, var0, var_alpha = struct.unpack_from("<3d", c, o)
        return rls(idx, w, outputs, lam, delta, max_trace, bool(online), P, base_idx, ql, qo, gamma, var0, var_alpha,
                   **kw)
    if fam == "gru":
        c = get("GRU_")
        d, H, stream = struct.unpack_from("<3I", c, 0)
        Wih, o = f32(c, 12, 3 * H * d)
        Whh, o = f32(c, o, 3 * H * H)
        bih, o = f32(c, o, 3 * H)
        bhh, o = f32(c, o, 3 * H)
        return gru(Wih.reshape(3 * H, d), Whh.reshape(3 * H, H), bih, bhh, dense_layers(), outputs,
                   stream=bool(stream), **kw)
    if fam == "tcn":
        convs = []
        for tg, c in chunks:
            if tg == "TCNL":
                cin, cout, k, dil, act, res = struct.unpack_from("<6I", c, 0)
                W, o = f32(c, 24, cout * cin * k)
                b, _ = f32(c, o, cout)
                convs.append(dict(W=W.reshape(cout, cin, k), b=b, dilation=dil, act=ACT_NAMES[act], residual=bool(res)))
        return tcn(convs, dense_layers(), outputs, **kw)
    raise ValueError(f"unknown family {fam}")


def read(path):
    with open(path, "rb") as f:
        return from_bytes(f.read())


# ---------------------------------------------------------------- reference forward pass

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
        v = _act(a, W.astype(np.float64) @ v + b)
    return v


def _sig(v):
    return 1 / (1 + np.exp(-v))


class Runner:
    """Numpy reference of engine/predictor.cpp, following the same per-frame protocol:

        r.begin_frame(history)     history: list of D-vectors, oldest first (the engine keeps 64)
        r.predict(x)               -> outputs (any number of calls per frame)
        r.end_frame(x, h, y_ms)    after the frame: online updates / streaming state

    Normalisation is done in float32 exactly like the engine; everything else in float64.
    """

    def __init__(self, m: Spkm):
        self.m = m
        p = m.params
        mean, std = m.norm if m.norm is not None else (np.zeros(F, np.float32), np.ones(F, np.float32))
        self.mean, self.inv = mean.astype(np.float32), (np.float32(1) / std.astype(np.float32))
        hm, hs = m.hnorm if m.hnorm is not None else (np.zeros(D, np.float32), np.ones(D, np.float32))
        self.hmean, self.hinv = hm.astype(np.float32), (np.float32(1) / hs.astype(np.float32))
        self.enc = None
        if m.family == "rls":
            n = len(p["idx"]) + 1
            self.w = p["w"].astype(np.float64).copy()
            self.P = p["P"].copy() if p["P"] is not None else np.eye(n) * p["delta"]
            self.qoff = p["q_offsets"].astype(np.float64).copy()
            self.var = p["var0"]
        if m.family == "gru":
            self.h = np.zeros(p["Whh"].shape[1])

    def _xn(self, x):
        return ((np.asarray(x, np.float32) - self.mean) * self.inv).astype(np.float64)

    def _hn(self, h):
        return ((np.asarray(h, np.float32) - self.hmean) * self.hinv).astype(np.float64)

    def _gru_step(self, xin, h):
        p = self.m.params
        H = len(h)
        gi = p["Wih"].astype(np.float64) @ xin + p["bih"]
        gh = p["Whh"].astype(np.float64) @ h + p["bhh"]
        r = _sig(gi[:H] + gh[:H])
        z = _sig(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        return (1 - z) * n + z * h

    def begin_frame(self, history):
        m, p = self.m, self.m.params
        if m.family == "gru" and not p["stream"]:
            h = np.zeros(p["Whh"].shape[1])
            for v in history[-m.history_len:]:
                h = self._gru_step(self._hn(v), h)
            self.h = h
        elif m.family == "tcn":
            L = m.history_len
            seq = np.zeros((L, D))
            hist = history[-L:]
            for i, v in enumerate(hist):
                seq[L - len(hist) + i] = self._hn(v)
            for c in p["convs"]:
                W = c["W"].astype(np.float64)
                cout, cin, k = W.shape
                out = np.tile(c["b"].astype(np.float64), (L, 1))
                for j in range(k):
                    shift = (k - 1 - j) * c["dilation"]
                    if shift < L:
                        out[shift:] += seq[:L - shift] @ W[:, :, j].T
                out = _act(c["act"], out)
                if c["residual"]:
                    out = out + seq
                seq = out
            self.enc = seq[-1]

    def _rls_phi(self, xn):
        return np.append(xn[self.m.params["idx"]], 1.0)

    def _rls_mean(self, xn, x):
        p = self.m.params
        mval = float(self.w @ self._rls_phi(xn))
        if p["base_idx"] >= 0:
            mval += float(np.float32(x[p["base_idx"]]))
        return mval

    def predict(self, x):
        m, p = self.m, self.m.params
        xn = self._xn(x)
        if m.family == "rls":
            from math import erfc, sqrt
            mean = self._rls_mean(xn, x)
            out = []
            for o in m.outputs:
                name = o[0]
                if name == "t_mean_ms":
                    out.append(mean)
                elif name == "p_miss":
                    out.append(0.5 * erfc((m.budget_ms - mean) / (sqrt(max(self.var, 1e-12)) * sqrt(2.0))))
                else:
                    q = int(name[3:-3]) / 100
                    out.append(mean + self.qoff[int(np.argmin(np.abs(p["q_levels"] - q)))])
            return np.array(out)
        if m.family == "linear":
            raw = p["W"].astype(np.float64) @ xn + p["b"]
        elif m.family == "mlp":
            raw = _dense(p["layers"], xn)
        elif m.family == "gbt":
            raw = p["base"].astype(np.float64).copy()
            xn32 = (np.asarray(x, np.float32) - self.mean) * self.inv
            for out, nodes in p["trees"]:
                i = 0
                while nodes[i][0] >= 0:
                    f, t, l, r, _, dl = nodes[i]
                    v = xn32[f]
                    i = (l if dl else r) if np.isnan(v) else (l if v <= np.float32(t) else r)
                raw[out] += nodes[i][4]
        elif m.family == "gru":
            raw = _dense(p["head"], np.concatenate([self.h, xn]))
        else:
            raw = _dense(p["head"], np.concatenate([self.enc, xn]))
        res = []
        xf = lambda name: float(np.float32(x[FEATURE_NAMES.index(name)]))
        for v, o in zip(raw, m.outputs):
            name, tr, base = o[:3]
            add = o[3] if len(o) > 3 else "-"
            if tr == "mulexp":
                v = xf(base) * np.exp(v)
            else:
                if base != "-":
                    v = v + xf(base)
                if tr == "sigmoid":
                    v = _sig(v)
                elif tr == "exp":
                    v = np.exp(v)
                elif tr == "softplus":
                    v = v if v > 20 else np.log1p(np.exp(v))
            if add != "-":
                v = v + xf(add)
            res.append(v)
        return np.array(res)

    def end_frame(self, x, h, y_ms):
        m, p = self.m, self.m.params
        if m.family == "gru" and p["stream"]:
            self.h = self._gru_step(self._hn(h), self.h)
        if m.family != "rls" or not p["online"]:
            return
        xn = self._xn(x)
        phi = self._rls_phi(xn)
        e = float(np.float32(y_ms)) - self._rls_mean(xn, x)
        self.qoff += p["gamma"] * ((e > self.qoff).astype(np.float64) - (1 - p["q_levels"]))
        self.var = (1 - p["var_alpha"]) * self.var + p["var_alpha"] * e * e
        Pphi = self.P @ phi
        denom = p["lam"] + phi @ Pphi
        k = Pphi / denom
        self.w = self.w + k * e
        self.P = (self.P - np.outer(k, Pphi)) / p["lam"]
        tr = np.trace(self.P)
        if tr > p["max_trace"]:
            self.P *= p["max_trace"] / tr


# ---------------------------------------------------------------- random models (tests/benchmarks only)

DEFAULT_OUTPUTS = [("t_p50_ms", "identity", "-"), ("t_p95_ms", "identity", "-"), ("t_p99_ms", "identity", "-"),
                   ("p_miss", "sigmoid", "-")]


def random_norm(rng, n):
    return rng.uniform(-3, 3, n).astype(np.float32), rng.uniform(0.5, 3, n).astype(np.float32)


def _rand_layers(rng, sizes, hidden_act="relu"):
    layers = []
    for i in range(len(sizes) - 1):
        W = rng.normal(0, 1 / np.sqrt(sizes[i]), (sizes[i + 1], sizes[i]))
        layers.append((W, rng.normal(0, 0.1, sizes[i + 1]), hidden_act if i < len(sizes) - 2 else "identity"))
    return layers


def _rand_tree(rng, depth, n_feat):
    nodes = []

    def build(d):
        i = len(nodes)
        nodes.append(None)
        if d == 0 or rng.random() < 0.1:
            nodes[i] = (-1, 0.0, 0, 0, float(rng.normal(0, 0.3)), False)
        else:
            f, t, dl = int(rng.integers(n_feat)), float(rng.normal(0, 1)), bool(rng.random() < 0.5)
            l = build(d - 1)
            r = build(d - 1)
            nodes[i] = (f, t, l, r, 0.0, dl)
        return i

    build(depth)
    return nodes


def random_model(family, rng, size="small", outputs=None, budget_ms=1000 / 60):
    """Random-weight model for parity tests and inference benchmarks (never trained)."""
    outs = list(outputs or DEFAULT_OUTPUTS)
    no = len(outs)
    big = size == "bench"
    kw = dict(norm=random_norm(rng, F), budget_ms=budget_ms, info={"note": "random weights (test/bench only)"})
    if family == "linear":
        return linear(rng.normal(0, 0.3, (no, F)), rng.normal(0, 1, no), outs, **kw)
    if family == "mlp":
        return mlp(_rand_layers(rng, [F, 64, 64, no] if big else [F, 24, 16, no]), outs, **kw)
    if family == "gbt":
        n_trees, depth = (300, 4) if big else (40, 5)
        trees = [(int(t % no), _rand_tree(rng, depth, F)) for t in range(n_trees)]
        return gbt(rng.normal(1, 0.5, no), trees, outs, **kw)
    if family == "rls":
        idx = rng.choice(F, 16 if big else 10, replace=False)
        outs = [("t_mean_ms", "identity", "-"), ("t_p95_ms", "identity", "-"), ("t_p99_ms", "identity", "-"),
                ("p_miss", "identity", "-")]
        return rls(idx, rng.normal(0, 0.5, len(idx) + 1), outs, lam=0.995, delta=10.0, max_trace=1e4,
                   base_idx=FEATURE_NAMES.index("naive_ms"), q_levels=[0.95, 0.99], q_offsets=[0.5, 1.0],
                   gamma=0.02, var0=1.0, var_alpha=0.02, **kw)
    hn = random_norm(rng, D)
    if family == "gru":
        H = 32 if big else 12
        s = 1 / np.sqrt(H)
        return gru(rng.uniform(-s, s, (3 * H, D)), rng.uniform(-s, s, (3 * H, H)), rng.uniform(-s, s, 3 * H),
                   rng.uniform(-s, s, 3 * H), _rand_layers(rng, [H + F, 32 if big else 16, no]), outs,
                   history_len=32 if big else 16, stream=False, hnorm=hn, **kw)
    if family == "mlp_mulexp":  # LIT_REVIEW_HW options A/B: y = t_ref * exp(raw) [+ noise term]
        outs = [("t_p50_ms", "mulexp", "t_ref_ms"), ("t_p95_ms", "mulexp", "t_ref_ms", "fp_noise_p99_ms"),
                ("t_p99_ms", "mulexp", "t_ref_ms", "fp_noise_p999_ms"), ("p_miss", "sigmoid", "-")]
        return mlp(_rand_layers(rng, [F, 64, 64, 4] if big else [F, 16, 4]), outs, **kw)
    if family == "gru_stream":
        m = random_model("gru", rng, size, outputs, budget_ms)
        m.params["stream"] = True
        return m
    if family == "tcn":
        C = 32 if big else 8
        dils = [1, 2, 4] if big else [1, 2]
        convs, cin = [], D
        for i, d in enumerate(dils):
            convs.append(dict(W=rng.normal(0, 1 / np.sqrt(cin * 3), (C, cin, 3)), b=rng.normal(0, 0.1, C), dilation=d,
                              act="relu", residual=i > 0))
            cin = C
        return tcn(convs, _rand_layers(rng, [C + F, 32 if big else 16, no]), outs, history_len=32 if big else 12,
                   hnorm=hn, **kw)
    raise ValueError(family)


# ---------------------------------------------------------------- checkpoints (checkpoints/README.md)

# (source, description) per feature, for manifest.json. Sources: planned | quality | history | system.
FEATURE_INFO = {
    "n_steps": ("planned", "fixed sim steps this frame will run"),
    "n_steps_active": ("planned", "sim steps not frozen by hit-stop"),
    "acc_ms": ("planned", "fixed-step accumulator before the steps (ms)"),
    "hitstop_ms": ("planned", "remaining hit-stop (ms)"),
    "trauma": ("planned", "screen-shake trauma 0..1"),
    "scene_menu": ("planned", "top scene is the title screen"),
    "scene_play": ("planned", "top scene is gameplay"),
    "scene_overlay": ("planned", "top scene is pause/game over"),
    "entities": ("planned", "live ECS entities"),
    "enemies": ("planned", "live enemies"),
    "enemy_k0": ("planned", "Drifter enemies"),
    "enemy_k1": ("planned", "Dart enemies"),
    "enemy_k2": ("planned", "Splitter enemies"),
    "enemy_k3": ("planned", "Mini enemies"),
    "enemy_k4": ("planned", "Spinner enemies"),
    "projectiles": ("planned", "player bullets"),
    "enemy_projectiles": ("planned", "enemy shots"),
    "pickups": ("planned", "sparks"),
    "telegraphs": ("planned", "spawn warnings alive"),
    "effects": ("planned", "ring effects alive"),
    "popups": ("planned", "score popups alive"),
    "tele_due_now": ("planned", "warnings spawning during this frame's steps"),
    "tele_due_6": ("planned", "warnings expiring within 6 steps"),
    "tele_due_30": ("planned", "warnings expiring within 30 steps"),
    "burst_due": ("planned", "particles of known bursts this frame, before quality density"),
    "particles_alive": ("planned", "live particles"),
    "particles_streak": ("planned", "live particles drawn as lines"),
    "particles_expiring": ("planned", "particles dying in the next update"),
    "wave": ("planned", "wave number"),
    "wave_to_spawn": ("planned", "enemies still to be announced this wave"),
    "q_render_scale": ("quality", "candidate effective render scale (px per logical px)"),
    "q_render_mpx": ("quality", "candidate render buffer size (Mpx)"),
    "q_particle_density": ("quality", "candidate particle density multiplier"),
    "q_glow_scale": ("quality", "candidate glow radius multiplier (0 = off)"),
    "est_raster_mpx": ("quality", "previous frame's non-glow raster work rescaled to the candidate scale (Mpx)"),
    "est_glow_mpx": ("quality", "previous frame's full-quality glow area at the candidate scale and glow level (Mpx)"),
    "est_particles": ("quality", "live particles + known burst particles at the candidate density"),
    "naive_ms": ("quality", "steps-aware analytic estimate: steps x EMA(step) + EMA(ms/Mpx) x candidate pixel work"),
    "t_prev_ms": ("history", "t_total of the previous frame (ms)"),
    "t_prev2_ms": ("history", "t_total two frames back (ms)"),
    "t_ema_ms": ("history", "EMA(0.2) of t_total (ms)"),
    "t_max8_ms": ("history", "max t_total over the last 8 frames (ms)"),
    "t_steps_prev_ms": ("history", "sim-step time of the previous frame (ms)"),
    "t_render_prev_ms": ("history", "render time of the previous frame (ms)"),
    "steps_prev": ("history", "sim steps of the previous frame"),
    "cpu_ghz": ("system", "current core frequency (GHz)"),
    "ctxsw_prev": ("system", "involuntary context switches during the previous frame"),
    "q_aa": ("quality", "candidate anti-aliasing: 1 on, 0 off (aliased spans)"),
    "est_aa_mpx": ("quality", "previous frame's circle+ring+line bounding-box work at the candidate scale (Mpx)"),
    "t_ref_steps_ms": ("fingerprint", "reference sim time: steps x fingerprint ms/step at the current entity count"),
    "t_ref_render_ms": ("fingerprint", "reference render time: fingerprint ns/px per primitive class x candidate pixel work"),
    "t_ref_ms": ("fingerprint", "t_ref_steps_ms + t_ref_render_ms (LIT_REVIEW_HW option A normaliser)"),
    "fp_noise_p99_ms": ("fingerprint", "FWQ noise probe: p99 excess over the median 50 us quantum (ms)"),
    "fp_noise_p999_ms": ("fingerprint", "FWQ noise probe: p99.9 excess (ms)"),
    "fp_sleep_p99_ms": ("fingerprint", "sleep_for(1 ms) overshoot p99 (ms)"),
    "fp_log_k_step": ("fingerprint", "log10 sim kernel ms/step (normal load)"),
    "fp_log_k_rpx": ("fingerprint", "log10 canned-frame render ns per buffer pixel at out-scale 1"),
    "fp_log_cliff": ("fingerprint", "log10 cost ratio per pixel at out-scale 3.2 vs 1 (cache cliff)"),
    "fp_log_k_glow": ("fingerprint", "log10 glow ns per pixel"),
    "fp_log_k_part": ("fingerprint", "log10 particle update ns per particle"),
    "fp_log_ghz": ("fingerprint", "log10 effective GHz (dependent add chain)"),
    "fp_log_lat_l3": ("fingerprint", "log10 pointer-chase latency at 1.5 MB (ns)"),
    "fp_log_lat_dram": ("fingerprint", "log10 pointer-chase latency at 128 MB (ns)"),
    "fp_log_bw_dram": ("fingerprint", "log10 single-thread triad bandwidth, 96 MB (GB/s)"),
}
assert list(FEATURE_INFO) == FEATURE_NAMES

OUTPUT_INFO = {"t_mean_ms": ("ms", "expected t_total of the next frame"),
               "p_miss": ("probability", "P(next frame t_total > budget_ms)")}

CALL_SITE = ("Engine::beginFrame (engine/engine.cpp), once per loop iteration after the sim-step count "
             "is known and before steps/render; evaluated per candidate quality by the controller")


def write_checkpoint(run_dir, model: Spkm, dataset, metrics, architecture, created_utc=None, max_overhead_us=50,
                     norm_in_manifest=False):
    """Write model.spkm + manifest.json for a run folder (checkpoints/README.md). Does NOT write DONE:
    run tools/validate_checkpoint.py --no-done first, then create DONE."""
    import datetime
    import json
    import os
    os.makedirs(run_dir, exist_ok=True)
    run_id = os.path.basename(os.path.normpath(run_dir))
    created = created_utc or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    model.info.setdefault("run_id", run_id)
    model.info.setdefault("created_utc", created)
    write(model, os.path.join(run_dir, "model.spkm"))
    feats = []
    for i, n in enumerate(FEATURE_NAMES):
        src, desc = FEATURE_INFO[n]
        f = {"name": n, "source": src, "description": desc}
        if norm_in_manifest and model.norm is not None:  # documentation only: the engine reads NORM from model.spkm
            f["mean"], f["std"] = float(model.norm[0][i]), float(model.norm[1][i])
        feats.append(f)
    outs = []
    for o in model.outputs:
        name = o[0]
        unit, desc = OUTPUT_INFO.get(name, ("ms", f"{name[3:-3]}th percentile of the next frame's t_total"))
        outs.append({"name": name, "unit": unit, "description": desc})
    manifest = {
        "schema_version": 1, "run_id": run_id, "created_utc": created,
        "model": {"family": model.family, "architecture": architecture, "num_parameters": int(model.num_parameters()),
                  "engine_file": "model.spkm", "engine_format_version": SPKM_VERSION},
        "dataset": dataset,
        "inputs": {"feature_set": FEATURE_SET, "feature_hash": feature_hash(), "history_len": model.history_len,
                   "features": feats},
        "outputs": outs,
        "engine_hook": {"call_site": CALL_SITE, "predicts": "next_frame", "cadence_frames": 1,
                        "budget_ms": model.budget_ms, "max_overhead_us": max_overhead_us},
        "metrics": metrics,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
