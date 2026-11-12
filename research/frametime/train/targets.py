"""Pluggable target normalisers (LIT_REVIEW_HW section 3.3 option A).

A normaliser maps the frame time y (ms) to the space the model is trained in, and maps
predicted quantiles back to ms. Quantiles commute with monotone maps, so a quantile
learnt in log-ratio space is a valid quantile in ms after decoding.

    none               z = y                                    export: identity, no base
    residual           z = y - x[f]                             export: identity, base f (contract section 5)
    log_ratio          z = log(y / max(x[f], floor))            export: mulexp, base f (.spkm v2: y = x[f]*exp(raw));
                                                                 with .spkm v1 only via a log(t_ref) feature
    log_ratio_inverse  z = log(y * max(x[f], floor))            e.g. f = cpu_ghz: "cycles" (t x GHz)
    log_ratio_ref      z = log(y / t_ref), t_ref = sum_i x[f_i] * header[path_i]   (v2 fingerprint references)

Optional per-quantile tail term (LIT_REVIEW_HW option B, .spkm v2 OUTS "add" column):
    "add": {"0.95": "fp_noise_p99_ms", "0.99": "fp_noise_p999_ms"}
    -> t_q = decode(z_q) + x[add_q]. Each quantile is then trained on its own encoded target
    z_q = encode(max(y - x[add_q], eps)); the map is monotone in y, so quantiles carry over.

Config (configs/<feature_set>.json "normalisers"):
    {"type": "log_ratio_ref", "terms": [["q_render_mpx", "fingerprint.k_rpx"], ["n_steps", "fingerprint.k_step"]],
     "floor": 0.05, "log_feature": "log_t_ref"}
"""
import numpy as np

import spkm

_EPS_Y = 1e-3


def _lookup(header, path):
    v = header
    for k in path.split("."):
        if not isinstance(v, dict) or k not in v:
            raise KeyError(f"header has no {path!r} (needed by a log_ratio_ref normaliser)")
        v = v[k]
    return float(v)


class Target:
    def __init__(self, name, spec):
        self.name = name
        self.spec = dict(spec)
        self.kind = spec.get("type", "none")
        if self.kind not in ("none", "residual", "log_ratio", "log_ratio_inverse", "log_ratio_ref"):
            raise ValueError(f"unknown normaliser type {self.kind}")
        self.feature = spec.get("feature")
        if self.feature is not None and self.feature not in spkm.FEATURE_NAMES:
            raise SystemExit(f"normaliser {name}: feature {self.feature!r} not in {spkm.FEATURE_SET}")
        self.fi = spkm.FEATURE_NAMES.index(self.feature) if self.feature else None
        self.floor = float(spec.get("floor", 1e-3))
        self.log_feature = spec.get("log_feature")
        self.add = {float(k): v for k, v in spec.get("add", {}).items()}
        for v in self.add.values():
            if v not in spkm.FEATURE_NAMES:
                raise SystemExit(f"normaliser {name}: add feature {v!r} not in {spkm.FEATURE_SET}")

    def add_feature(self, q):
        return self.add.get(round(float(q), 4))

    def add_values(self, X, q):
        f = self.add_feature(q)
        return None if f is None else np.asarray(X)[:, spkm.FEATURE_NAMES.index(f)].astype(np.float32).astype(np.float64)

    def encode_q(self, y, X, ref, quantiles):
        """(N, k) training targets, one column per quantile (differs only with add terms)."""
        cols = []
        for q in quantiles:
            a = self.add_values(X, q)
            yy = np.asarray(y, np.float64) if a is None else np.maximum(np.asarray(y, np.float64) - a, _EPS_Y)
            cols.append(self.encode(yy, ref))
        return np.stack(cols, 1)

    def decode_q(self, Z, X, ref, quantiles):
        """(N, k) training-space quantiles -> ms, including add terms."""
        out = np.empty(np.shape(Z))
        for i, q in enumerate(quantiles):
            out[:, i] = self.decode(np.asarray(Z)[:, i], ref)
            a = self.add_values(X, q)
            if a is not None:
                out[:, i] += a
        return out

    # ---- reference
    def ref(self, X, split=None, rows=None):
        """Per-row reference (ms for ratios; additive base for residual). Shape (N,) float64."""
        X = np.asarray(X)
        if self.kind == "none":
            return np.zeros(len(X))
        if self.kind == "residual":
            return X[:, self.fi].astype(np.float64)
        if self.kind == "log_ratio":
            return np.maximum(X[:, self.fi].astype(np.float64), self.floor)
        if self.kind == "log_ratio_inverse":
            return 1.0 / np.maximum(X[:, self.fi].astype(np.float64), self.floor)
        # log_ratio_ref: per-session header constants x per-row feature values
        if split is None:
            raise ValueError("log_ratio_ref needs the split (session headers)")
        sess = split.sess if rows is None else split.sess[rows]
        ref = np.zeros(len(X))
        for fname, path in self.spec["terms"]:
            fi = spkm.FEATURE_NAMES.index(fname)
            per_sess = np.array([_lookup(s.header, path) for s in split.sessions])
            ref += X[:, fi].astype(np.float64) * per_sess[sess]
        return np.maximum(ref, self.floor)

    @property
    def is_log(self):
        return self.kind.startswith("log_ratio")

    def encode(self, y, ref):
        y = np.asarray(y, np.float64)
        if self.kind == "none":
            return y
        if self.kind == "residual":
            return y - ref
        return np.log(np.maximum(y, _EPS_Y) / ref)

    def engine_ref(self, X, split=None, rows=None):
        """The reference exactly as the engine applies it when the model is exported (raw feature value for
        `mulexp`, no floor); falls back to ref() for references the engine cannot express."""
        if self.kind == "log_ratio" and self.export() is not None:
            return np.asarray(X)[:, self.fi].astype(np.float32).astype(np.float64)
        return self.ref(X, split, rows)

    def check(self, X, split=None):
        """Refuse references that are (mostly) missing, e.g. t_ref_ms on logs without a fingerprint."""
        if self.kind in ("none", "residual"):
            return None
        raw = np.asarray(X)[:, self.fi] if self.fi is not None else self.ref(X, split)
        bad = float(np.mean(raw < self.floor)) if self.kind != "log_ratio_inverse" else 0.0
        if bad > 0.5:
            return (f"normaliser {self.name}: reference {self.feature or 'ref'} is below {self.floor} on "
                    f"{bad:.0%} of rows (e.g. no hardware fingerprint in these logs)")
        return None

    def decode(self, z, ref):
        """z: (N,) or (N, k) predictions in training space -> ms."""
        z = np.asarray(z, np.float64)
        r = ref[:, None] if z.ndim == 2 else ref
        if self.kind == "none":
            return z
        if self.kind == "residual":
            return z + r
        return np.exp(z) * r

    def export(self):
        """(transform, base) for the quantile outputs in OUTS, or None if the engine cannot express it."""
        if self.add and "mulexp" not in getattr(spkm, "TRANSFORMS", ()):
            return None  # the add column arrived with .spkm v2
        if self.kind == "none":
            return ("identity", "-")
        if self.kind == "residual":
            return ("identity", self.feature)
        if self.kind == "log_ratio" and "mulexp" in getattr(spkm, "TRANSFORMS", ()):
            return ("mulexp", self.feature)        # spkm v2: y = x[base] * exp(raw)
        if self.log_feature and self.log_feature in spkm.FEATURE_NAMES:
            return ("exp", self.log_feature)       # y = exp(raw + log t_ref)
        return None

    def export_reason(self):
        if self.export() is not None:
            return None
        return (f"{self.kind} target: the engine cannot apply this reference (no `mulexp` with a feature equal "
                f"to it, no log(t_ref) feature in {spkm.FEATURE_SET}). Evaluated offline only.")

    def describe(self):
        d = {"name": self.name, "type": self.kind}
        if self.add:
            d["add"] = {str(k): v for k, v in self.add.items()}
        if self.feature:
            d["feature"] = self.feature
        if self.kind.startswith("log_ratio"):
            d["floor"] = self.floor
        if self.kind == "log_ratio_ref":
            d["terms"] = self.spec["terms"]
        return d


def get_target(cfg, name):
    specs = cfg.get("normalisers", {})
    if name not in specs:
        if name == "none":
            return Target("none", {"type": "none"})
        raise SystemExit(f"unknown target normaliser {name!r}; configured: {sorted(specs)}")
    return Target(name, specs[name])
