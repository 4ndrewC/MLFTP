"""Data pipeline: sessions -> features (features.py, bit-exact with the engine) + targets,
split by the MANIFEST's session split, cached per session, concatenated per split.

Feature-set agnostic: names, counts and the history width come from spkm.py/features.py,
the levers from the config + the log header, the dataset from the config/CLI.

    ds = Dataset(cfg, splits=("train", "val", "test"))
    tr = ds.split("train")
    tr.X (N, F) float32 features      tr.y (N,) float32 t_total ms (the target)
    tr.H (N, D) float32 history vecs  tr.offsets session boundaries (rows)
    tr.aux[...] per-frame extras for evaluation and the controller simulator
    ds.counterfactual(session, quality) -> (n, F) features at another quality setting

Memory: float32 features, small ints for the rest; ~0.2 KB/frame in total, i.e. about
200 MB for the 0.94 M-frame laptop-v1 dataset (plus counterfactual rows on request).
"""
import hashlib
import json
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from common import DEFAULT_CACHE_DIR, TRAIN_DIR, log, resolve_path
import features
import framelog
import spkm

FEATURE_NAMES = list(spkm.FEATURE_NAMES)
F = spkm.F
D = spkm.D
HISTORY_NAMES = FEATURE_NAMES + list(spkm.HISTORY_EXTRA)

# Per-frame extras kept next to X/y (name -> (record field(s), dtype, scale)).
_AUX = {
    "t_steps": ("t_steps_us", np.float32, 1e-3),
    "t_render": ("t_render_us", np.float32, 1e-3),
    "t_r_particles": ("t_r_particles_us", np.float32, 1e-3),
    "n_steps": ("n_steps", np.uint8, None),
    "flags": ("flags", np.uint8, None),
    "step0": ("step0", np.uint32, None),
    "contend": ("contend", np.uint8, None),
    "render_scale": ("render_scale", np.float32, None),
    "px_clear": ("px_clear", np.float32, None),
    "px_glow_req": ("px_glow_req", np.float32, None),
    "cpu_mhz": ("cpu_mhz", np.uint16, None),
}
_RASTER = ("px_rect", "px_circle", "px_ring", "px_line", "px_poly")
_AA_PX = ("px_circle", "px_ring", "px_line")  # primitives drawn anti-aliased (the v2 AA lever)
STRESS = ["S0", "S1", "S2", "S3", "S4"]
LOOPS = ["emulate", "paced", "windowed", "fast"]
CORES = ["P", "E"]


CACHE_VERSION = 2  # bump when the cached per-session arrays change (e.g. new aux columns)


def feature_code_hash():
    """Cache key: changes whenever the feature code, the feature list or the cache layout changes."""
    h = hashlib.sha256(f"cache-v{CACHE_VERSION}".encode())
    for f in ("features.py", "framelog.py"):
        h.update((TRAIN_DIR / f).read_bytes())
    h.update(spkm.feature_hash().encode())
    return h.hexdigest()[:10]


def name_index(names):
    """Indices of the given feature names that exist in this feature set (unknown names ignored)."""
    pos = {n: i for i, n in enumerate(FEATURE_NAMES)}
    return [pos[n] for n in names if n in pos]


def feature_mask(cfg, spec):
    """Boolean mask over the feature set from a spec: 'all', 'group1+group2' (config feature_groups or
    ablation_groups; 'group?' = skip if the group does not exist), '-group' (all except), or a list of names. Features of the active set
    that no config group lists (e.g. new v2 features) go into the group of their spkm.FEATURE_INFO
    source, so masks keep working when the feature set grows."""
    if spec in (None, "all"):
        return np.ones(F, bool)
    if isinstance(spec, (list, tuple)):
        m = np.zeros(F, bool)
        m[name_index(spec)] = True
        return m
    groups = feature_groups(cfg)
    neg = spec.startswith("-")
    m = np.zeros(F, bool)
    for g in spec.lstrip("-").split("+"):
        optional = g.endswith("?")
        g = g.rstrip("?")
        if g not in groups:
            if optional:
                continue
            raise SystemExit(f"unknown feature group {g!r}; known: {sorted(groups)}")
        m[name_index(groups[g])] = True
    return ~m if neg else m


def feature_groups(cfg):
    groups = {k: list(v) for k, v in cfg.get("feature_groups", {}).items()}
    for k, v in cfg.get("ablation_groups", {}).items():
        groups.setdefault(k, list(v))
    listed = {n for k in cfg.get("feature_groups", {}) for n in groups[k]}
    src_to_group = {"planned": "planned", "quality": "planned", "history": "history", "system": "system",
                    "fingerprint": "fingerprint"}
    info = getattr(spkm, "FEATURE_INFO", {})
    for n in FEATURE_NAMES:  # unlisted features: by their declared source
        if n not in listed:
            g = src_to_group.get(info.get(n, ("other",))[0], "other")
            groups.setdefault(g, []).append(n)
            if "feature_groups" in cfg:
                log(f"note: feature {n!r} not in any config feature_group; using source group {g!r}")
    return groups


# ---------------------------------------------------------------- per-session loading

def _lever_columns(cfg, rec):
    """(n, n_levers) uint8 quality levels. A lever the log does not record (e.g. q_aa in v1 logs, where AA
    was always on) reads as level 0, which is what the engine used."""
    cols = []
    for lv in cfg["levers"]:
        if lv["field"] in rec.dtype.names:
            cols.append(rec[lv["field"]])
        else:
            cols.append(np.zeros(len(rec), np.uint8))
    return np.stack(cols, axis=1).astype(np.uint8)


def _session_arrays(path, cfg, max_frames):
    header, rec = framelog.read_log(path)
    if max_frames:
        rec = rec[:max_frames]
    X = features.compute_features(rec, header)
    Hv = features.history_vectors(rec, X)
    assert X.shape[1] == F and Hv.shape[1] == D, "feature width differs from spkm.py"
    out = {"X": X.astype(np.float32), "hx": Hv[:, F:].astype(np.float32), "y": Hv[:, F].astype(np.float32)}
    names = rec.dtype.names
    for k, (field, dt, scale) in _AUX.items():
        if field in names:
            v = rec[field].astype(np.float64)
            out[k] = (v * scale if scale else v).astype(dt)
        else:
            out[k] = np.zeros(len(rec), dt)
    out["raster_px"] = sum(rec[k].astype(np.float64) for k in _RASTER if k in names).astype(np.float32)
    out["px_aa"] = sum(rec[k].astype(np.float64) for k in _AA_PX if k in names).astype(np.float32)
    out["q"] = _lever_columns(cfg, rec)
    return header, out


def _cache_job(a):
    path, cpath, cfg, max_frames = a
    if os.path.exists(cpath):
        return cpath
    header, arrs = _session_arrays(path, cfg, max_frames)
    tmp = cpath + f".tmp{os.getpid()}.npz"
    np.savez(tmp, header=np.frombuffer(json.dumps(header).encode(), np.uint8), **arrs)
    os.replace(tmp, cpath)
    return cpath


VAL_FRACTION = 0.15
_CARVE_SALT = "spark-ft-val-carve-v1"  # changing it re-chooses the validation sessions


def _stable_key(session_id):
    return hashlib.sha1(f"{_CARVE_SALT}:{session_id}".encode()).hexdigest()


def check_supported_manifest(folder, manifest):
    """Only v2 logs (spark-framelog format_version 2, feature set spark-ft-v2) can be trained on. v1 logs
    (research/frametime/data/, phase 1) are not a training target."""
    fv = manifest.get("format_version")
    name = manifest.get("dataset", "?")
    if fv != 2 or str(name).startswith("laptop-v1"):
        raise SystemExit(f"{folder}: dataset {name!r} (format_version {fv}) is v1 data, NOT SUPPORTED for training. "
                         f"Training targets spark-ft-v2 only: use research/frametime/data-v2 "
                         f"(see TRAINING.md section 9).")


def carve_validation(sessions, fraction=VAL_FRACTION):
    """If a MANIFEST has no 'val' sessions, move about `fraction` of its 'train' sessions to 'val'.
    Stratified by the 'stress' field (each stress level with >= 2 train sessions keeps at least one val
    session), chosen by a stable hash of the session id, not by random state. Test sessions are never
    touched. Returns (sessions, info); info is None when the manifest already has val sessions."""
    if any(s["split"] == "val" for s in sessions):
        return sessions, None
    groups = {}
    for s in sessions:
        if s["split"] == "train":
            groups.setdefault(s.get("stress"), []).append(s)
    chosen, by_stress = set(), {}
    for st, lst in sorted(groups.items(), key=lambda kv: str(kv[0])):
        lst = sorted(lst, key=lambda s: _stable_key(s["id"]))
        k = max(1, round(fraction * len(lst))) if len(lst) >= 2 else 0
        chosen.update(s["id"] for s in lst[:k])
        by_stress[str(st)] = {"train": len(lst) - k, "val": k}
    out = [dict(s, split="val", carved_from="train") if s["id"] in chosen else s for s in sessions]
    n_train = sum(1 for s in sessions if s["split"] == "train")
    info = {"fraction": fraction, "salt": _CARVE_SALT, "train_before": n_train, "val": len(chosen),
            "train_after": n_train - len(chosen), "by_stress": by_stress}
    return out, info


class Session:
    def __init__(self, meta, arrs, header, path, max_frames):
        self.meta = meta
        self.id = meta["id"] if meta.get("dataset") is None else f"{meta['dataset']}/{meta['id']}"
        self.header = header
        self.path = path
        self.max_frames = max_frames
        self.a = arrs
        self.n = len(arrs["y"])
        self._cf = {}

    @property
    def stress(self):
        return self.meta.get("stress", "?")


def _empty_array(k, n_levers):
    """Zero-row array with the shape and dtype a non-empty split would have for key k."""
    if k == "X":
        return np.zeros((0, F), np.float32)
    if k == "hx":
        return np.zeros((0, D - F), np.float32)
    if k == "y":
        return np.zeros(0, np.float32)
    if k == "q":
        return np.zeros((0, n_levers), np.uint8)
    if k in ("raster_px", "px_aa"):
        return np.zeros(0, np.float32)
    return np.zeros(0, _AUX[k][1])


class SplitData:
    """All sessions of one split, concatenated (rows in session order). An empty split still has the
    right shapes: X (0, F), hx (0, D-F), q (0, n_levers)."""

    def __init__(self, name, sessions, n_levers=0):
        self.name = name
        self.sessions = sessions
        lens = [s.n for s in sessions]
        self.offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        self.N = int(self.offsets[-1])
        cat = lambda k: np.concatenate([s.a[k] for s in sessions]) if sessions else _empty_array(k, n_levers)
        self.X = cat("X")
        self.y = cat("y")
        self.hx = cat("hx")
        self.aux = {k: cat(k) for k in list(_AUX) + ["raster_px", "px_aa", "q"]}
        self.sess = np.repeat(np.arange(len(sessions), dtype=np.int32), lens)
        self.row_start = self.offsets[self.sess]          # first row of each row's session
        code = lambda vals, key: np.repeat(np.array([vals.index(s.meta.get(key)) if s.meta.get(key) in vals
                                                     else 255 for s in sessions], np.uint8), lens)
        self.stress = code(STRESS, "stress")
        self.loop = code(LOOPS, "loop")
        self.core = code(CORES, "core_type")
        self._H = None

    @property
    def H(self):
        """History vectors [x, t_total, t_steps, t_render] (N, D) float32 (MODEL_CONTRACT section 3)."""
        if self._H is None:
            self._H = np.concatenate([self.X, self.hx], axis=1)
        return self._H

    def frames(self):
        return self.N

    def bytes(self):
        return int(sum(s.meta.get("bytes", 0) for s in self.sessions))

    def subset_rows(self, step):
        return np.arange(0, self.N, step)


def parse_holdout(spec):
    """'key=glob[,key=glob...]' -> list of (key, glob). Keys: any MANIFEST session field (hostname, core_type,
    corunner, condition, plan, loop, stress, cpu, ...) or 'dataset'."""
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        k, _, g = part.partition("=")
        if not g:
            raise SystemExit(f"--holdout needs key=glob, got {part!r}")
        out.append((k.strip(), g.strip()))
    return out


def _matches(s, conds):
    from fnmatch import fnmatch
    return all(fnmatch(str(s.get(k, "")), g) for k, g in conds)


class Dataset:
    """One or more dataset folders (MANIFEST.json + sessions/), split by the manifests' session split.

    holdout='key=glob,...' re-splits for transfer tests: matching sessions become the test set (all of
    them); non-matching sessions keep their train/val split and their own test sessions are dropped, so
    the training data equals the in-domain run's. Example: holdout='core_type=E' (laptop P -> E core),
    holdout='dataset=delta*' (laptop -> EPYC 7763)."""

    def __init__(self, cfg, data_dir=None, cache_dir=None, splits=("train", "val", "test"), max_sessions=None,
                 max_frames=None, workers=1, dry_run=False, dry_frames=2000, holdout=None):
        self.cfg = cfg
        spec = data_dir or cfg.get("data_dirs") or cfg["data_dir"]
        dirs = spec.split(",") if isinstance(spec, str) else list(spec)
        self.dirs = [resolve_path(d.strip()) for d in dirs if d.strip()]
        self.manifests = [json.loads((d / "MANIFEST.json").read_text()) for d in self.dirs]
        for d, m in zip(self.dirs, self.manifests):
            check_supported_manifest(d, m)
        names = [m.get("dataset", d.name) for m, d in zip(self.manifests, self.dirs)]
        self.name = "+".join(names)
        self.manifest = self.manifests[0]
        self.budget_ms = float(cfg.get("budget_ms") or self.manifest.get("budget_ms") or 1000 / 60)
        self.holdout = holdout
        conds = parse_holdout(holdout)
        if dry_run:
            max_sessions, max_frames = 1, dry_frames
        self.max_frames = max_frames
        cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        tag = f"{spkm.FEATURE_SET}-{feature_code_hash()}" + (f"-n{max_frames}" if max_frames else "")
        self.cache = cache_root / self.name / tag
        by_split = {}
        self.carved = {}
        for d, m, name in zip(self.dirs, self.manifests, names):
            cdir = cache_root / name / tag
            cdir.mkdir(parents=True, exist_ok=True)
            sessions, cinfo = carve_validation(m["sessions"])
            if cinfo:
                self.carved[name] = cinfo
                log(f"{name}: MANIFEST has no val sessions; carved {cinfo['val']} of {cinfo['train_before']} train "
                    f"sessions into val (about {VAL_FRACTION:.0%}, stratified by stress, stable hash of the session "
                    f"id): by stress {cinfo['by_stress']}; train now {cinfo['train_after']}")
            for s in sessions:
                s = dict(s, dataset=name, _dir=str(d), _cache=str(cdir / f"{s['id']}.npz"))
                sp = s["split"]
                if conds:
                    sp = "test" if _matches(s, conds) else (None if sp == "test" else sp)
                if sp is not None:
                    by_split.setdefault(sp, []).append(s)
        self.splits = {}
        jobs, metas = [], {}
        for sp in splits:
            sel = by_split.get(sp, [])
            if dry_run:  # the shortest sessions keep the dry run light
                sel = sorted(sel, key=lambda s: s["frames"])
            if max_sessions:
                sel = sel[:max_sessions]
            metas[sp] = sel
            for s in sel:
                jobs.append((str(Path(s["_dir"]) / s["file"]), s["_cache"], cfg, max_frames))
        todo = [j for j in jobs if not os.path.exists(j[1])]
        if todo:
            log(f"computing features for {len(todo)} sessions -> {self.cache.parent.parent} (workers={workers})")
            if workers > 1 and len(todo) > 1:
                with get_context("fork").Pool(min(workers, len(todo))) as pool:
                    for _ in pool.imap_unordered(_cache_job, todo):
                        pass
            else:
                for j in todo:
                    _cache_job(j)
        for sp, sel in metas.items():
            sess = []
            for s in sel:
                z = np.load(s["_cache"])
                header = json.loads(z["header"].tobytes().decode())
                arrs = {k: z[k] for k in z.files if k != "header"}
                sess.append(Session(s, arrs, header, str(Path(s["_dir"]) / s["file"]), max_frames))
            self.splits[sp] = SplitData(sp, sess, len(cfg["levers"]))
        self.levers = cfg["levers"]
        self.ladder = [tuple(q) for q in cfg["ladder"]]
        for q in self.ladder:
            assert len(q) == len(self.levers), "ladder entries must have one level per lever"

    def split(self, name):
        return self.splits[name]

    def info(self):
        """Dataset block for manifests and results: name, frames, bytes, split."""
        sp = {k: {"sessions": len(v.sessions), "frames": v.N, "bytes": v.bytes()} for k, v in self.splits.items()}
        n = sum(v["frames"] for v in sp.values())
        b = sum(v["bytes"] for v in sp.values())
        frac = {k: round(v["frames"] / n, 4) if n else 0 for k, v in sp.items()}
        return {"name": self.name, "path": ",".join(str(d) for d in self.dirs), "num_frames": int(n),
                "size_bytes": int(b), "split": dict(frac, by="session", detail=sp, holdout=self.holdout),
                "feature_set": spkm.FEATURE_SET, "max_frames_per_session": self.max_frames,
                "val_carved_from_train": self.carved or None}

    # ------------------------------------------------------------ counterfactual rows

    def counterfactual(self, session, quality):
        """Features of every frame of `session` recomputed at quality levels `quality` (one per lever).
        Uses features.compute_features(..., quality=...), the engine-exact what-if path."""
        quality = tuple(int(v) for v in quality)
        if quality in session._cf:
            return session._cf[quality]
        cpath = Path(session.meta["_cache"][:-4] + f"_cf_{'-'.join(map(str, quality))}.npy")
        if cpath.exists():
            X = np.load(cpath)
        else:
            header, rec = framelog.read_log(session.path)
            if session.max_frames:
                rec = rec[:session.max_frames]
            X = features.compute_features(rec, header, quality=quality).astype(np.float32)
            tmp = str(cpath) + f".tmp{os.getpid()}.npy"
            np.save(tmp, X)
            os.replace(tmp, cpath)
        session._cf[quality] = X
        return X

    def counterfactual_split(self, split, quality):
        sd = self.split(split)
        return np.concatenate([self.counterfactual(s, quality) for s in sd.sessions])


# ---------------------------------------------------------------- normalisation statistics

def norm_stats(A, eps=1e-6):
    """mean/std in float32 over rows; constant columns get std 1 (the engine divides by std)."""
    A64 = A.astype(np.float64)
    mean = A64.mean(0)
    std = A64.std(0)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalise(A, mean, std):
    """Exactly the engine's float32 xn = (x - mean) * (1/std)."""
    inv = np.float32(1) / std.astype(np.float32)
    return ((A.astype(np.float32) - mean.astype(np.float32)) * inv).astype(np.float32)


# ---------------------------------------------------------------- simple per-session sequence helpers

def per_session_apply(y, offsets, fn):
    """Apply fn(segment) -> array of same length to each session segment."""
    out = np.empty(len(y), np.float64)
    for a, b in zip(offsets[:-1], offsets[1:]):
        out[a:b] = fn(np.asarray(y[a:b], np.float64))
    return out


def lag(y, offsets, k=1, fill=0.0):
    """y[t-k] within each session (fill at session start): what the engine has before frame t."""
    def f(s):
        o = np.full(len(s), fill)
        if k < len(s):
            o[k:] = s[:-k]
        return o
    return per_session_apply(y, offsets, f)


def ema_prev(y, offsets, alpha):
    """EMA of t_total over frames < t, initialised to the first value (0 before any frame)."""
    def f(s):
        o = np.zeros(len(s))
        e = None
        for i in range(1, len(s)):
            e = s[i - 1] if e is None else (1 - alpha) * e + alpha * s[i - 1]
            o[i] = e
        return o
    return per_session_apply(y, offsets, f)


def transition_mask(flags, offsets, radius=5):
    """Frames within +-radius frames of a quality change (flags bit1), never crossing sessions."""
    m = np.zeros(len(flags), bool)
    for a, b in zip(offsets[:-1], offsets[1:]):
        for i in np.nonzero((flags[a:b] & 2) != 0)[0]:
            m[a + max(0, i - radius):a + min(b - a, i + radius + 1)] = True
    return m


def spike_mask(y, offsets, factor=1.5, window=30):
    """Frames with t > factor x the median of the previous `window` frames (causal; per session)."""
    from numpy.lib.stride_tricks import sliding_window_view

    def f(s):
        if len(s) < 2:
            return np.zeros(len(s))
        pad = np.concatenate([np.full(window, np.nan), s[:-1]])
        win = sliding_window_view(pad, window)[:len(s)]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(win, axis=1)
        return np.where(np.isfinite(med), s > factor * med, False)
    return per_session_apply(y, offsets, f).astype(bool)


# ---------------------------------------------------------------- CLI: build the caches once (prep job)

_PREP_DS = None


def _cf_job(i, q):
    s = _PREP_DS.split("test").sessions[i]
    _PREP_DS.counterfactual(s, q)
    s._cf.clear()  # keep worker memory flat; the rows are on disk now

def main(argv=None):
    """python data.py [--cf-ladder]: compute + cache features for every session (and counterfactual rows of the
    test sessions at every ladder level), print a summary. No model is involved."""
    import argparse
    import common
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--cf-ladder", action="store_true", help="also cache counterfactual test rows per ladder level")
    ap.add_argument("--cf-levers", action="store_true", help="also cache test rows for each single-lever level")
    common.add_common_args(ap)
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    t = common.Timer()
    ds = Dataset(cfg, args.data, args.cache_dir, holdout=args.holdout, max_sessions=args.max_sessions, workers=args.workers,
                 dry_run=args.dry_run, dry_frames=args.dry_frames)
    for k, v in ds.splits.items():
        miss = float(np.mean(v.y > ds.budget_ms)) if v.N else 0.0
        log(f"{k:5s}: {len(v.sessions):3d} sessions {v.N:9,d} frames {v.bytes() / 1e6:7.1f} MB  miss rate {miss:.3f}  "
            f"X {v.X.nbytes / 1e6:.0f} MB")
    qs = []
    if args.cf_ladder:
        qs += list(ds.ladder)
    if args.cf_levers:
        for i, lv in enumerate(cfg["levers"]):
            qs += [tuple(v if j == i else 0 for j in range(len(cfg["levers"]))) for v in range(lv["levels"])]
    qs = list(dict.fromkeys(qs))
    if qs and "test" in ds.splits:
        global _PREP_DS
        _PREP_DS = ds
        jobs = [(i, q) for i in range(len(ds.split("test").sessions)) for q in qs]
        log(f"counterfactual rows: {len(qs)} quality settings x {len(ds.split('test').sessions)} test sessions")
        if args.workers > 1:  # forked workers share ds; only (index, quality) is sent; results land in the cache
            with get_context("fork").Pool(args.workers) as pool:
                pool.starmap(_cf_job, jobs)
        else:
            for j in jobs:
                _cf_job(*j)
    log(f"cache ready at {ds.cache} in {t.s():.1f}s")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
