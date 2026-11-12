"""Shared helpers for the phase-2 training code: paths, config, run ids, code version,
hardware description, seeding, JSON output and logging.

Nothing in this module imports torch or lightgbm; it only reports their versions
if another module already imported them.
"""
import datetime
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent
FT_DIR = TRAIN_DIR.parent                      # research/frametime
PROJECT_ROOT = FT_DIR.parent.parent
CHECKPOINTS_DIR = FT_DIR / "checkpoints"
RESULTS_DIR = FT_DIR / "results"
CONFIG_DIR = TRAIN_DIR / "configs"
DEFAULT_CACHE_DIR = Path(os.environ.get("SPARK_FT_CACHE", FT_DIR / "cache"))

if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import spkm  # noqa: E402  (features/spkm are owned by phase 1.5; we only read from them)


# ---------------------------------------------------------------- logging / timing

_T0 = time.time()


def log(*msg):
    print(f"[{time.time() - _T0:8.1f}s]", *msg, flush=True)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def s(self):
        return time.time() - self.t0


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(family, name):
    """checkpoints/README.md format: YYYYMMDD-HHMMSS_<family>_<short-name>."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-." else "-" for c in name)
    return f"{ts}_{family}_{safe}"


# ---------------------------------------------------------------- config

def load_config(path=None, feature_set=None):
    """Config for the active feature set. Resolution order: explicit path, $SPARK_FT_CONFIG,
    configs/<feature_set>.json where feature_set defaults to spkm.FEATURE_SET."""
    path = path or os.environ.get("SPARK_FT_CONFIG")
    fs = feature_set or spkm.FEATURE_SET
    if path is None:
        path = CONFIG_DIR / f"{fs}.json"
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no training config for feature set {fs!r} at {path}.\n"
                         f"Copy configs/spark-ft-v2.json, set feature_set, levers and ladder for the new set "
                         f"(see TRAINING.md 'New feature set').")
    cfg = json.loads(path.read_text())
    cfg["_path"] = str(path)
    if cfg.get("feature_set") == spkm.FEATURE_SET_V1 or cfg.get("unsupported"):
        raise SystemExit(f"config {path} is for {spkm.FEATURE_SET_V1!r} (phase-1 features, data/ logs): NOT SUPPORTED for "
                         f"training. The engine runs {spkm.FEATURE_SET!r} only; use configs/spark-ft-v2.json "
                         f"and data-v2/.")
    if cfg.get("feature_set") != spkm.FEATURE_SET:
        raise SystemExit(f"config {path} is for {cfg.get('feature_set')!r} but train/spkm.py implements "
                         f"{spkm.FEATURE_SET!r}; use the matching config or update spkm.py/features.py")
    cfg.setdefault("quantiles", [0.5, 0.95, 0.99])
    cfg.setdefault("history_overrides", {})
    return cfg


def resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------- versions / hardware

def code_version():
    """Hash of the training sources (train/*.py and configs), so results are traceable without git."""
    h = hashlib.sha256()
    files = sorted(list(TRAIN_DIR.glob("*.py")) + list(CONFIG_DIR.glob("*.json")))
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def hardware_info(device=None):
    info = {
        "hostname": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "cpus_available": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    if "lightgbm" in sys.modules:
        info["lightgbm"] = sys.modules["lightgbm"].__version__
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        info["torch"] = torch.__version__
        info["device"] = str(device) if device is not None else "cpu"
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    else:
        info["device"] = "cpu"
    return info


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- no-fit guard

NO_FIT_ENV = "FRAMETIME_NO_FIT"


def no_fit():
    """True in dry runs. Every function that fits anything (offline or online, numpy, LightGBM or torch)
    calls require_fitting() first, so a dry run cannot fit even if a branch is wrong."""
    return os.environ.get(NO_FIT_ENV) == "1"


def require_fitting(what):
    if no_fit():
        raise RuntimeError(f"{what}: fitting is disabled ({NO_FIT_ENV}=1, set by --dry-run). Dry runs export "
                           f"untrained weights and must not fit, update or train anything.")


def freeze_online(m):
    """Dry runs: turn off the engine's per-frame RLS update of a loaded model (forward pass only, weights
    stay at their initial values). Non-RLS models are returned unchanged."""
    if no_fit() and getattr(m, "family", None) == "rls":
        m.params["online"] = False
    return m


# ---------------------------------------------------------------- JSON

def clean(o):
    """Recursively make a result record JSON-safe: NaN/inf -> None, numpy scalars and arrays -> Python
    numbers and lists, Path -> str, bool kept as bool. Any other type is an error (raised before anything is
    written, so no half-written record)."""
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"clean(): not serialisable into a result record: {type(o).__name__}")


def write_json(path, obj):
    """Atomic write (tmp + rename) so readers never see half a file. The record is cleaned first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(clean(obj), indent=2)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(body)
    os.replace(tmp, path)


def rel(p):
    """Path relative to the project root when possible (for portable reports)."""
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- CLI

def add_common_args(ap):
    ap.add_argument("--config", default=None, help="training config JSON (default: configs/<feature_set>.json)")
    ap.add_argument("--data", default=None,
                    help="dataset folder(s) with MANIFEST.json, comma-separated to mix (default: config data_dir)")
    ap.add_argument("--holdout", default=None,
                    help="transfer test: sessions matching key=glob[,key=glob] form the test set, e.g. core_type=E, "
                         "dataset=laptop-delta*, hostname=gpua*; other sessions keep train/val")
    ap.add_argument("--cache-dir", default=None, help=f"feature cache (default {rel(DEFAULT_CACHE_DIR)} or $SPARK_FT_CACHE)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")),
                    help="processes for feature computation / threads for LightGBM")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny slice (1 session per split, <= --dry-frames frames), NO fitting: random/untrained "
                         "weights, outputs to --out-root (never checkpoints/)")
    ap.add_argument("--dry-frames", type=int, default=2000)
    ap.add_argument("--out-root", default=None,
                    help="root for checkpoints/ and results/ (default: research/frametime; required outside it "
                         "for --dry-run)")
    ap.add_argument("--max-sessions", type=int, default=None, help="debug: limit sessions per split")
    return ap


def output_dirs(args):
    """(checkpoints_dir, results_dir). Dry runs must not write into the real checkpoints/ folder."""
    if args.dry_run:
        os.environ[NO_FIT_ENV] = "1"   # from here on every fitting call site refuses (require_fitting)
    if args.out_root is None:
        if args.dry_run:
            root = Path(os.environ.get("TMPDIR", "/tmp")) / "spark-ft-dryrun"
        else:
            return CHECKPOINTS_DIR, RESULTS_DIR
    else:
        root = Path(args.out_root).resolve()
    if args.dry_run and (root == FT_DIR or CHECKPOINTS_DIR in [root, *root.parents]):
        raise SystemExit("--dry-run must write outside research/frametime/checkpoints (use a scratch --out-root)")
    return root / "checkpoints", root / "results"
