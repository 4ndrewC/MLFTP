#!/usr/bin/env python3
"""Check that checkpoint folders follow the model contract (checkpoints/README.md).

Usage:
  validate_checkpoint.py RUN_DIR [RUN_DIR ...]      check runs, print a summary, exit 1 on any error
  validate_checkpoint.py --no-done RUN_DIR           same, but don't require the DONE marker
                                                     (use this on the cluster before writing DONE)

Besides the manifest schema, it opens the .spkm named by model.engine_file and checks that it
agrees with the manifest (format version, family, feature set/hash, feature names and order,
history length, outputs, budget, parameter count): see MODEL_CONTRACT.md section 6.

Standard library only, so it runs anywhere.
"""
import json
import math
import re
import struct
import sys
import zlib
from pathlib import Path

SCHEMA_VERSION = 1
FAMILIES = {"linear", "rls", "gbt", "mlp", "gru", "tcn", "other"}
SPKM_VERSION = 2
SPKM_FAMILIES = {"linear", "rls", "gbt", "mlp", "gru", "tcn"}
OUTPUT_NAME = re.compile(r"^(t_mean_ms|p_miss|t_p([1-9]|[1-9][0-9])_ms)$")


def read_spkm(path):
    """Parse the parts of a .spkm the manifest must agree with. Returns (info, errors)."""
    data = path.read_bytes()
    if len(data) < 16 or data[:4] != b"SPKM":
        return None, ["engine file is not a .spkm (bad magic)"]
    version, n_chunks, crc = struct.unpack_from("<3I", data, 4)
    out = {"version": version}
    if version != SPKM_VERSION:
        return out, [f".spkm format version {version} unsupported (expected {SPKM_VERSION})"]
    if zlib.crc32(data[16:]) & 0xFFFFFFFF != crc:
        return out, [".spkm CRC mismatch (truncated or corrupt)"]
    pos, chunks = 16, []
    for _ in range(n_chunks):
        if pos + 8 > len(data):
            return out, [".spkm chunk table truncated"]
        tag = data[pos:pos + 4].decode("ascii", "replace")
        (size,) = struct.unpack_from("<I", data, pos + 4)
        chunks.append((tag, data[pos + 8:pos + 8 + size]))
        pos += 8 + size + (-size) % 4
    if pos != len(data):
        return out, [".spkm has trailing bytes or a chunk past EOF"]
    get = lambda t: next((c for tg, c in chunks if tg == t), None)
    if get("INFO") is None or get("FEAT") is None or get("OUTS") is None:
        return out, [".spkm is missing INFO, FEAT or OUTS"]
    info = dict(l.split("=", 1) for l in get("INFO").decode().splitlines() if "=" in l)
    out["info"] = info
    out["features"] = [l for l in get("FEAT").decode().splitlines() if l]
    out["outputs"] = [l.split()[0] for l in get("OUTS").decode().splitlines() if l.strip()]
    params = 0
    for tag, c in chunks:  # parameter count, same definition as the engine (engine/predictor.cpp)
        if tag == "LINR":
            n_out, n_in = struct.unpack_from("<2I", c, 0)
            params += n_out * n_in + n_out
        elif tag == "DENS":
            n_in, n_out, _ = struct.unpack_from("<3I", c, 0)
            params += n_in * n_out + n_out
        elif tag == "GBTE":
            (n_out,) = struct.unpack_from("<I", c, 0)
            o = 4 + 4 * n_out
            (n_trees,) = struct.unpack_from("<I", c, o)
            o += 4
            params += n_out
            for _ in range(n_trees):
                _, nn = struct.unpack_from("<2I", c, o)
                params += 2 * nn
                o += 8 + 24 * nn
        elif tag == "RLS_":
            (ni,) = struct.unpack_from("<I", c, 0)
            o = 4 + 4 * ni + struct.calcsize("<i3d2I")
            has_p = struct.unpack_from("<I", c, o - 4)[0]
            n = ni + 1
            o += 8 * n + (8 * n * n if has_p else 0)
            (nq,) = struct.unpack_from("<I", c, o)
            params += n + n * n + 2 * nq
        elif tag == "GRU_":
            d, h, _ = struct.unpack_from("<3I", c, 0)
            params += 3 * h * d + 3 * h * h + 6 * h
        elif tag == "TCNL":
            cin, cout, k = struct.unpack_from("<3I", c, 0)
            params += cout * cin * k + cout
    out["num_parameters"] = params
    return out, []


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def check_run(run_dir: Path, require_done: bool = True):
    """Return (errors, manifest_or_None)."""
    errs = []
    if not run_dir.is_dir():
        return [f"{run_dir}: not a directory"], None
    if require_done and not (run_dir / "DONE").exists():
        errs.append("missing DONE marker (run incomplete?)")

    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        return errs + ["missing manifest.json"], None
    try:
        m = json.loads(mpath.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return errs + [f"manifest.json is not valid JSON: {e}"], None
    if not isinstance(m, dict):
        return errs + ["manifest.json must be a JSON object"], None

    def need(obj, key, check, what, prefix=""):
        if not isinstance(obj, dict) or key not in obj:
            errs.append(f"missing {prefix}{key}")
            return None
        v = obj[key]
        if not check(v):
            errs.append(f"{prefix}{key} must be {what} (got {v!r})")
            return None
        return v

    nonempty_str = lambda v: isinstance(v, str) and v.strip() != ""

    if need(m, "schema_version", _is_int, "an int") not in (None, SCHEMA_VERSION):
        errs.append(f"schema_version {m['schema_version']} unsupported (expected {SCHEMA_VERSION})")
    rid = need(m, "run_id", nonempty_str, "a non-empty string")
    if rid is not None and rid != run_dir.name:
        errs.append(f"run_id {rid!r} does not match folder name {run_dir.name!r}")
    need(m, "created_utc", nonempty_str, "a non-empty string")

    model = need(m, "model", lambda v: isinstance(v, dict), "an object")
    if model is not None:
        fam = need(model, "family", nonempty_str, "a string", "model.")
        if fam is not None and fam not in FAMILIES:
            errs.append(f"model.family {fam!r} not one of {sorted(FAMILIES)}")
        need(model, "architecture", nonempty_str, "a non-empty string", "model.")
        need(model, "num_parameters", lambda v: _is_int(v) and v >= 0, "an int >= 0", "model.")
        ef = need(model, "engine_file", nonempty_str, "a file name", "model.")
        if ef is not None:
            p = (run_dir / ef).resolve()
            if run_dir.resolve() not in p.parents:
                errs.append(f"model.engine_file {ef!r} must be inside the run folder")
            elif not p.is_file():
                errs.append(f"model.engine_file {ef!r} not found in run folder")
        need(model, "engine_format_version", lambda v: _is_int(v) and v >= 1, "an int >= 1", "model.")

    ds = need(m, "dataset", lambda v: isinstance(v, dict), "an object")
    if ds is not None:
        need(ds, "name", nonempty_str, "a non-empty string", "dataset.")
        need(ds, "num_frames", lambda v: _is_int(v) and v > 0, "an int > 0", "dataset.")
        need(ds, "size_bytes", lambda v: _is_int(v) and v > 0, "an int > 0", "dataset.")

    inp = need(m, "inputs", lambda v: isinstance(v, dict), "an object")
    if inp is not None:
        need(inp, "history_len", lambda v: _is_int(v) and v >= 0, "an int >= 0", "inputs.")
        need(inp, "feature_set", nonempty_str, "a non-empty string", "inputs.")
        need(inp, "feature_hash", lambda v: isinstance(v, str) and re.fullmatch(r"[0-9a-f]{16}", v) is not None,
             "16 lowercase hex digits", "inputs.")
        feats = need(inp, "features", lambda v: isinstance(v, list) and len(v) > 0,
                     "a non-empty list", "inputs.")
        if feats is not None:
            seen = set()
            for i, f in enumerate(feats):
                pre = f"inputs.features[{i}]."
                if not isinstance(f, dict):
                    errs.append(f"inputs.features[{i}] must be an object")
                    continue
                n = need(f, "name", nonempty_str, "a non-empty string", pre)
                if n is not None:
                    if n in seen:
                        errs.append(f"duplicate feature name {n!r}")
                    seen.add(n)
                need(f, "description", nonempty_str, "a non-empty string", pre)
                if ("mean" in f) != ("std" in f):
                    errs.append(f"{pre}mean and std must be given together")
                if "mean" in f:
                    need(f, "mean", _is_num, "a finite number", pre)
                    need(f, "std", lambda v: _is_num(v) and v > 0, "a number > 0", pre)

    outs = need(m, "outputs", lambda v: isinstance(v, list) and len(v) > 0, "a non-empty list")
    if outs is not None:
        for i, o in enumerate(outs):
            pre = f"outputs[{i}]."
            if not isinstance(o, dict):
                errs.append(f"outputs[{i}] must be an object")
                continue
            on = need(o, "name", nonempty_str, "a non-empty string", pre)
            if on is not None and not OUTPUT_NAME.match(on):
                errs.append(f"{pre}name {on!r} must be t_mean_ms, t_pNN_ms or p_miss (MODEL_CONTRACT.md)")
            need(o, "unit", nonempty_str, "a non-empty string", pre)
            need(o, "description", nonempty_str, "a non-empty string", pre)

    hook = need(m, "engine_hook", lambda v: isinstance(v, dict), "an object")
    if hook is not None:
        need(hook, "call_site", nonempty_str, "a non-empty string", "engine_hook.")
        need(hook, "predicts", nonempty_str, "a non-empty string", "engine_hook.")
        need(hook, "cadence_frames", lambda v: _is_int(v) and v >= 1, "an int >= 1", "engine_hook.")
        need(hook, "budget_ms", lambda v: _is_num(v) and v > 0, "a number > 0", "engine_hook.")
        need(hook, "max_overhead_us", lambda v: _is_num(v) and v > 0, "a number > 0", "engine_hook.")

    need(m, "metrics", lambda v: isinstance(v, dict), "an object")

    # Cross-check the manifest against the weight file the engine will load.
    if not errs and isinstance(model, dict):
        sp, serr = read_spkm((run_dir / model["engine_file"]).resolve())
        errs += serr
        if sp and not serr:
            info = sp["info"]
            def same(what, a, b):
                if a != b:
                    errs.append(f"{what}: manifest has {a!r}, model.spkm has {b!r}")
            same("model.engine_format_version", model["engine_format_version"], sp["version"])
            if model["family"] != "other":
                same("model.family", model["family"], info.get("family"))
            same("model.num_parameters", model["num_parameters"], sp["num_parameters"])
            same("inputs.feature_set", inp.get("feature_set"), info.get("feature_set"))
            same("inputs.feature_hash", inp.get("feature_hash"), info.get("feature_hash"))
            names = [f.get("name") for f in inp["features"]]
            n_feat = int(info.get("n_features", -1))
            if names != sp["features"][:n_feat]:
                diff = next((i for i, (a, b) in enumerate(zip(names, sp["features"])) if a != b), min(len(names), n_feat))
                errs.append(f"inputs.features differ from model.spkm FEAT (first difference at index {diff}; "
                            f"manifest has {len(names)} names, model {n_feat})")
            same("inputs.history_len", inp["history_len"], int(info.get("history_len", -1)))
            same("outputs (names, in order)", [o.get("name") for o in outs], sp["outputs"])
            try:
                b = float(info.get("budget_ms", "nan"))
                if not abs(hook["budget_ms"] - b) <= 1e-6 * max(1.0, b):
                    errs.append(f"engine_hook.budget_ms {hook['budget_ms']} != model.spkm budget_ms {b}")
            except ValueError:
                errs.append("model.spkm budget_ms is not a number")
            same("engine_hook.cadence_frames", hook["cadence_frames"], 1)
            same("engine_hook.predicts", hook["predicts"], "next_frame")
    return errs, m


def _fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def summary_row(run_dir, m):
    model, ds = m.get("model", {}), m.get("dataset", {})
    return (run_dir.name,
            str(model.get("family", "?")),
            f"{model.get('num_parameters', '?'):,}" if _is_int(model.get("num_parameters")) else "?",
            f"{ds.get('num_frames', '?'):,}" if _is_int(ds.get("num_frames")) else "?",
            _fmt_bytes(ds["size_bytes"]) if _is_int(ds.get("size_bytes")) else "?",
            str(model.get("architecture", "?")))


def main(argv):
    require_done = True
    if "--no-done" in argv:
        require_done = False
        argv = [a for a in argv if a != "--no-done"]
    if not argv or any(a in ("-h", "--help") for a in argv):
        print(__doc__.strip())
        return 0 if argv else 2

    bad, rows = 0, []
    for arg in argv:
        run_dir = Path(arg)
        errs, m = check_run(run_dir, require_done)
        if errs:
            bad += 1
            print(f"FAIL {run_dir.name}")
            for e in errs:
                print(f"     - {e}")
        if m is not None and not errs:
            rows.append(summary_row(run_dir, m))

    if rows:
        head = ("run", "family", "params", "frames", "data size", "architecture")
        widths = [max(len(r[i]) for r in rows + [head]) for i in range(len(head) - 1)]
        line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)) + "  " + r[-1]
        print(line(head))
        for r in rows:
            print(line(r))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
