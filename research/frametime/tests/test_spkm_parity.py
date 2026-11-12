#!/usr/bin/env python3
"""Parity test: Python reference (train/spkm.py) vs the engine's C++ predictor.

For every family, builds a model with RANDOM weights (nothing is trained),
exports it to .spkm, replays the same random frame sequence through both
implementations using the engine's per-frame protocol, and compares outputs.
Also checks the read-back round trip and that the C++ loader rejects bad files.

Run via `make parity` (builds build/spkm_eval first), or:
    python3 research/frametime/tests/test_spkm_parity.py --eval build/spkm_eval
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
import spkm  # noqa: E402

FAMILIES = ["linear", "mlp", "mlp_mulexp", "gbt", "rls", "gru", "gru_stream", "tcn"]


def make_cases(rng, n, model):
    """Random frame sequence: features drawn around the model's normalisation, y ~ a few ms."""
    mean, std = model.norm if model.norm is not None else (np.zeros(spkm.F), np.ones(spkm.F))
    x = (mean + std * rng.normal(0, 1, (n, spkm.F))).astype(np.float32)
    h = rng.normal(0, 2, (n, spkm.D)).astype(np.float32)
    y = (4 + x[:, :5].sum(1) * 0.3 + rng.normal(0, 0.5, n)).astype(np.float32)
    return x, h, y


def run_python(model, x, h, y):
    r = spkm.Runner(model)
    hist, outs = [], []
    for i in range(len(x)):
        r.begin_frame(hist[-spkm.MAX_HISTORY:])
        outs.append(r.predict(x[i]))
        hist.append(h[i])
        r.end_frame(x[i], h[i], y[i])
    return np.array(outs)


def run_cpp(evalbin, path, x, h, y, tmp):
    cases = os.path.join(tmp, "cases.bin")
    out = os.path.join(tmp, "out.bin")
    with open(cases, "wb") as f:
        f.write(struct.pack("<3I", len(x), spkm.F, spkm.D))
        f.write(np.concatenate([x, h, y[:, None]], axis=1).astype("<f4").tobytes())
    subprocess.run([evalbin, "run", path, cases, out], check=True)
    return np.fromfile(out, "<f4")


def check_cpp(evalbin, path):
    p = subprocess.run([evalbin, "check", path], capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=os.path.join(HERE, "..", "..", "..", "build", "spkm_eval"))
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for fam in FAMILIES:
            for size in ("small", "bench"):
                m = spkm.random_model(fam, rng, size)
                path = os.path.join(tmp, f"{fam}_{size}.spkm")
                nbytes = spkm.write(m, path)
                back = spkm.read(path)  # round trip through the reader
                assert spkm.to_bytes(back) == spkm.to_bytes(m), f"{fam}: read/write round trip changed the file"
                x, h, y = make_cases(rng, args.frames, m)
                ref = run_python(m, x, h, y)
                got = run_cpp(args.eval, path, x, h, y, tmp).reshape(ref.shape)
                err = np.abs(got - ref)
                tol = 1e-4 + 1e-4 * np.abs(ref)
                ok = bool(np.all(err <= tol)) and np.all(np.isfinite(got))
                failures += not ok
                print(f"{'ok  ' if ok else 'FAIL'} {fam:10s} {size:5s} params={m.num_parameters():6d} "
                      f"bytes={nbytes:7d} max_abs_err={err.max():.2e} max_rel_err={(err / (np.abs(ref) + 1e-9)).max():.2e} "
                      f"out_range=[{ref.min():.3g}, {ref.max():.3g}]")

        # Loader must reject: wrong version, wrong feature hash, truncation, renamed feature, bad CRC.
        m = spkm.random_model("mlp", rng)
        good = spkm.to_bytes(m)
        bad = {}
        bad["wrong_version"] = good[:4] + struct.pack("<I", spkm.SPKM_VERSION + 1) + good[8:]
        bad["v1_file"] = good[:4] + struct.pack("<I", 1) + good[8:]
        h0 = spkm.feature_hash()
        tampered = good[16:].replace(f"feature_hash={h0}".encode(), b"feature_hash=0123456789abcdef")
        bad["wrong_hash"] = good[:12] + struct.pack("<I", zlib.crc32(tampered) & 0xFFFFFFFF) + tampered
        renamed = good[16:].replace(b"\ncpu_ghz\n", b"\ncpu_ghx\n")
        bad["renamed_feature"] = good[:12] + struct.pack("<I", zlib.crc32(renamed) & 0xFFFFFFFF) + renamed
        bad["truncated"] = good[:len(good) // 2]
        bad["bit_flip"] = good[:-5] + bytes([good[-5] ^ 1]) + good[-4:]
        for name, data in [("good", good)] + list(bad.items()):
            p = os.path.join(tmp, name + ".spkm")
            with open(p, "wb") as f:
                f.write(data)
            rc, msg = check_cpp(args.eval, p)
            ok = (rc == 0) == (name == "good")
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} loader {name:16s} -> {msg}")
    print(f"\n{'PASS' if not failures else 'FAIL'}: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
