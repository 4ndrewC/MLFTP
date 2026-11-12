#!/usr/bin/env python3
"""Feature parity: the engine's feature vectors (engine/features.cpp, dumped with
--log-features) must equal what train/features.py computes from the frame log.

Runs short headless sessions (a few seconds each), including one where the
random controller changes quality mid-session and one at a raised out-scale.
Also checks the log reader on those files (record count, field table, sanity).
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from features import compute_features, history_vectors  # noqa: E402
from framelog import read_features, read_log  # noqa: E402
import spkm  # noqa: E402

RUNS = [
    ["--loop", "emulate", "--seed", "3", "--frames", "900"],
    ["--loop", "emulate", "--seed", "4", "--frames", "1800", "--ctl", "random", "--out-scale", "1.5",
     "--burst-mult", "2", "--spawn-mult", "2"],
    ["--loop", "fast", "--seed", "5", "--frames", "600", "--quality", "L5"],
    ["--loop", "emulate", "--seed", "6", "--frames", "600", "--quality", "1,2,1,1", "--out-scale", "2.2"],
    ["--loop", "emulate", "--seed", "7", "--frames", "700", "--fp", "none"],   # no fingerprint: fp terms 0
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=os.path.join(HERE, "..", "..", "..", "build", "spark_warden"))
    args = ap.parse_args()
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        # One real fingerprint (~3 s), used by every run (cache lookups would make the test machine-dependent).
        fpf = os.path.join(tmp, "fp.json")
        subprocess.run([args.game, "--fingerprint", fpf], check=True, stdout=subprocess.DEVNULL)
        for i, extra in enumerate(RUNS):
            log, feat = os.path.join(tmp, f"r{i}.sfl"), os.path.join(tmp, f"r{i}.sff")
            fp_args = [] if "--fp" in extra else ["--fp", fpf]
            subprocess.run([args.game, "--headless", "--bot", "--log", log, "--log-features", feat] + fp_args + extra,
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            header, rec = read_log(log)
            fh, Xe = read_features(feat)
            ok = True
            if fh["features"] != spkm.FEATURE_NAMES or fh["feature_hash"] != spkm.feature_hash():
                print("FAIL feature names/hash in the engine dump differ from spkm.py")
                ok = False
            Xp = compute_features(rec, header)
            assert Xp.shape == Xe.shape, (Xp.shape, Xe.shape)
            err = np.abs(Xp.astype(np.float64) - Xe)
            tol = 1e-5 + 1e-5 * np.abs(Xe)
            bad = np.argwhere(err > tol)
            if len(bad):
                ok = False
                for k, j in bad[:5]:
                    print(f"   frame {k} {spkm.FEATURE_NAMES[j]}: engine {Xe[k, j]!r} python {Xp[k, j]!r}")
            Hv = history_vectors(rec, Xp)
            sane = (len(rec) > 100 and np.all(rec["frame"] == np.arange(len(rec))) and np.all(rec["t_total_us"] > 0)
                    and Hv.shape[1] == spkm.D and np.all(np.isfinite(Xp)))
            ok = ok and sane
            changes = int((rec["flags"] & 2 != 0).sum())
            has_fp = header["machine"]["fingerprint"] is not None
            ok = ok and has_fp == ("--fp" not in extra) and (Xp[:, 51].max() > 0) == has_fp
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} run {i}: {len(rec)} frames, max_abs_err {err.max():.2e}, "
                  f"quality changes {changes}, AA-off frames {int((rec['q_aa'] == 1).sum())}, fingerprint {has_fp}, "
                  f"t_ref p50 {np.median(Xp[:, 51]):.2f} ms vs t_total p50 {np.median(rec['t_total_us']) / 1e3:.2f} ms")
    print(f"\n{'PASS' if not failures else 'FAIL'}: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
