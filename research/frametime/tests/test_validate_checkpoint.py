#!/usr/bin/env python3
"""validate_checkpoint.py accepts a run written by spkm.write_checkpoint and catches every
manifest/.spkm mismatch listed in MODEL_CONTRACT.md section 6. Random weights only."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
import spkm  # noqa: E402

VALIDATOR = os.path.join(HERE, "..", "tools", "validate_checkpoint.py")


def validate(run):
    p = subprocess.run([sys.executable, VALIDATOR, run], capture_output=True, text=True)
    return p.returncode, p.stdout


def main():
    rng = np.random.default_rng(3)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for fam in ("mlp", "gbt", "rls", "gru", "tcn", "linear"):
            run = os.path.join(tmp, f"20261108-000000_{fam}_parity-test")
            m = spkm.random_model(fam, rng)
            spkm.write_checkpoint(run, m, dataset={"name": "none (random weights)", "num_frames": 1, "size_bytes": 1,
                                                   "split": {"by": "session"}},
                                  metrics={}, architecture=f"random {fam} for validator tests")
            open(os.path.join(run, "DONE"), "w").close()
            rc, out = validate(run)
            failures += rc != 0
            print(f"{'ok  ' if rc == 0 else 'FAIL'} valid {fam} run accepted" + ("" if rc == 0 else "\n" + out))

        base = os.path.join(tmp, "20261108-000000_mlp_parity-test")
        mutations = {
            "swapped features": lambda m: m["inputs"]["features"].insert(0, m["inputs"]["features"].pop(1)),
            "wrong feature hash": lambda m: m["inputs"].update(feature_hash="0123456789abcdef"),
            "missing feature hash": lambda m: m["inputs"].pop("feature_hash"),
            "wrong history_len": lambda m: m["inputs"].update(history_len=8),
            "outputs reordered": lambda m: m["outputs"].reverse(),
            "unknown output name": lambda m: m["outputs"][0].update(name="frame_ms_p95"),
            "wrong num_parameters": lambda m: m["model"].update(num_parameters=m["model"]["num_parameters"] + 1),
            "wrong family": lambda m: m["model"].update(family="gbt"),
            "wrong format version": lambda m: m["model"].update(engine_format_version=3),
            "wrong budget": lambda m: m["engine_hook"].update(budget_ms=33.3),
        }
        for name, mutate in mutations.items():
            run = os.path.join(tmp, "20261108-000001_mlp_mutant")
            shutil.rmtree(run, ignore_errors=True)
            shutil.copytree(base, run)
            mpath = os.path.join(run, "manifest.json")
            m = json.load(open(mpath))
            m["run_id"] = os.path.basename(run)
            mutate(m)
            json.dump(m, open(mpath, "w"))
            rc, out = validate(run)
            failures += rc == 0
            reason = next((l.strip() for l in out.splitlines() if l.strip().startswith("-")), "")
            print(f"{'ok  ' if rc != 0 else 'FAIL'} rejects {name:22s} {reason}")
        # A truncated weight file.
        run = os.path.join(tmp, "20261108-000002_mlp_truncated")
        shutil.copytree(base, run)
        m = json.load(open(os.path.join(run, "manifest.json")))
        m["run_id"] = os.path.basename(run)
        json.dump(m, open(os.path.join(run, "manifest.json"), "w"))
        data = open(os.path.join(run, "model.spkm"), "rb").read()
        open(os.path.join(run, "model.spkm"), "wb").write(data[:len(data) // 2])
        rc, out = validate(run)
        failures += rc == 0
        print(f"{'ok  ' if rc != 0 else 'FAIL'} rejects truncated model.spkm")
    print(f"\n{'PASS' if not failures else 'FAIL'}: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
