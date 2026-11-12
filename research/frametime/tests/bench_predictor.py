#!/usr/bin/env python3
"""Inference cost per model family, with RANDOM weights at realistic sizes (nothing is trained).

1. Isolated: build/spkm_eval bench (predict / begin_frame / end_frame per call).
2. In game: a short headless run with --model and --ctl model; reports the logged
   t_pred_us (features + controller, several candidate evaluations + the logged prediction).

    python3 research/frametime/tests/bench_predictor.py [--eval build/spkm_eval] [--game build/spark_warden]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "train"))
import spkm  # noqa: E402
from framelog import read_log  # noqa: E402

FAMILIES = ["linear", "rls", "mlp", "mlp_mulexp", "gbt", "gru_stream", "gru", "tcn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=os.path.join(ROOT, "build", "spkm_eval"))
    ap.add_argument("--game", default=os.path.join(ROOT, "build", "spark_warden"))
    ap.add_argument("--cpu", default="2", help="pin to this CPU (taskset) for stable numbers")
    ap.add_argument("--frames", type=int, default=1800)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    pin = ["taskset", "-c", args.cpu] if args.cpu else []
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for fam in FAMILIES:
            m = spkm.random_model(fam, rng, "bench")
            paths[fam] = os.path.join(tmp, fam + ".spkm")
            spkm.write(m, paths[fam])
            print(f"{fam:10s} params {m.num_parameters():6d}  {m.info.get('note', '')}")
        print("\nIsolated (spkm_eval bench, 20000 calls each):")
        for fam in FAMILIES:
            out = subprocess.run(pin + [args.eval, "bench", paths[fam]], capture_output=True, text=True, check=True)
            print(f"  {fam:10s} " + out.stdout.strip())
        print(f"\nIn game (headless emulate, {args.frames} steps, --ctl model, out-scale 2, t_pred_us per frame):")
        for fam in ["none"] + FAMILIES:
            log = os.path.join(tmp, fam + ".sfl")
            cmd = pin + [args.game, "--headless", "--bot", "--loop", "emulate", "--frames", str(args.frames),
                         "--seed", "5", "--out-scale", "2", "--log", log, "--fp", "none"]
            if fam != "none":
                cmd += ["--model", paths[fam], "--ctl", "model"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _, r = read_log(log)
            tp, tl = r["t_pred_us"][10:], r["t_log_us"][10:]
            ev = r["ctl_evals"][10:]
            print(f"  {fam:10s} t_pred p50 {np.percentile(tp, 50):7.2f} p99 {np.percentile(tp, 99):7.2f} "
                  f"max {tp.max():7.1f} us | evals/frame {ev.mean():.2f} | logging p50 {np.percentile(tl, 50):5.2f} "
                  f"mean {tl.mean():5.2f} p99 {np.percentile(tl, 99):5.2f} us")


if __name__ == "__main__":
    main()
