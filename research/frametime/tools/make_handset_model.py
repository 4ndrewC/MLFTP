#!/usr/bin/env python3
"""Write a HAND-SET (not trained) linear checkpoint for exercising the --model / --ctl model path.

    t_p50_ms = 1.00 * naive_ms + 0.05
    t_p95_ms = 1.15 * naive_ms + 0.25
    t_p99_ms = 1.30 * naive_ms + 0.40
    p_miss   = sigmoid(4 * (1.15 * naive_ms - budget) / budget * 4)

naive_ms (feature 37) is the engine's steps-aware analytic estimate (MODEL_CONTRACT.md). The
coefficients are picked by hand, not fitted to data. Usage:

    python3 research/frametime/tools/make_handset_model.py OUT_DIR
    build/spark_warden --model OUT_DIR --ctl model
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
import spkm  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    budget = 1000 / 60
    k = spkm.FEATURE_NAMES.index("naive_ms")
    W = np.zeros((4, spkm.F), np.float32)
    b = np.zeros(4, np.float32)
    # Outputs add raw naive_ms through the OUTS base column, so W holds (slope - 1).
    for i, (slope, icpt) in enumerate([(1.0, 0.05), (1.15, 0.25), (1.30, 0.40)]):
        W[i, k], b[i] = slope - 1.0, icpt
    W[3, k], b[3] = 16 * 1.15 / budget, -16.0  # logistic in naive_ms, 50% at naive*1.15 = budget
    m = spkm.linear(W, b, [("t_p50_ms", "identity", "naive_ms"), ("t_p95_ms", "identity", "naive_ms"),
                           ("t_p99_ms", "identity", "naive_ms"), ("p_miss", "sigmoid", "-")],
                    budget_ms=budget, info={"note": "hand-set coefficients, not trained"})
    spkm.write_checkpoint(out, m, dataset={"name": "none (hand-set weights)", "num_frames": 1, "size_bytes": 1,
                                           "split": {"by": "session"}},
                          metrics={"note": "hand-set baseline, not trained"},
                          architecture="linear on naive_ms (hand-set): p50/p95/p99 = a*naive+b, p_miss logistic")
    open(os.path.join(out, "DONE"), "w").close()
    print("wrote", out)


if __name__ == "__main__":
    main()
