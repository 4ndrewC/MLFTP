#!/usr/bin/env python3
"""Summarise a collected dataset: frames, bytes, splits, and the frame-time distribution
per stress level / loop mode / core type (does the data have real budget pressure?).

    python3 research/frametime/tools/summarize_data.py [research/frametime/data] [--markdown]

Reads MANIFEST.json and every session log (numpy only; nothing is fitted).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from framelog import read_log  # noqa: E402

BUDGET = 1000 / 60


def dist(t):
    if len(t) == 0:
        return None
    p = np.percentile(t, [1, 50, 90, 95, 99, 99.9])
    return dict(frames=len(t), p1=p[0], p50=p[1], p90=p[2], p95=p[3], p99=p[4], p999=p[5], max=t.max(),
                over=(t > BUDGET).mean(), over_half=(t > BUDGET / 2).mean())


def table(rows, md):
    head = ["group", "sessions", "frames", "p1", "p50", "p90", "p95", "p99", "p99.9", "max", ">16.7ms", ">8.3ms"]
    lines = []
    if md:
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
    else:
        lines.append(f"{'group':22s} {'sess':>5s} {'frames':>9s} " + " ".join(f"{h:>7s}" for h in head[3:]))
    for name, ns, d in rows:
        vals = [f"{d[k]:.2f}" for k in ("p1", "p50", "p90", "p95", "p99", "p999", "max")] + \
               [f"{100 * d['over']:.2f}%", f"{100 * d['over_half']:.1f}%"]
        if md:
            lines.append(f"| {name} | {ns} | {d['frames']:,} | " + " | ".join(vals) + " |")
        else:
            lines.append(f"{name:22s} {ns:5d} {d['frames']:9,d} " + " ".join(f"{v:>7s}" for v in vals))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    with open(os.path.join(args.data, "MANIFEST.json")) as f:
        man = json.load(f)
    groups = {}
    extra = {"steps": np.zeros(16, np.int64), "ctl_changes": 0, "contend_frames": 0, "frames": 0}
    sub = {}
    for s in man["sessions"]:
        _, r = read_log(os.path.join(args.data, s["file"]))
        t = r["t_total_us"].astype(np.float64) / 1000
        keys = [("stress", s["stress"]), ("loop", s["loop"]), ("core", s.get("core_type")), ("split", s["split"]),
                ("policy", s["policy"]), ("plan", s.get("plan", "bulk")), ("condition", s.get("condition", "parallel")),
                ("corunner", s.get("corunner") or "none"), ("all", "all")]
        for k in keys:
            g = groups.setdefault(k, {"t": [], "n": 0})
            g["t"].append(t)
            g["n"] += 1
        sub.setdefault((s["stress"], s["loop"]), []).append(t)
        extra["steps"] += np.bincount(np.minimum(r["n_steps"], 15), minlength=16)
        extra["ctl_changes"] += int((r["flags"] & 2 != 0).sum())
        extra["contend_frames"] += int(r["contend"].sum())
        extra["frames"] += len(r)
    md = args.markdown
    tot = man["totals"]
    print(f"dataset {man['dataset']}: {tot['sessions']} sessions, {tot['frames']:,} frames, "
          f"{tot['bytes'] / 2**20:.1f} MB ({man['record_size']} B/frame)")
    print("split:", {k: (v["sessions"], v["frames"]) for k, v in tot["by_split"].items()})
    for kind in ("all", "stress", "loop", "core", "policy", "plan", "condition", "corunner", "split"):
        rows = [(f"{kind}={name}" if kind != "all" else "all", g["n"], dist(np.concatenate(g["t"])))
                for (k, name), g in sorted(groups.items()) if k == kind]
        print(f"\nframe time (ms) by {kind}:")
        print(table(rows, md))
    rows = [(f"{st}/{lp}", len(ts), dist(np.concatenate(ts))) for (st, lp), ts in sorted(sub.items())]
    print("\nframe time (ms) by stress/loop:")
    print(table(rows, md))
    st = extra["steps"]
    print(f"\nsteps per frame: " + ", ".join(f"{i}: {100 * st[i] / st.sum():.2f}%" for i in range(5) if st[i]))
    print(f"quality changes: {extra['ctl_changes']:,} ({1000 * extra['ctl_changes'] / extra['frames']:.2f} per 1000 frames); "
          f"frames with a contention thread busy at start: {100 * extra['contend_frames'] / extra['frames']:.1f}%")


if __name__ == "__main__":
    main()
