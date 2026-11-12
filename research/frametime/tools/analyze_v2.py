#!/usr/bin/env python3
"""Descriptive analysis of a v2 dataset (no model fitting: means, quantiles, paired differences,
correlations). Sections:

  levers    each lever's marginal cost per stress level, from the paired lever-sweep sessions
  matched   solo vs parallel: the same sessions run one at a time and six at once
  vhw       virtual hardware: co-runners and P vs E core vs the P-core baseline, and how well the
            fingerprint reference time t_ref tracks each change (LIT_REVIEW_HW option A)
  features  feature correlation with t_total (Pearson, Spearman), redundant pairs, constant features

    python3 research/frametime/tools/analyze_v2.py [research/frametime/data-v2] [--markdown] [--sections ...]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from features import compute_features  # noqa: E402
from framelog import read_log  # noqa: E402
import spkm  # noqa: E402

SKIP = 120  # warm-up frames dropped from every session


def load(data, e):
    h, r = read_log(os.path.join(data, e["file"]))
    return h, r


def t_ms(r):
    return r["t_total_us"][SKIP:].astype(np.float64) / 1000


def table(head, rows, md):
    if md:
        out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    else:
        w = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
        out = ["  ".join(str(c).rjust(n) for c, n in zip(head, w))]
        out += ["  ".join(str(c).rjust(n) for c, n in zip(row, w)) for row in rows]
    return "\n".join(out)


def levers(data, man, md):
    groups = {}
    for e in man["sessions"]:
        if e.get("plan") == "levers":
            groups.setdefault(e["stress"], []).append(e)
    names = ["scale", "particles", "glow", "aa"]
    lv_val = [("1", "0.8", "0.65", "0.5"), ("1", "0.6", "0.35", "0.15"), ("1", "0.75", "0.5", "off"), ("on", "off")]
    rows, base_of = [], {}
    for stress in sorted(groups):
        ss = groups[stress]
        base = next((e for e in ss if e["quality_start"] == [0, 0, 0, 0]), None)
        if not base:
            continue
        tb = t_ms(load(data, base)[1])
        base_of[stress] = tb.mean()
        rows.append([stress, "L0 (all full)", "-", f"{tb.mean():.2f}", f"{np.percentile(tb, 95):.2f}", "-", "-"])
        for e in ss:
            q = e["quality_start"]
            nz = [i for i, v in enumerate(q) if v]
            if len(nz) != 1:
                continue
            i = nz[0]
            t = t_ms(load(data, e)[1])
            n = min(len(t), len(tb))
            saved = tb[:n].mean() - t[:n].mean()
            rows.append([stress, names[i], lv_val[i][q[i]], f"{t.mean():.2f}", f"{np.percentile(t, 95):.2f}",
                         f"{saved:.2f}", f"{100 * saved / tb[:n].mean():.0f}%"])
    print("\n### Lever marginal cost (paired: same game, one lever lowered; emulate, P-core, solo)\n")
    print(table(["stress", "lever", "level", "mean ms", "p95 ms", "saved ms", "saved %"], rows, md))


def matched(data, man, md):
    groups = {}
    for e in man["sessions"]:
        if e.get("plan") == "matched":
            groups.setdefault(e["group"], {})[e["condition"]] = e
    rows, ratios = [], {"mean": [], "p50": [], "p95": [], "p99": []}
    for g, d in sorted(groups.items()):
        if "solo" not in d or "parallel" not in d:
            continue
        a, b = t_ms(load(data, d["solo"])[1]), t_ms(load(data, d["parallel"])[1])
        _, ra = load(data, d["solo"])
        _, rb = load(data, d["parallel"])
        st = {k: (f(a), f(b)) for k, f in (("mean", np.mean), ("p50", np.median),
                                             ("p95", lambda x: np.percentile(x, 95)),
                                             ("p99", lambda x: np.percentile(x, 99)))}
        for k in ratios:
            ratios[k].append(st[k][1] / st[k][0])
        e = d["solo"]
        rows.append([e["stress"], e["loop"], f"cpu{e['cpu']} ({e['core_type']})",
                     f"{st['mean'][0]:.2f} / {st['mean'][1]:.2f}", f"{st['p95'][0]:.2f} / {st['p95'][1]:.2f}",
                     f"{st['p99'][0]:.2f} / {st['p99'][1]:.2f}", f"{st['mean'][1] / st['mean'][0]:.3f}",
                     f"{int(np.median(ra['cpu_mhz']))} / {int(np.median(rb['cpu_mhz']))}"])
    print("\n### Solo vs parallel (same session spec; solo = one game at a time, parallel = all at once)\n")
    print(table(["stress", "loop", "cpu", "mean solo/par", "p95 solo/par", "p99 solo/par", "mean ratio",
                 "MHz p50 solo/par"], rows, md))
    if ratios["mean"]:
        print("\nratio parallel/solo, geometric mean over groups: " + ", ".join(
            f"{k} {np.exp(np.mean(np.log(v))):.3f} (range {min(v):.3f}-{max(v):.3f})" for k, v in ratios.items()))


def vhw(data, man, md):
    groups = {}
    for e in man["sessions"]:
        if e.get("plan") == "vhw":
            groups.setdefault(e["group"], []).append(e)
    rows = []
    for g, ss in sorted(groups.items()):
        base = next((e for e in ss if not e.get("corunner") and e["core_type"] == "P"), None)
        if not base:
            continue
        hb, rb = load(data, base)
        tb = t_ms(rb)
        refb = compute_features(rb, hb)[SKIP:, 51].astype(np.float64)
        for e in ss:
            h, r = load(data, e)
            t = t_ms(r)
            ref = compute_features(r, h)[SKIP:, 51].astype(np.float64)
            name = (e.get("corunner") or "none") + f" @{e['core_type']}-core"
            fp = h["machine"].get("fingerprint") or {}
            f = fp.get("fp", {})
            rows.append([e["stress"], e["loop"], name, f"{t.mean():.2f}", f"{t.mean() / tb.mean():.3f}",
                         f"{ref.mean() / refb.mean():.3f}" if refb.mean() > 0 else "-",
                         f"{np.median(t / np.maximum(ref, 1e-6)):.3f}" if ref.mean() > 0 else "-",
                         f"{f.get('lat_dram_ns', 0):.0f}", f"{f.get('bw_dram_gbs', 0):.1f}",
                         f"{f.get('frame_ms_s1', 0):.3f}"])
    print("\n### Virtual hardware (same games; ratios vs the P-core, no-co-runner session of the group)\n")
    print(table(["stress", "loop", "config", "mean ms", "actual ratio", "t_ref ratio", "median t/t_ref",
                 "fp DRAM ns", "fp DRAM GB/s", "fp frame ms"], rows, md))


def features_corr(data, man, md, max_rows=400000):
    sess = [e for e in man["sessions"] if e["split"] == "train" and e.get("plan") in ("bulk", "solo")]
    Xs, ys = [], []
    rng = np.random.default_rng(0)
    per = max(1, max_rows // max(1, len(sess)))
    for e in sess:
        h, r = load(data, e)
        X = compute_features(r, h)[SKIP:]
        y = r["t_total_us"][SKIP:].astype(np.float64) / 1000
        idx = rng.choice(len(y), min(per, len(y)), replace=False) if len(y) > per else np.arange(len(y))
        Xs.append(X[idx].astype(np.float64))
        ys.append(y[idx])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    names = spkm.FEATURE_NAMES

    def rank(v):
        o = np.argsort(v, kind="mergesort")
        rk = np.empty(len(v))
        rk[o] = np.arange(len(v))
        return rk

    sd = X.std(0)
    const = [names[i] for i in range(X.shape[1]) if sd[i] == 0]
    var = [i for i in range(X.shape[1]) if sd[i] > 0]
    pear = {i: np.corrcoef(X[:, i], y)[0, 1] for i in var}
    ry = rank(y)
    spear = {i: np.corrcoef(rank(X[:, i]), ry)[0, 1] for i in var}
    top = sorted(var, key=lambda i: -abs(spear[i]))
    print(f"\n### Feature correlation with t_total ({len(y):,} frames from {len(sess)} train sessions, bulk+solo)\n")
    print(table(["feature", "source", "Spearman", "Pearson"],
                [[names[i], spkm.FEATURE_INFO[names[i]][0], f"{spear[i]:+.3f}", f"{pear[i]:+.3f}"] for i in top[:20]],
                md))
    print(f"\nweakest (|Spearman| < 0.05): {', '.join(names[i] for i in top if abs(spear[i]) < 0.05) or 'none'}")
    print(f"constant within these sessions: {', '.join(const) or 'none'}")
    C = np.corrcoef(X[:, var].T)
    pairs = []
    for a in range(len(var)):
        for b in range(a + 1, len(var)):
            if abs(C[a, b]) >= 0.95:
                pairs.append((names[var[a]], names[var[b]], C[a, b]))
    pairs.sort(key=lambda p: -abs(p[2]))
    print(f"\nredundant pairs (|Pearson| >= 0.95 between features): {len(pairs)}")
    print(table(["feature A", "feature B", "r"], [[a, b, f"{r:+.3f}"] for a, b, r in pairs[:25]], md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=os.path.join(HERE, "..", "data-v2"))
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--sections", default="levers,matched,vhw,features")
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.data, "MANIFEST.json")))
    for s in args.sections.split(","):
        {"levers": levers, "matched": matched, "vhw": vhw, "features": features_corr}[s](args.data, man, args.markdown)


if __name__ == "__main__":
    main()
