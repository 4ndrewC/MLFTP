#!/usr/bin/env python3
"""Scan checkpoints/ and results/ and write results/RESULTS.md (one combined table + sections).

    aggregate.py [--out-root ROOT] [--all] [--include-dry]

--all keeps every run of an experiment (default: the latest per experiment).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import common
from common import rel

FORM_INPUTS = ["f_mlp_hist", "f_mlp_plan", "f_mlp_planprev", "f_mlp_all", "f_gbt_hist", "f_gbt_plan", "f_gbt_planprev",
               "f_gbt_all"]
FORM_TARGETS = ["f_mlp_all", "f_mlp_resid", "f_mlp_lognaive", "f_mlp_tref", "f_mlp_treftail", "f_mlp_cycles",
                "f_gbt_all", "f_gbt_lognaive", "f_gbt_tref"]
FORM_PMISS = ["f_mlp_all", "f_mlp_pmiss_derived"]


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            out.append(json.loads(Path(f).read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    return out


def f(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def hw(t):
    h = t.get("hardware", {})
    s = h.get("gpu") or h.get("cpu_model", "?")
    return f"{h.get('hostname', '?')} ({s})"


def row_metrics(r, sp="test"):
    return r.get("headline", {}).get(sp, {})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--include-dry", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.out_root) if args.out_root else common.FT_DIR
    res, ck = root / "results", root / "checkpoints"
    runs = [r for r in load(str(res / "runs" / "*.json")) if args.include_dry or not r.get("dry_run")]
    skipped = [r for r in runs if r.get("skipped")]
    runs = [r for r in runs if not r.get("skipped")]
    if not args.all:
        latest = {}
        for r in runs:
            if r["experiment"] not in latest or r["created_utc"] > latest[r["experiment"]]["created_utc"]:
                latest[r["experiment"]] = r
        runs = list(latest.values())
    by_exp = {r["experiment"]: r for r in runs}
    ctl = [c for c in load(str(res / "controller" / "*.json")) if args.include_dry or not c.get("dry_run")]
    imp = [c for c in load(str(res / "importance" / "*.json")) if args.include_dry or not c.get("dry_run")]

    L = ["# Frame-time prediction: results", "",
         f"Generated {common.utc_now()} by `train/aggregate.py` from `{rel(res)}` and `{rel(ck)}`. "
         f"{len(runs)} model/baseline runs, {len(ctl)} controller simulations, {len(imp)} importance studies.", "",
         "**How to read.** All metrics are on the held-out **test sessions** (split by session). "
         "`pinball p95` is the quantile loss of the 95th-percentile output in ms (lower is better); `cov p95` is "
         "the fraction of frames at or below the predicted p95 (target 0.95); `PR-AUC` scores `p_miss` against frames "
         "over the budget (head = dedicated classifier output, derived = computed from the quantiles). `params` is "
         "the exact count the engine reports. `est µs` is the estimated in-game cost per frame (2 controller evals / "
         "one eval per ladder level), scaled from the measured laptop numbers in MODEL_CONTRACT section 7; the real "
         "number is `t_pred_us` in a game log. Baselines are calibrated with train residual quantiles.", ""]
    if runs:
        d = runs[0]["dataset"]
        L += [f"**Dataset:** {d['name']} — {d['num_frames']:,} frames, {d['size_bytes'] / 1e6:.1f} MB; " +
              ", ".join(f"{k} {v['sessions']} sessions / {v['frames']:,} frames" for k, v in d["split"]["detail"].items())
              + f"; feature set {runs[0]['features']['feature_set']}.", ""]

    L += ["## All models and baselines (test)", "",
          "| experiment | family | params | est µs (2 / worst) | inputs | target | MAE ms | pinball p50 | pinball p95 | "
          "pinball p99 | cov p95 | cov p99 | crossing | PR-AUC head / derived | train s | trained on | exported |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"baselines": 0, "linear": 1, "trees": 2, "nn": 3, "formulation": 4, "baselines-transfer": 5,
             "formulation-transfer": 6}
    for r in sorted(runs, key=lambda r: (order.get(r["group"], 9), row_metrics(r).get("pinball_p95") or 1e9)):
        m, h, c = r["model"], row_metrics(r), r["model"].get("cost", {})
        L.append(f"| {r['experiment']} | {m['family']} | {f(m['num_parameters'])} | "
                 f"{f(c.get('ingame_us_est'), 1)} / {f(c.get('ingame_us_est_worst'), 1)} | {r['features']['mask']} "
                 f"({r['features']['n_used']}) | {r['target'].get('name')} | {f(h.get('mae_ms'))} | "
                 f"{f(h.get('pinball_p50'))} | {f(h.get('pinball_p95'))} | {f(h.get('pinball_p99'))} | "
                 f"{f(h.get('coverage_p95'), 3)} | {f(h.get('coverage_p99'), 3)} | {f(h.get('crossing_rate'), 3)} | "
                 f"{f(h.get('pmiss_pr_auc'), 3)} / {f(h.get('pmiss_derived_pr_auc'), 3)} | "
                 f"{f(r['training']['time_s'], 0)} | {hw(r['training'])} | "
                 f"{'yes' if r.get('checkpoint') and not r['checkpoint'].get('validator_errors') else 'no'} |")
    L += ["", "Architectures:", ""]
    for r in sorted(runs, key=lambda r: (order.get(r["group"], 9), r["experiment"])):
        ckp = r.get("checkpoint") or {}
        L.append(f"- `{r['experiment']}` ({r['run_id']}): {r['model']['architecture']}. {r['model'].get('details', '')}"
                 + (f" Checkpoint `{ckp.get('path')}`." if ckp.get("path") else "")
                 + (f" Not exported: {r['model']['export_note']}" if r["model"].get("export_note") else ""))

    def cmp_table(title, names, ref):
        rows = [by_exp[n] for n in names if n in by_exp]
        if not rows:
            return []
        base = row_metrics(by_exp[ref]) if ref in by_exp else {}
        out = ["", f"### {title}", "", "| experiment | inputs | target | p_miss | pinball p95 | Δ vs " + ref +
               " | MAE | cov p95 | PR-AUC head / derived | over-budget pinball p95 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            h = row_metrics(r)
            dlt = (h.get("pinball_p95") - base.get("pinball_p95")) if h.get("pinball_p95") is not None and \
                base.get("pinball_p95") is not None else None
            out.append(f"| {r['experiment']} | {r['features']['mask']} | {r['target'].get('name')} | "
                       f"{r['training']['hyperparams'].get('pmiss')} | {f(h.get('pinball_p95'))} | {f(dlt)} | "
                       f"{f(h.get('mae_ms'))} | {f(h.get('coverage_p95'), 3)} | {f(h.get('pmiss_pr_auc'), 3)} / "
                       f"{f(h.get('pmiss_derived_pr_auc'), 3)} | {f(h.get('over_budget_pinball_p95'))} |")
        return out

    L += ["", "## Formulation experiments"]
    L += cmp_table("Inputs: history-only vs planned-work-only vs both", FORM_INPUTS, "f_mlp_all")
    L += cmp_table("Targets: raw ms vs residual vs hardware-normalised", FORM_TARGETS, "f_mlp_all")
    L += cmp_table("p_miss: dedicated head vs derived from quantiles", FORM_PMISS, "f_mlp_all")
    if skipped:
        L += ["", "Skipped: " + "; ".join(f"`{s['experiment']}`: {s['skipped']}" for s in skipped)]

    transfer = [r for r in runs if "@" in r["experiment"]]
    if transfer:
        L += ["", "## Hardware transfer (held-out sessions as test; LIT_REVIEW_HW section 3.5)", "",
              "| experiment | held out | test frames | pinball p95 | cov p95 | cov p99 | MAE | in-domain pinball p95 | ratio |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(transfer, key=lambda r: r["experiment"]):
            base_name, _, hold = r["experiment"].partition("@")
            h, b = row_metrics(r), row_metrics(by_exp[base_name]) if base_name in by_exp else {}
            ratio = h.get("pinball_p95") / b["pinball_p95"] if h.get("pinball_p95") and b.get("pinball_p95") else None
            L.append(f"| {base_name} | {hold} | {f(r['dataset']['split']['detail'].get('test', {}).get('frames'))} | "
                     f"{f(h.get('pinball_p95'))} | {f(h.get('coverage_p95'), 3)} | {f(h.get('coverage_p99'), 3)} | "
                     f"{f(h.get('mae_ms'))} | {f(b.get('pinball_p95'))} | {f(ratio, 2)} |")
        L.append("\nThe in-domain column is the same experiment on the dataset's own split (different test "
                 "sessions), so the ratio is indicative only.")

    best = sorted([r for r in runs if row_metrics(r).get("pinball_p95") is not None],
                  key=lambda r: row_metrics(r)["pinball_p95"])[:6]
    if best:
        L += ["", "## By stress level (test pinball p95 / coverage p95), best six", "",
              "| experiment | " + " | ".join(["S0", "S1", "S2", "S3", "S4"]) + " | transitions | spikes |",
              "|---|---|---|---|---|---|---|---|"]
        for r in best:
            bs = r["metrics"].get("test", {}).get("by_stress", {})
            cells = [f"{f(bs[s].get('pinball_p95'), 3)} / {f(bs[s].get('coverage_p95'), 3)}" if s in bs else "-"
                     for s in ["S0", "S1", "S2", "S3", "S4"]]
            tr = r["metrics"]["test"].get("transitions", {})
            sp = r["metrics"]["test"].get("spikes", {})
            L.append(f"| {r['experiment']} | " + " | ".join(cells) +
                     f" | {f(tr.get('pinball_p95'), 3)} / {f(tr.get('coverage_p95'), 3)} | "
                     f"{f(sp.get('pinball_p95'), 3)} / {f(sp.get('coverage_p95'), 3)} |")

    if ctl:
        c = max(ctl, key=lambda c: c["created_utc"])
        L += ["", f"## Controller simulation ({c['run_id']})", "",
              "Closed loop on the test sessions with the analytic counterfactual simulator (see the run's .md for the "
              "full sweep and the simulator check on paired sessions). Default margin 0.10.", "",
              "| budget ms | model | misses/min | mean level | switches/min | p99 ms | evals/frame | est µs | "
              "misses/min at reactive's level | reactive misses/min | reactive level | reactive switches/min |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in c["summary"]:
            L.append(f"| {s['budget_ms']:.2f} | {s['model']} | {f(s['misses_per_min'], 2)} | {f(s['mean_level'], 2)} | "
                     f"{f(s['switches_per_min'], 2)} | {f(s['p99_ms'], 2)} | {f(s['mean_evals'], 2)} | "
                     f"{f(s.get('est_pred_us_mean'), 1)} | {f(s['misses_per_min_at_reactive_level'], 2)} | "
                     f"{f(s['reactive_misses_per_min'], 2)} | {f(s['reactive_mean_level'], 2)} | "
                     f"{f(s['reactive_switches_per_min'], 2)} |")
        L.append(f"\nFull table: `{rel(res / 'controller' / (c['run_id'] + '.md'))}`.")
    if imp:
        i = max(imp, key=lambda c: c["created_utc"])
        keep = [r["feature"] for r in i["features"] if r["recommendation"].startswith("keep")]
        red = [f"{r['feature']}→{r['cluster_rep']}" for r in i["features"] if r["recommendation"].startswith("redundant")]
        drop = [r["feature"] for r in i["features"] if r["recommendation"].startswith("drop")]
        L += ["", f"## Feature and lever importance ({i['run_id']})", "",
              f"PCA: {i['pca']['n_for_95pct']} components explain 95% of the feature variance.", "",
              f"- **Keep ({len(keep)}):** {', '.join(keep)}", f"- **Redundant ({len(red)}):** {', '.join(red)}",
              f"- **Drop candidates ({len(drop)}):** {', '.join(drop)}", "", "| lever | range ms | share | recommendation |",
              "|---|---|---|---|"]
        for lv in i["levers"]:
            L.append(f"| {lv['lever']} | {lv['pred_range_ms']:.3f} | {lv['pred_share_of_best_quality_cost']:.1%} | "
                     f"{lv['recommendation']} |")
        if i.get("ladder"):
            L.append("\nSuggested ladder: " + " → ".join(str(x["q"]) for x in i["ladder"]["suggested_ladder"]))
        L.append(f"\nFull tables: `{rel(res / 'importance' / (i['run_id'] + '.md'))}`.")

    cks = []
    for mf in sorted(glob.glob(str(ck / "*" / "manifest.json"))):
        try:
            m = json.loads(Path(mf).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cks.append((Path(mf).parent, m))
    if cks:
        L += ["", "## Checkpoints", "", "| run | family | params | architecture | frames | data MB | train s | trained on | DONE |",
              "|---|---|---|---|---|---|---|---|---|"]
        for p, m in cks:
            t = m.get("training", {})
            L.append(f"| {m['run_id']} | {m['model']['family']} | {f(m['model']['num_parameters'])} | "
                     f"{m['model']['architecture']} | {f(m['dataset'].get('num_frames'))} | "
                     f"{m['dataset'].get('size_bytes', 0) / 1e6:.1f} | {f(t.get('time_s'), 0)} | {hw(t) if t else '-'} | "
                     f"{'yes' if (p / 'DONE').exists() else 'no'} |")
    out = res / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".md.tmp")
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
