#!/usr/bin/env python3
"""Train, export, validate and report frame-time predictors (one or more experiments).

    python train.py mlp gbt_q                       named experiments (experiments.py)
    python train.py --group linear                  every experiment of a group
    python train.py --group formulation --task 2    one array-job slice ($SLURM_ARRAY_TASK_ID)
    python train.py --group nn --dry-run --out-root /tmp/x
                                                    tiny slice, NO fitting (random/untrained weights),
                                                    full export + validation into a scratch folder

Outputs per run: checkpoints/<run_id>/{model.spkm, manifest.json, REPORT.md, native/, DONE}
(only for models the engine can run) and results/runs/<run_id>.json (+ results/preds/<run_id>.npz).
"""
import argparse
import os
import sys

import numpy as np

import common
from common import Timer, log, make_run_id
import data as Dt
import experiments as E
import metrics as M
import runlib
import spkm
import targets as T


# ---------------------------------------------------------------- shared preparation

class Prep:
    """Normalised features, training-space targets and miss labels for train/val/test."""

    def __init__(self, ds, cfg, exp):
        self.ds, self.cfg, self.exp = ds, cfg, exp
        self.budget = ds.budget_ms
        self.quantiles = list(cfg["quantiles"])
        self.mask = Dt.feature_mask(cfg, exp["features"])
        self.target = T.get_target(cfg, exp["target"])
        tr = ds.split("train")
        self.mean, self.std = Dt.norm_stats(tr.X)
        self.sd = {sp: ds.split(sp) for sp in ("train", "val", "test")}
        self.Xn = {sp: Dt.normalise(sd.X, self.mean, self.std) for sp, sd in self.sd.items()}
        self.ref = {sp: self.target.ref(sd.X, sd) for sp, sd in self.sd.items()}
        self.Z = {sp: self.target.encode_q(sd.y, sd.X, self.ref[sp], self.quantiles).astype(np.float32)
                  for sp, sd in self.sd.items()}   # (N, n_quantiles) training-space targets
        self.Mb = {sp: (sd.y > self.budget).astype(np.float32) for sp, sd in self.sd.items()}
        self.pmiss = exp["pmiss"] == "head"
        ex = self.target.export()
        self.export_note = self.target.export_reason()
        self.decode = ex is None
        self.outputs = runlib.outputs_for(self.target if ex is not None else None, self.quantiles, self.pmiss)

    def hist_norm(self):
        if not hasattr(self, "hmean"):
            self.hmean, self.hstd = Dt.norm_stats(self.sd["train"].H)
            self.Hn = {sp: Dt.normalise(sd.H, self.hmean, self.hstd) for sp, sd in self.sd.items()}
        return self.hmean, self.hstd

    def to_ms(self, sp, Q, logit=None):
        """Training-space quantiles (N, k) [+ logits] -> prediction dict in ms (engine reference)."""
        ref = self.target.engine_ref(self.sd[sp].X, self.sd[sp])
        D = self.target.decode_q(Q, self.sd[sp].X, ref, self.quantiles)
        out = {M.qname(q): D[:, i] for i, q in enumerate(self.quantiles)}
        if logit is not None:
            out["p_miss"] = 1 / (1 + np.exp(-np.asarray(logit, np.float64)))
        return out

    def info(self, run_id):
        return {"run_id": run_id, "experiment": self.exp["name"]}


# ---------------------------------------------------------------- torch-based kinds

def _torch_setup(args, seed):
    import torch
    import nn as N
    dev = N.get_device(args.device)
    if dev.type == "cpu":
        torch.set_num_threads(max(1, args.workers))
    common.set_seed(seed)
    return torch, N, dev


def _init_bias(lin, Z, qs, miss=None):
    """Start the output bias at the train targets' quantiles (and the miss log-odds), not 0: with few
    large-batch steps the bias otherwise cannot reach raw-ms targets (linear_qr / Lo underfit)."""
    import torch
    Z = np.asarray(Z, np.float64)
    b = [np.quantile(Z[:, i] if Z.ndim == 2 else Z, q) for i, q in enumerate(qs)]
    if miss is not None:
        r = float(np.clip(np.mean(miss), 1e-4, 1 - 1e-4))
        b.append(np.log(r / (1 - r)))
    with torch.no_grad():
        lin.bias.copy_(torch.as_tensor(b, dtype=lin.bias.dtype, device=lin.bias.device))


def train_linear_qr(args, ds, cfg, exp, prep, run_id):
    torch, N, dev = _torch_setup(args, args.seed)
    hp = exp["hp"]
    net = N.LinearHeads(len(prep.quantiles), prep.pmiss, prep.mask)
    _init_bias(net.lin, prep.Z["train"], prep.quantiles, prep.Mb["train"] if prep.pmiss else None)
    obj = N.Objective(prep.quantiles, hp.get("bce_weight", 1.0) if prep.pmiss else 0.0, hp.get("cross_weight", 0.0),
                      device=dev)
    t = Timer()
    info = {}
    if not args.dry_run:
        info = N.fit_tabular(net, obj, prep.Xn["train"], prep.Z["train"], prep.Mb["train"], prep.Xn["val"],
                             prep.Z["val"], prep.Mb["val"], hp, dev, args.seed)
    tt = t.s()
    W, b = net.export_linear()
    m = spkm.linear(W, b, prep.outputs, norm=(prep.mean, prep.std), budget_ms=prep.budget, info=prep.info(run_id))
    preds, check = _torch_predict_and_check(prep, m, net, dev)
    arch = f"linear quantile regression {spkm.F}->{len(prep.outputs)} (pinball, non-crossing penalty)"
    details = (f"independent rows for {', '.join(o[0] for o in m.outputs)}; outputs {prep.outputs[0][1]}"
               f"{' + ' + prep.outputs[0][2] if prep.outputs[0][2] != '-' else ''}; p_miss = sigmoid(logistic row)")
    return m, arch, details, tt, dict(info, export_check=check, params=N.count_params(net)), preds, \
        {"model.pt": lambda p: torch.save(net.state_dict(), p)}, dev


def _torch_predict_and_check(prep, m, net, dev, kind="tab"):
    """Framework predictions for val/test in ms + parity check against the exported .spkm."""
    import torch
    import nn as N
    net.eval()
    preds = {}
    with torch.no_grad():
        for sp in ("val", "test"):
            if kind == "tab":
                Qs, Ls = [], []
                X = torch.as_tensor(prep.Xn[sp], device=dev)
                for i in range(0, len(X), 65536):
                    Q, L = net(X[i:i + 65536])
                    Qs.append(Q.cpu().numpy())
                    if L is not None:
                        Ls.append(L.cpu().numpy())
                Q = np.concatenate(Qs)
                L = np.concatenate(Ls) if Ls else None
            elif kind == "gru":
                Q, L = N.gru_stream_outputs(net, _seqdict(prep, sp), dev)
            else:
                Hn = torch.as_tensor(prep.Hn[sp], device=dev)
                X = torch.as_tensor(prep.Xn[sp], device=dev)
                rs = torch.as_tensor(prep.sd[sp].row_start, device=dev)
                Qs, Ls = [], []
                for a in range(0, len(X), 16384):
                    rows = torch.arange(a, min(len(X), a + 16384), device=dev)
                    Qb, Lb = net(N.tcn_windows(Hn, rows, rs, net.L), X[rows])
                    Qs.append(Qb.cpu().numpy())
                    if Lb is not None:
                        Ls.append(Lb.cpu().numpy())
                Q = np.concatenate(Qs)
                L = np.concatenate(Ls) if Ls else None
            preds[sp] = prep.to_ms(sp, Q, L)
    exported = {sp: runlib.predict_model(m, prep.sd[sp], prep.target, prep.decode) for sp in ("val", "test")}
    check = {}
    for k in preds["val"]:
        check[k] = runlib.check_close(exported["val"][k], preds["val"][k], f"val {k}: torch vs .spkm")
    check["runner"] = runlib.runner_parity(m, prep.sd["val"])
    return exported, check


def _seqdict(prep, sp):
    prep.hist_norm()
    sd = prep.sd[sp]
    return {"Xn": prep.Xn[sp], "Hn": prep.Hn[sp], "Z": prep.Z[sp], "M": prep.Mb[sp], "offsets": sd.offsets,
            "row_start": sd.row_start}


def train_lo(args, ds, cfg, exp, prep, run_id):
    torch, N, dev = _torch_setup(args, args.seed)
    hp = exp["hp"]
    mask_t = torch.as_tensor(prep.mask, dtype=torch.float32, device=dev)

    class LoNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(spkm.F, 1)

        def forward(self, xn):
            return self.lin(xn * mask_t), None

    net = LoNet().to(dev)
    _init_bias(net.lin, prep.Z["train"], [0.5])

    def loss(Q, _logit, z, _m):
        return N.asym_sq_loss(Q[:, 0], z[:, 0] if z.dim() == 2 else z, hp["alpha"]) \
            + hp.get("l1", 0.0) * net.lin.weight.abs().sum()

    t = Timer()
    info = {}
    if not args.dry_run:
        info = N.fit_tabular(net, loss, prep.Xn["train"], prep.Z["train"], prep.Mb["train"], prep.Xn["val"],
                             prep.Z["val"], prep.Mb["val"], hp, dev, args.seed)
    tt = t.s()
    net.eval()
    with torch.no_grad():
        zv = net(torch.as_tensor(prep.Xn["val"], device=dev))[0][:, 0].cpu().numpy()
    yv = prep.target.decode(zv, prep.target.engine_ref(prep.sd["val"].X, prep.sd["val"]))
    cov = M.coverage(prep.sd["val"].y, yv)
    name = __import__("linear").lo_output_name(cov)
    tr, base = prep.outputs[0][1], prep.outputs[0][2]
    W = (net.lin.weight * mask_t[None, :]).detach().cpu().double().numpy()
    bb = net.lin.bias.detach().cpu().double().numpy()
    m = spkm.linear(W, bb, [(name, tr, base)], norm=(prep.mean, prep.std), budget_ms=prep.budget,
                    info=prep.info(run_id))
    preds = {sp: runlib.predict_model(m, prep.sd[sp], prep.target, prep.decode) for sp in ("val", "test")}
    check = runlib.check_close(preds["val"][name], yv, "Lo output: torch vs .spkm")
    arch = f"Lo et al. asymmetric linear {spkm.F}->1 (alpha={hp['alpha']:g}, L1={hp.get('l1', 0):g})"
    details = (f"loss pos(p-y)^2 + alpha*neg(p-y)^2 + L1; single output named {name} by its validation "
               f"coverage {cov:.3f} (an alpha-expectile, not a calibrated quantile)")
    return m, arch, details, tt, dict(info, export_check=check, val_coverage=cov, params=spkm.F + 1), preds, \
        {"model.pt": lambda p: torch.save(net.state_dict(), p)}, dev


def train_mlp(args, ds, cfg, exp, prep, run_id):
    torch, N, dev = _torch_setup(args, args.seed)
    hp, arch_hp = exp["hp"], exp["arch"]
    net = N.MLPNet(arch_hp["hidden"], len(prep.quantiles), prep.pmiss, prep.mask)
    obj = N.Objective(prep.quantiles, hp.get("bce_weight", 1.0) if prep.pmiss else 0.0, device=dev)
    t = Timer()
    info = {}
    if not args.dry_run:
        info = N.fit_tabular(net, obj, prep.Xn["train"], prep.Z["train"], prep.Mb["train"], prep.Xn["val"],
                             prep.Z["val"], prep.Mb["val"], hp, dev, args.seed)
    tt = t.s()
    net.to(dev)
    m = spkm.mlp(net.export_layers(), prep.outputs, norm=(prep.mean, prep.std), budget_ms=prep.budget,
                 info=prep.info(run_id))
    preds, check = _torch_predict_and_check(prep, m, net, dev)
    sizes = "-".join(str(s) for s in [spkm.F] + arch_hp["hidden"]) + f"-{len(prep.outputs)}"
    arch = f"MLP {sizes}, ReLU, monotone quantile head" + ("" if prep.pmiss else " (no p_miss head)")
    details = (f"trunk {arch_hp['hidden']} ReLU; head: location a and relu increments (p50<=p95<=p99 by "
               f"construction), exported as 2 DENS layers (relu pairs + identity sum); mask {exp['features']}; "
               f"target {prep.target.name}")
    return m, arch, details, tt, dict(info, export_check=check, params=N.count_params(net)), preds, \
        {"model.pt": lambda p: torch.save(net.state_dict(), p)}, dev


def train_gru(args, ds, cfg, exp, prep, run_id):
    torch, N, dev = _torch_setup(args, args.seed)
    hp, a = exp["hp"], exp["arch"]
    if exp["features"] != "all":
        raise SystemExit("feature masks are not supported for sequence models")
    hm, hs = prep.hist_norm()
    net = N.GRUNet(a["H"], a["head_hidden"], len(prep.quantiles), prep.pmiss)
    obj = N.Objective(prep.quantiles, hp.get("bce_weight", 1.0) if prep.pmiss else 0.0, device=dev)
    t = Timer()
    info = {}
    if not args.dry_run:
        info = N.fit_gru(net, obj, _seqdict(prep, "train"), _seqdict(prep, "val"), hp, dev, args.seed)
    tt = t.s()
    net.to(dev)
    m = net.export(prep.outputs, (prep.mean, prep.std), (hm, hs), prep.budget, prep.info(run_id))
    preds, check = _torch_predict_and_check(prep, m, net, dev, kind="gru")
    arch = f"streaming GRU H={a['H']} over {spkm.D}-dim history vectors + head [{a['H']}+{spkm.F}]-{a['head_hidden']}-q"
    details = (f"nn.GRU gate order r,z,n; state carried across frames (stream=1, updated in endFrame); head on "
               f"[h, xn]; trained with truncated BPTT windows {hp.get('window')}+{hp.get('burn_in')} burn-in")
    return m, arch, details, tt, dict(info, export_check=check, params=N.count_params(net)), preds, \
        {"model.pt": lambda p: torch.save(net.state_dict(), p)}, dev


def train_tcn(args, ds, cfg, exp, prep, run_id):
    torch, N, dev = _torch_setup(args, args.seed)
    hp, a = exp["hp"], exp["arch"]
    if exp["features"] != "all":
        raise SystemExit("feature masks are not supported for sequence models")
    hm, hs = prep.hist_norm()
    net = N.TCNNet(a["channels"], a["kernel"], a["dilations"], a["L"], a["head_hidden"], len(prep.quantiles),
                   prep.pmiss)
    obj = N.Objective(prep.quantiles, hp.get("bce_weight", 1.0) if prep.pmiss else 0.0, device=dev)
    t = Timer()
    info = {}
    if not args.dry_run:
        info = N.fit_tcn(net, obj, _seqdict(prep, "train"), _seqdict(prep, "val"), hp, dev, args.seed)
    tt = t.s()
    net.to(dev)
    m = net.export(prep.outputs, (prep.mean, prep.std), (hm, hs), prep.budget, prep.info(run_id))
    preds, check = _torch_predict_and_check(prep, m, net, dev, kind="tcn")
    arch = (f"TCN {len(a['dilations'])}x{a['channels']}ch k={a['kernel']} d={','.join(map(str, a['dilations']))} "
            f"L={a['L']} + head [{a['channels']}+{spkm.F}]-{a['head_hidden']}-q")
    details = ("causal dilated Conv1d (PyTorch layout), ReLU, residual on layers 2+, zero padding in normalised "
               "space at session start; head on [last-step channels, xn]")
    return m, arch, details, tt, dict(info, export_check=check, params=N.count_params(net)), preds, \
        {"model.pt": lambda p: torch.save(net.state_dict(), p)}, dev


# ---------------------------------------------------------------- trees

def train_gbt(args, ds, cfg, exp, prep, run_id):
    import trees as TR
    hp = exp["hp"]
    keep = np.nonzero(prep.mask)[0]
    t = Timer()
    dumps, boosters, info = [], [], {"rounds": {}}
    names = [M.qname(q) for q in prep.quantiles] + (["p_miss"] if prep.pmiss else [])
    if args.dry_run:
        rng = np.random.default_rng(args.seed)
        for n in names:
            dumps.append(TR.random_dump(rng, 6, hp.get("max_depth", 4), len(keep)))
    else:
        tr, va = prep.sd["train"], prep.sd["val"]
        Xtr, Xva = np.ascontiguousarray(tr.X[:, keep]), np.ascontiguousarray(va.X[:, keep])
        for i, q in enumerate(prep.quantiles):
            log(f"  LightGBM quantile {q}")
            b, bi = TR.fit_lgb(Xtr, prep.Z["train"][:, i], Xva, prep.Z["val"][:, i], "quantile", q, hp, args.seed,
                               args.workers, hp["n_rounds"])
            boosters.append(b)
            info["rounds"][M.qname(q)] = bi["best_iteration"]
        if prep.pmiss:
            log("  LightGBM p_miss (binary)")
            b, bi = TR.fit_lgb(Xtr, prep.Mb["train"], Xva, prep.Mb["val"], "binary", None, hp, args.seed, args.workers,
                               hp["pmiss_rounds"])
            boosters.append(b)
            info["rounds"]["p_miss"] = bi["best_iteration"]
        info["lgb_params"] = bi["params"]
        dumps = [TR.booster_dump(b) for b in boosters]
    tt = t.s()
    m = TR.export_gbt(dumps, prep.outputs, prep.budget, prep.info(run_id), feat_map=keep)
    check = {}
    if boosters:  # exported trees vs LightGBM's own raw scores; fold a constant init score into base if needed
        va = prep.sd["val"]
        rows = np.arange(min(len(va.X), 20000))
        P = __import__("fastpred").StepPredictor(m, 1)
        raw = P.trees.raw(P.xn(va.X[rows]), len(boosters))
        for o, b in enumerate(boosters):
            lg = b.predict(va.X[rows][:, keep], raw_score=True, num_iteration=b.best_iteration or None)
            d = lg - raw[:, o]
            if np.std(d) < 1e-4 and abs(np.mean(d)) > 1e-6:
                m.params["base"][o] += np.float32(np.mean(d))
                log(f"  folded constant {np.mean(d):.6f} into base[{o}]")
            check[names[o]] = runlib.check_close(raw[:, o] + (np.mean(d) if np.std(d) < 1e-4 else 0), lg,
                                                 f"{names[o]}: LightGBM vs export")
    preds = {sp: runlib.predict_model(m, prep.sd[sp], prep.target, prep.decode) for sp in ("val", "test")}
    check["runner"] = runlib.runner_parity(m, prep.sd["val"])
    n_trees = len(m.params["trees"])
    arch = f"GBT (LightGBM) {n_trees} trees, depth<={hp.get('max_depth', 4)}, {len(names)} outputs"
    details = (f"one quantile booster per output (objective=quantile) + binary p_miss booster; raw features (no NORM), "
               f"float32-floored thresholds; rounds {info.get('rounds')}; mask {exp['features']}; "
               f"target {prep.target.name}")
    native = {}
    for n, b in zip(names, boosters):
        native[f"{n}.txt"] = (lambda bb: (lambda p: bb.save_model(str(p))))(b)
    return m, arch, details, tt, dict(info, export_check=check, n_trees=n_trees), preds, native, None


# ---------------------------------------------------------------- RLS

def train_rls(args, ds, cfg, exp, prep, run_id):
    import linear as L
    hp = exp["hp"]
    rc = cfg["rls"]
    idx = np.array(Dt.name_index(rc["features"])[: rc.get("max_features", 16)], np.uint32)
    base_name = rc.get("base")
    base_idx = spkm.FEATURE_NAMES.index(base_name) if base_name in spkm.FEATURE_NAMES else -1
    qs = np.array(prep.quantiles)
    tr, va = prep.sd["train"], prep.sd["val"]
    zb = lambda sd: sd.y.astype(np.float64) - (sd.X[:, base_idx].astype(np.float64) if base_idx >= 0 else 0)
    t = Timer()
    grid = []
    best = None
    lams = hp["lams"] if not args.dry_run else hp["lams"][:1]
    for lam in lams:
        if args.dry_run:
            n = len(idx) + 1
            w, P, qo, v0 = np.zeros(n), np.eye(n), np.zeros(len(qs)), 1.0
        else:
            w, P, qo, v0 = L.rls_init(prep.Xn["train"], zb(tr), idx, lam, hp.get("ridge", 1e-3), qs)
        # dry run: forward pass only (online=False), so the per-frame RLS update never touches the data
        m = L.rls_model(idx, w, P, qs, qo, v0, lam, hp["gamma"], hp["var_alpha"], hp["max_trace"], base_idx,
                        (prep.mean, prep.std), prep.budget, prep.info(run_id), online=not common.no_fit())
        pv = runlib.predict_model(m, va)
        score = M.pinball(va.y, pv["t_p95_ms"], 0.95) + M.pinball(va.y, pv["t_p50_ms"], 0.5)
        grid.append({"lambda": lam, "val_pinball_p50_plus_p95": score})
        log(f"  RLS lambda={lam}: val pinball p50+p95 {score:.5f}")
        if best is None or score < best[0]:
            best = (score, lam, m, pv)
    tt = t.s()
    _, lam, m, pv = best
    preds = {"val": pv, "test": runlib.predict_model(m, prep.sd["test"])}
    check = runlib.runner_parity(m, va)
    names = [spkm.FEATURE_NAMES[i] for i in idx]
    arch = f"online RLS, {len(idx)} features + bias, base {base_name}, lambda={lam}"
    details = (f"features {names}; offline ridge start values and P from one forgetting window; ACI quantile offsets "
               f"(gamma={hp['gamma']}), Gaussian p_miss (var_alpha={hp['var_alpha']}); evaluated online per session "
               f"(engine endFrame update)")
    return m, arch, details, tt, {"lambda_grid": grid, "export_check": check}, preds, {}, None


KINDS = {"linear_qr": train_linear_qr, "lo": train_lo, "mlp": train_mlp, "gru": train_gru, "tcn": train_tcn,
         "gbt": train_gbt, "rls": train_rls}
FAMILY = {"linear_qr": "linear", "lo": "linear", "mlp": "mlp", "gru": "gru", "tcn": "tcn", "gbt": "gbt", "rls": "rls"}


def run_experiment(args, ds, cfg, name, ckpt_dir, res_dir):
    exp = dict(E.EXPERIMENTS[name])
    fam = FAMILY[exp["kind"]]
    label = name + (f"@{args.holdout}" if args.holdout else "")   # transfer runs are separate experiments
    run_id = make_run_id(fam, label)
    log(f"=== {name} ({exp['kind']}, features={exp['features']}, target={exp['target']}, pmiss={exp['pmiss']}) "
        f"-> {run_id}")
    prep = Prep(ds, cfg, exp)
    bad = prep.target.check(prep.sd["train"].X, prep.sd["train"])
    if bad and not args.dry_run:
        log(f"SKIP {name}: {bad}")
        common.write_json(res_dir / "runs" / f"{run_id}.json",
                          {"schema": runlib.RESULT_SCHEMA, "run_id": run_id, "experiment": label, "group": exp["group"],
                           "family": fam, "skipped": bad, "created_utc": common.utc_now()})
        return None
    common.set_seed(args.seed)
    m, arch, details, tt, info, preds, native, dev = KINDS[exp["kind"]](args, ds, cfg, exp, prep, run_id)
    hp = dict(exp["hp"], features=exp["features"], target=exp["target"], pmiss=exp["pmiss"], arch=exp["arch"])
    rec = runlib.finish(args, ds, cfg, group=exp["group"] + ("-transfer" if args.holdout else ""), experiment=label,
                        family=fam, run_id=run_id, model=m,
                        architecture=arch, details=details, feature_mask=prep.mask, target=prep.target,
                        train_time_s=tt, train_info=info, preds=preds, hp=hp, device=dev, native=native,
                        export_note=prep.export_note, ckpt_dir=ckpt_dir, res_dir=res_dir,
                        worst_evals=len(ds.ladder))
    h = rec["headline"].get("test", {})
    log(f"--- {name}: params {rec['model']['num_parameters']:,}, est {rec['model']['cost'].get('ingame_us_est')} us; "
        f"test MAE {h.get('mae_ms', float('nan')):.4f} pinball p95 {h.get('pinball_p95', float('nan')):.4f} "
        f"cov95 {h.get('coverage_p95', float('nan')):.3f}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*", help=f"names from experiments.py: {', '.join(E.EXPERIMENTS)}")
    ap.add_argument("--group", default=None, help="linear | trees | nn | formulation")
    ap.add_argument("--task", type=int, default=None, help="array slice of a group (default $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--device", default="auto", help="auto (cuda if available) | cpu")
    ap.add_argument("--list", action="store_true")
    common.add_common_args(ap)
    args = ap.parse_args(argv)
    if args.list:
        for n, e in E.EXPERIMENTS.items():
            print(f"{e['group']:12s} {n:22s} {e['kind']:9s} features={e['features']} target={e['target']} "
                  f"pmiss={e['pmiss']}")
        return 0
    names = list(args.experiments)
    if args.group:
        task = args.task if args.task is not None else (int(os.environ["SLURM_ARRAY_TASK_ID"])
                                                       if "SLURM_ARRAY_TASK_ID" in os.environ else None)
        names += E.task_list(args.group, task) if task is not None else E.by_group(args.group)
    unknown = [n for n in names if n not in E.EXPERIMENTS]
    if not names or unknown:
        ap.error(f"no/unknown experiments {unknown}; use --list")
    cfg = common.load_config(args.config)
    ckpt_dir, res_dir = common.output_dirs(args)
    log(f"experiments: {names}; feature set {spkm.FEATURE_SET} ({spkm.F} features); outputs -> {ckpt_dir}, {res_dir}"
        + ("  [DRY RUN: no fitting]" if args.dry_run else ""))
    ds = Dt.Dataset(cfg, args.data, args.cache_dir, holdout=args.holdout, max_sessions=args.max_sessions, workers=args.workers,
                    dry_run=args.dry_run, dry_frames=args.dry_frames)
    log(f"dataset {ds.name}: " + ", ".join(f"{k} {v.N:,} frames" for k, v in ds.splits.items()))
    failed = []
    for n in names:
        try:
            run_experiment(args, ds, cfg, n, ckpt_dir, res_dir)
        except Exception:  # keep going with the other experiments of the job; fail the job at the end
            import traceback
            traceback.print_exc()
            failed.append(n)
    if failed:
        log(f"FAILED: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
