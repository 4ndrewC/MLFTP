"""Torch models (linear heads, MLP, streaming GRU, TCN), losses, training loops and the
exporters to the engine's .spkm format (MODEL_CONTRACT section 5).

Monotone quantiles (p50 <= p95 <= p99) are structural for MLP/GRU/TCN: the head emits a
location `a` and non-negative increments relu(d_i), and the export writes that as two
DENS layers using relu(a) - relu(-a) = a, so the engine computes exactly the same thing:

    layer 1 (relu):     [a, -a, d_1 .. d_{k-1}, l, -l]          (l = p_miss logit)
    layer 2 (identity): p50 = a+ - a-, p95 = p50 + d_1, ..., p_miss logit = l+ - l-

Imports torch, so only the NN/linear entry points import this module.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import spkm
from common import log, require_fitting

F, D = spkm.F, spkm.D


def get_device(prefer="auto"):
    if prefer == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


# ---------------------------------------------------------------- heads

class QuantileHead(nn.Module):
    """Monotone quantile head (+ optional p_miss logit)."""

    def __init__(self, n_in, nq, pmiss=True, delta_init=0.5):
        super().__init__()
        self.nq, self.pmiss = nq, pmiss
        self.lin = nn.Linear(n_in, nq + (1 if pmiss else 0))
        with torch.no_grad():
            self.lin.bias[1:nq] = delta_init  # start with positive gaps so relu passes gradient

    def forward(self, h):
        z = self.lin(h)  # (..., nq [+1]); works for (B, k) and (B, T, k)
        a = z[..., :1]
        d = torch.relu(z[..., 1:self.nq])
        Q = torch.cat([a, a + torch.cumsum(d, -1)], -1) if self.nq > 1 else a
        logit = z[..., self.nq] if self.pmiss else None
        return Q, logit

    def export_layers(self):
        W = self.lin.weight.detach().cpu().double().numpy()
        b = self.lin.bias.detach().cpu().double().numpy()
        nq = self.nq
        rows, bias = [W[0], -W[0]], [b[0], -b[0]]
        for i in range(1, nq):
            rows.append(W[i])
            bias.append(b[i])
        if self.pmiss:
            rows += [W[nq], -W[nq]]
            bias += [b[nq], -b[nq]]
        W1, b1 = np.stack(rows), np.array(bias)
        n1 = len(rows)
        n_out = nq + (1 if self.pmiss else 0)
        W2 = np.zeros((n_out, n1))
        for q in range(nq):
            W2[q, 0], W2[q, 1] = 1, -1
            W2[q, 2:2 + q] = 1
        if self.pmiss:
            W2[nq, n1 - 2], W2[nq, n1 - 1] = 1, -1
        return [(W1, b1, "relu"), (W2, np.zeros(n_out), "identity")]


class LinearHeads(nn.Module):
    """Independent linear quantile rows (+ p_miss row): exports as the `linear` family (LINR)."""

    def __init__(self, nq, pmiss=True, mask=None):
        super().__init__()
        self.nq, self.pmiss = nq, pmiss
        self.lin = nn.Linear(F, nq + (1 if pmiss else 0))
        self.register_buffer("mask", torch.ones(F) if mask is None else torch.as_tensor(mask, dtype=torch.float32))

    def forward(self, xn):
        z = self.lin(xn * self.mask)
        return z[:, :self.nq], (z[:, self.nq] if self.pmiss else None)

    def export_linear(self):
        W = (self.lin.weight * self.mask[None, :]).detach().cpu().double().numpy()
        return W, self.lin.bias.detach().cpu().double().numpy()


# ---------------------------------------------------------------- networks

class MLPNet(nn.Module):
    def __init__(self, hidden, nq, pmiss=True, mask=None):
        super().__init__()
        layers, n = [], F
        for h in hidden:
            layers += [nn.Linear(n, h), nn.ReLU()]
            n = h
        self.trunk = nn.Sequential(*layers)
        self.head = QuantileHead(n, nq, pmiss)
        self.register_buffer("mask", torch.ones(F) if mask is None else torch.as_tensor(mask, dtype=torch.float32))
        self.hidden = list(hidden)

    def forward(self, xn):
        return self.head(self.trunk(xn * self.mask))

    def export_layers(self):
        out = []
        lins = [m for m in self.trunk if isinstance(m, nn.Linear)]
        for i, l in enumerate(lins):
            W = l.weight.detach().cpu().double().numpy().copy()
            if i == 0:
                W = W * self.mask.cpu().numpy()[None, :]
            out.append((W, l.bias.detach().cpu().double().numpy(), "relu"))
        if not lins:  # head directly on the (masked) features
            hl = self.head.export_layers()
            hl[0] = (hl[0][0] * self.mask.cpu().numpy()[None, :], hl[0][1], "relu")
            return hl
        return out + self.head.export_layers()


class GRUNet(nn.Module):
    """Streaming GRU over history vectors; head on [h, xn] (contract: gru, stream=1)."""

    def __init__(self, H, head_hidden, nq, pmiss=True):
        super().__init__()
        self.H = H
        self.gru = nn.GRU(D, H, batch_first=True)
        self.fc = nn.Linear(H + F, head_hidden)
        self.head = QuantileHead(head_hidden, nq, pmiss)

    def states(self, hn_prev, h0=None):
        """hn_prev: (B, T, D) history vectors of frames t-1 (row 0 is ignored: state starts at h0).
        Returns (B, T, H): the state the engine has when predicting frame t."""
        B, T, _ = hn_prev.shape
        h0 = torch.zeros(1, B, self.H, device=hn_prev.device) if h0 is None else h0
        if T == 1:
            return h0.transpose(0, 1)
        out, _ = self.gru(hn_prev[:, 1:], h0)
        return torch.cat([h0.transpose(0, 1), out], 1)

    def head_out(self, s, xn):
        return self.head(torch.relu(self.fc(torch.cat([s, xn], -1))))

    def export(self, outputs, norm, hnorm, budget_ms, info):
        g = self.gru
        Wih = g.weight_ih_l0.detach().cpu().double().numpy()
        Whh = g.weight_hh_l0.detach().cpu().double().numpy()
        bih = g.bias_ih_l0.detach().cpu().double().numpy()
        bhh = g.bias_hh_l0.detach().cpu().double().numpy()
        head = [(self.fc.weight.detach().cpu().double().numpy(), self.fc.bias.detach().cpu().double().numpy(), "relu")]
        head += self.head.export_layers()
        return spkm.gru(Wih, Whh, bih, bhh, head, outputs, history_len=1, stream=True, norm=norm, hnorm=hnorm,
                        budget_ms=budget_ms, info=info)


class TCNNet(nn.Module):
    """Causal dilated conv stack over the last L history vectors; head on [last-step channels, xn]."""

    def __init__(self, channels, kernel, dilations, L, head_hidden, nq, pmiss=True):
        super().__init__()
        self.L, self.k, self.dil = L, kernel, list(dilations)
        convs, cin = [], D
        for d in dilations:
            convs.append(nn.Conv1d(cin, channels, kernel, dilation=d))
            cin = channels
        self.convs = nn.ModuleList(convs)
        self.res = [i > 0 for i in range(len(convs))]  # residual where c_in == c_out
        self.fc = nn.Linear(channels + F, head_hidden)
        self.head = QuantileHead(head_hidden, nq, pmiss)

    def encode(self, seq):
        """seq: (B, L, D) normalised, zeros where no history exists -> (B, C)."""
        x = seq.transpose(1, 2)
        for c, d, r in zip(self.convs, self.dil, self.res):
            y = torch.relu(c(Fn.pad(x, ((self.k - 1) * d, 0))))
            x = y + x if r else y
        return x[:, :, -1]

    def forward(self, seq, xn):
        return self.head(torch.relu(self.fc(torch.cat([self.encode(seq), xn], -1))))

    def export(self, outputs, norm, hnorm, budget_ms, info):
        convs = [dict(W=c.weight.detach().cpu().double().numpy(), b=c.bias.detach().cpu().double().numpy(),
                      dilation=d, act="relu", residual=r) for c, d, r in zip(self.convs, self.dil, self.res)]
        head = [(self.fc.weight.detach().cpu().double().numpy(), self.fc.bias.detach().cpu().double().numpy(), "relu")]
        head += self.head.export_layers()
        return spkm.tcn(convs, head, outputs, history_len=self.L, norm=norm, hnorm=hnorm, budget_ms=budget_ms,
                        info=info)


# ---------------------------------------------------------------- losses

def pinball_loss(Q, z, taus):
    """Q: (B, k); z: (B,) or (B, k) per-quantile targets; taus: (k,) tensor."""
    d = (z if z.dim() == 2 else z[:, None]) - Q
    return torch.maximum(taus * d, (taus - 1) * d).mean(0).sum()


def crossing_penalty(Q):
    return torch.relu(Q[:, :-1] - Q[:, 1:]).mean() if Q.shape[1] > 1 else Q.new_zeros(())


def asym_sq_loss(pred, z, alpha):
    """Lo et al. (MICRO 2015): pos(p - y)^2 + alpha * neg(p - y)^2 (under-prediction weighted alpha)."""
    e = pred - z
    return (torch.relu(e) ** 2 + alpha * torch.relu(-e) ** 2).mean()


class Objective:
    def __init__(self, taus, bce_weight=1.0, cross_weight=0.0, pos_weight=None, device="cpu"):
        self.taus = torch.as_tensor(taus, dtype=torch.float32, device=device)
        self.bce_weight, self.cross_weight = bce_weight, cross_weight
        self.pos_weight = None if pos_weight is None else torch.tensor(pos_weight, device=device)

    def __call__(self, Q, logit, z, miss):
        loss = pinball_loss(Q, z, self.taus)
        if logit is not None and self.bce_weight > 0:
            loss = loss + self.bce_weight * Fn.binary_cross_entropy_with_logits(logit, miss, pos_weight=self.pos_weight)
        if self.cross_weight > 0:
            loss = loss + self.cross_weight * crossing_penalty(Q)
        return loss


# ---------------------------------------------------------------- training loops

def _tensor(a, device, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)


def fit_tabular(net, loss_fn, Xtr, Ztr, Mtr, Xva, Zva, Mva, hp, device, seed, forward=None):
    """Mini-batch AdamW + cosine LR, early stopping on validation loss; restores the best weights.
    X*: normalised features (N, F) numpy float32, Z*: training-space targets, M*: miss labels."""
    require_fitting("nn.fit_tabular")
    forward = forward or (lambda n, xb: n(xb))
    g = torch.Generator(device="cpu").manual_seed(seed)
    net.to(device)
    Xtr_t, Ztr_t, Mtr_t = _tensor(Xtr, device), _tensor(Ztr, device), _tensor(Mtr, device)
    Xva_t, Zva_t, Mva_t = _tensor(Xva, device), _tensor(Zva, device), _tensor(Mva, device)
    opt = torch.optim.AdamW(net.parameters(), lr=hp.get("lr", 2e-3), weight_decay=hp.get("weight_decay", 1e-4))
    bs, epochs = hp.get("batch", 4096), hp.get("epochs", 30)
    steps = max(1, math.ceil(len(Xtr) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g).to(device)
        tot = 0.0
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            Q, logit = forward(net, Xtr_t[idx])
            loss = loss_fn(Q, logit, Ztr_t[idx], Mtr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if hp.get("clip"):
                nn.utils.clip_grad_norm_(net.parameters(), hp["clip"])
            opt.step()
            sched.step()
            tot += float(loss)
        vl = eval_tabular_loss(net, loss_fn, Xva_t, Zva_t, Mva_t, forward)
        hist.append({"epoch": ep, "train_loss": tot / steps, "val_loss": vl})
        log(f"  epoch {ep:3d} train {tot / steps:.5f} val {vl:.5f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= hp.get("patience", 5):
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return {"epochs_run": len(hist), "best_val_loss": best, "history": hist}


@torch.no_grad()
def eval_tabular_loss(net, loss_fn, X, Z, M, forward, bs=65536):
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X), bs):
        Q, logit = forward(net, X[i:i + bs])
        tot += float(loss_fn(Q, logit, Z[i:i + bs], M[i:i + bs])) * len(Q)
        n += len(Q)
    return tot / max(n, 1)


def _session_windows(offsets, W, rng, n):
    """n random window starts (row index) with at least W rows left in the session."""
    starts, ends = offsets[:-1], offsets[1:]
    ok = np.nonzero(ends - starts >= W)[0]
    if len(ok) == 0:
        raise ValueError(f"no session has {W} frames")
    w = (ends[ok] - starts[ok] - W + 1).astype(np.float64)
    s = rng.choice(ok, size=n, p=w / w.sum())
    return starts[s] + (rng.random(n) * (ends[s] - starts[s] - W + 1)).astype(np.int64), starts[s]


def fit_gru(net, loss_fn, tr, va, hp, device, seed):
    """Truncated BPTT on random windows (burn-in frames excluded from the loss unless the window starts
    at the session start, where h = 0 is exact); validation streams whole sessions like the engine."""
    require_fitting("nn.fit_gru")
    rng = np.random.default_rng(seed)
    net.to(device)
    T, burn, B = hp.get("window", 256), hp.get("burn_in", 64), hp.get("batch", 64)
    W = T + burn
    Xt, Ht, Zt, Mt = (_tensor(tr[k], device) for k in ("Xn", "Hn", "Z", "M"))
    opt = torch.optim.AdamW(net.parameters(), lr=hp.get("lr", 2e-3), weight_decay=hp.get("weight_decay", 1e-4))
    epochs, steps = hp.get("epochs", 30), hp.get("steps_per_epoch", 300)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    best, best_state, bad, hist = float("inf"), None, 0, []
    ar = torch.arange(W, device=device)
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for _ in range(steps):
            st, s0 = _session_windows(tr["offsets"], W, rng, B)
            at_start = torch.as_tensor(st == s0, device=device)
            rows = torch.as_tensor(st, device=device)[:, None] + ar[None, :]
            prev = torch.clamp(rows - 1, min=0)
            S = net.states(Ht[prev])
            Q, logit = net.head_out(S, Xt[rows])
            keep = (ar[None, :] >= burn) | at_start[:, None]
            k = keep.reshape(-1)
            Zr = Zt[rows]
            Zr = Zr.reshape(-1, Zr.shape[-1]) if Zr.dim() == 3 else Zr.reshape(-1)
            loss = loss_fn(Q.reshape(-1, Q.shape[-1])[k], None if logit is None else logit.reshape(-1)[k],
                           Zr[k], Mt[rows].reshape(-1)[k])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), hp.get("clip", 1.0))
            opt.step()
            sched.step()
            tot += float(loss)
        vl = gru_stream_loss(net, loss_fn, va, device)
        hist.append({"epoch": ep, "train_loss": tot / steps, "val_loss": vl})
        log(f"  epoch {ep:3d} train {tot / steps:.5f} val {vl:.5f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= hp.get("patience", 5):
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return {"epochs_run": len(hist), "best_val_loss": best, "history": hist}


@torch.no_grad()
def gru_stream_outputs(net, sd, device):
    """Whole-session streaming (h = 0 at session start): (Q, logit) numpy for every row."""
    net.eval()
    Qs, Ls = [], []
    for a, b in zip(sd["offsets"][:-1], sd["offsets"][1:]):
        Hn = _tensor(sd["Hn"][a:b], device)
        # states() expects row t = history vector of frame t-1 (row 0 ignored), as in fit_gru (Ht[rows - 1]).
        Hn = torch.cat([Hn[:1], Hn[:-1]], 0)[None]
        Xn = _tensor(sd["Xn"][a:b], device)[None]
        S = net.states(Hn)
        Q, logit = net.head_out(S, Xn)
        Qs.append(Q[0].cpu().numpy())
        if logit is not None:
            Ls.append(logit[0].cpu().numpy())
    return np.concatenate(Qs), (np.concatenate(Ls) if Ls else None)


@torch.no_grad()
def gru_stream_loss(net, loss_fn, va, device):
    Q, L = gru_stream_outputs(net, va, device)
    return float(loss_fn(_tensor(Q, device), None if L is None else _tensor(L, device), _tensor(va["Z"], device),
                         _tensor(va["M"], device)))


def tcn_windows(Hn_t, rows, row_start, L):
    """(B, L, D) history windows for the given rows: vectors of frames rows-L .. rows-1, zero before
    the session start (zero padding in normalised space, as the engine)."""
    idx = rows[:, None] - L + torch.arange(L, device=rows.device)[None, :]
    valid = idx >= row_start[rows][:, None]
    seq = Hn_t[torch.clamp(idx, min=0)]
    return seq * valid[:, :, None]


def fit_tcn(net, loss_fn, tr, va, hp, device, seed):
    require_fitting("nn.fit_tcn")
    Ht = _tensor(tr["Hn"], device)
    Hv = _tensor(va["Hn"], device)
    rs_tr = torch.as_tensor(tr["row_start"], device=device)
    rs_va = torch.as_tensor(va["row_start"], device=device)
    L = net.L

    def fwd_tr(n, xb_rows):
        return n(tcn_windows(Ht, xb_rows, rs_tr, L), Xtr_t[xb_rows])

    Xtr_t = _tensor(tr["Xn"], device)
    Xva_t = _tensor(va["Xn"], device)

    g = torch.Generator(device="cpu").manual_seed(seed)
    net.to(device)
    Ztr, Mtr = _tensor(tr["Z"], device), _tensor(tr["M"], device)
    Zva, Mva = _tensor(va["Z"], device), _tensor(va["M"], device)
    opt = torch.optim.AdamW(net.parameters(), lr=hp.get("lr", 2e-3), weight_decay=hp.get("weight_decay", 1e-4))
    bs, epochs = hp.get("batch", 2048), hp.get("epochs", 30)
    N = len(tr["Xn"])
    steps = min(max(1, math.ceil(N / bs)), hp.get("max_steps_per_epoch", 10 ** 9))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(N, generator=g).to(device)
        tot = 0.0
        for i in range(steps):
            rows = perm[i * bs:(i + 1) * bs]
            Q, logit = fwd_tr(net, rows)
            loss = loss_fn(Q, logit, Ztr[rows], Mtr[rows])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), hp.get("clip", 1.0))
            opt.step()
            sched.step()
            tot += float(loss)
        with torch.no_grad():
            net.eval()
            vt, n = 0.0, 0
            for a in range(0, len(va["Xn"]), 16384):
                rows = torch.arange(a, min(len(va["Xn"]), a + 16384), device=device)
                Q, logit = net(tcn_windows(Hv, rows, rs_va, L), Xva_t[rows])
                vt += float(loss_fn(Q, logit, Zva[rows], Mva[rows])) * len(rows)
                n += len(rows)
            vl = vt / max(n, 1)
        hist.append({"epoch": ep, "train_loss": tot / steps, "val_loss": vl})
        log(f"  epoch {ep:3d} train {tot / steps:.5f} val {vl:.5f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= hp.get("patience", 5):
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return {"epochs_run": len(hist), "best_val_loss": best, "history": hist}


def count_params(net):
    return int(sum(p.numel() for p in net.parameters()))
