"""Experiment registry: every trainable run of the study, by group.

    kind      linear_qr | lo | rls | gbt | mlp | gru | tcn
    features  feature-group mask (data.feature_mask): "all", "history+system+fingerprint?", ...
              ("?" marks a group that may not exist in the active feature set)
    target    normaliser name from the config (targets.py): none | resid | lognaive | tref | treftail | cycles
              (tref/treftail need fingerprinted logs: they are skipped, with a note, when t_ref_ms is 0)
    pmiss     "head" (dedicated classifier output) | "derived" (no head; P(miss) from the quantiles)

Sizes are chosen against the 50 us/frame budget with the controller's worst case (one eval per
ladder level), using cost.estimate() (scaled from the measured laptop numbers).
"""

HIST = "history+system+fingerprint?"
PLAN = "planned+system+fingerprint?"
PLANPREV = "planned+prevwork+system+fingerprint?"

_MLP_HP = dict(epochs=40, batch=4096, lr=2e-3, weight_decay=1e-4, patience=6, bce_weight=1.0)
_GBT_HP = dict(n_rounds=80, pmiss_rounds=60, learning_rate=0.1, num_leaves=15, max_depth=4, min_data_in_leaf=200,
               early_stopping=30)

EXPERIMENTS = {
    # ---- linear family (+ RLS online)
    "linear_qr": dict(group="linear", kind="linear_qr",
                      hp=dict(epochs=25, batch=16384, lr=1e-2, weight_decay=0.0, patience=5, bce_weight=1.0,
                              cross_weight=1.0)),
    "linear_qr_resid": dict(group="linear", kind="linear_qr", target="resid",
                            hp=dict(epochs=25, batch=16384, lr=1e-2, weight_decay=0.0, patience=5, bce_weight=1.0,
                                    cross_weight=1.0)),
    "lo_a100": dict(group="linear", kind="lo", hp=dict(alpha=100.0, l1=1e-4, epochs=25, batch=16384, lr=1e-2,
                                                       patience=5)),
    "lo_a10": dict(group="linear", kind="lo", hp=dict(alpha=10.0, l1=1e-4, epochs=25, batch=16384, lr=1e-2,
                                                      patience=5)),
    "rls_hybrid": dict(group="linear", kind="rls", hp=dict(lams=[0.99, 0.995, 0.999, 0.9995], gamma=0.005,
                                                           var_alpha=0.01, ridge=1e-3, max_trace=1e4)),
    # ---- trees
    "gbt_q": dict(group="trees", kind="gbt", hp=dict(_GBT_HP)),
    "gbt_q_large": dict(group="trees", kind="gbt",
                        hp=dict(_GBT_HP, n_rounds=250, pmiss_rounds=120, learning_rate=0.05,
                                note="accuracy reference: over the worst-case (all-levels) budget")),
    # ---- neural nets
    "mlp": dict(group="nn", kind="mlp", arch=dict(hidden=[64, 64]), hp=dict(_MLP_HP)),
    "gru_stream": dict(group="nn", kind="gru", arch=dict(H=32, head_hidden=32),
                       hp=dict(epochs=40, batch=64, window=256, burn_in=64, steps_per_epoch=300, lr=2e-3,
                               weight_decay=1e-4, patience=6, bce_weight=1.0, clip=1.0)),
    "tcn": dict(group="nn", kind="tcn", arch=dict(channels=24, kernel=3, dilations=[1, 2, 4], L=16, head_hidden=32),
                hp=dict(epochs=30, batch=2048, lr=2e-3, weight_decay=1e-4, patience=5, bce_weight=1.0, clip=1.0,
                        max_steps_per_epoch=400)),
    # ---- formulation: inputs (history-only vs planned-work-only vs both), MLP and GBT
    "f_mlp_all": dict(group="formulation", kind="mlp", arch=dict(hidden=[64, 64]), hp=dict(_MLP_HP)),
    "f_mlp_hist": dict(group="formulation", kind="mlp", features=HIST, arch=dict(hidden=[64, 64]), hp=dict(_MLP_HP)),
    "f_mlp_plan": dict(group="formulation", kind="mlp", features=PLAN, arch=dict(hidden=[64, 64]), hp=dict(_MLP_HP)),
    "f_mlp_planprev": dict(group="formulation", kind="mlp", features=PLANPREV, arch=dict(hidden=[64, 64]),
                           hp=dict(_MLP_HP)),
    "f_gbt_all": dict(group="formulation", kind="gbt", hp=dict(_GBT_HP)),
    "f_gbt_hist": dict(group="formulation", kind="gbt", features=HIST, hp=dict(_GBT_HP)),
    "f_gbt_plan": dict(group="formulation", kind="gbt", features=PLAN, hp=dict(_GBT_HP)),
    "f_gbt_planprev": dict(group="formulation", kind="gbt", features=PLANPREV, hp=dict(_GBT_HP)),
    # ---- formulation: targets (raw ms vs residual vs hardware-normalised)
    "f_mlp_resid": dict(group="formulation", kind="mlp", target="resid", arch=dict(hidden=[64, 64]), hp=dict(_MLP_HP)),
    "f_mlp_lognaive": dict(group="formulation", kind="mlp", target="lognaive", arch=dict(hidden=[64, 64]),
                           hp=dict(_MLP_HP)),
    "f_mlp_tref": dict(group="formulation", kind="mlp", target="tref", arch=dict(hidden=[64, 64]), hp=dict(_MLP_HP)),
    "f_mlp_treftail": dict(group="formulation", kind="mlp", target="treftail", arch=dict(hidden=[64, 64]),
                           hp=dict(_MLP_HP)),
    "f_gbt_tref": dict(group="formulation", kind="gbt", target="tref", hp=dict(_GBT_HP)),
    "f_mlp_cycles": dict(group="formulation", kind="mlp", target="cycles", arch=dict(hidden=[64, 64]),
                         hp=dict(_MLP_HP)),
    "f_gbt_lognaive": dict(group="formulation", kind="gbt", target="lognaive", hp=dict(_GBT_HP)),
    # ---- formulation: p_miss from a dedicated head vs derived from the quantiles
    "f_mlp_pmiss_derived": dict(group="formulation", kind="mlp", pmiss="derived", arch=dict(hidden=[64, 64]),
                                hp=dict(_MLP_HP, bce_weight=0.0)),
}

# Array-job slices: formulation task i runs these experiments in order.
FORMULATION_TASKS = [
    ["f_mlp_all", "f_mlp_hist", "f_mlp_plan", "f_mlp_planprev"],
    ["f_gbt_all", "f_gbt_hist", "f_gbt_plan", "f_gbt_planprev", "f_gbt_lognaive", "f_gbt_tref"],
    ["f_mlp_resid", "f_mlp_lognaive", "f_mlp_tref", "f_mlp_treftail", "f_mlp_cycles"],
    ["f_mlp_pmiss_derived"],
]
NN_TASKS = [["mlp"], ["gru_stream"], ["tcn"]]

for _n, _e in EXPERIMENTS.items():
    _e.setdefault("features", "all")
    _e.setdefault("target", "none")
    _e.setdefault("pmiss", "head")
    _e.setdefault("arch", {})
    _e["name"] = _n


def by_group(group):
    return [n for n, e in EXPERIMENTS.items() if e["group"] == group]


def task_list(group, index):
    tasks = {"formulation": FORMULATION_TASKS, "nn": NN_TASKS}.get(group)
    if tasks is None:
        return by_group(group)
    return tasks[index]
