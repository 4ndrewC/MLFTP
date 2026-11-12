#!/usr/bin/env python3
"""Unit tests for the phase-2 training code on tiny synthetic arrays. Nothing is fitted:
models use random/untrained weights; the point is that metrics, data helpers, exporters and the
vectorised evaluator agree with their definitions and with spkm.Runner (the engine's reference).

    python3 research/frametime/train/tests/test_train_pipeline.py      (torch tests skip without torch)
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common  # noqa: E402,F401
import controllers as C  # noqa: E402
import cost  # noqa: E402
import data as Dt  # noqa: E402
import fastpred  # noqa: E402
import metrics as M  # noqa: E402
import spkm  # noqa: E402
import targets as T  # noqa: E402
import trees as TR  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def runner_outputs(m, X, H, y, offsets):
    out = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        r = spkm.Runner(m)
        hist = []
        for i in range(a, b):
            r.begin_frame(hist[-spkm.MAX_HISTORY:])
            out.append(r.predict(X[i]))
            hist.append(H[i])
            r.end_frame(X[i], H[i], y[i])
    return np.array(out)


def random_frames(rng, n, m=None):
    mean, std = (m.norm if m is not None and m.norm is not None else (np.zeros(spkm.F), np.ones(spkm.F)))
    X = (mean + std * rng.normal(0, 1, (n, spkm.F))).astype(np.float32)
    H = rng.normal(0, 2, (n, spkm.D)).astype(np.float32)
    y = np.abs(4 + rng.normal(0, 1, n)).astype(np.float32)
    return X, H, y


class TestMetrics(unittest.TestCase):
    def test_pinball_and_coverage(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(M.pinball(y, np.array([2.0, 2.0, 2.0]), 0.5), (0.5 + 0 + 0.5) / 3)
        self.assertAlmostEqual(M.pinball(y, np.array([0.0, 0.0, 0.0]), 0.9), 0.9 * 2)
        self.assertAlmostEqual(M.coverage(y, np.array([1.0, 1.0, 5.0])), 2 / 3)

    def test_average_precision(self):
        rng = np.random.default_rng(0)
        lab = rng.random(500) < 0.2
        sc = rng.random(500) + lab * 0.3
        # brute force: mean over positives of precision at their rank (no ties)
        order = np.argsort(-sc)
        l = lab[order]
        prec = np.cumsum(l) / np.arange(1, len(l) + 1)
        self.assertAlmostEqual(M.average_precision(lab, sc), float(prec[l].mean()), places=10)
        try:
            from sklearn.metrics import average_precision_score
            self.assertAlmostEqual(M.average_precision(lab, sc), average_precision_score(lab, sc), places=10)
        except ImportError:
            pass

    def test_crossing(self):
        Q = np.array([[1, 2, 3], [1, 0.5, 3], [2, 2, 2]], float)
        self.assertAlmostEqual(M.crossing_rate(Q), 1 / 3)

    def test_pmiss_from_quantiles(self):
        Q = np.array([[5.0, 10.0, 15.0]] * 4)
        taus = [0.5, 0.95, 0.99]
        p = [M.pmiss_from_quantiles(Q[:1], taus, b)[0] for b in (2.5, 5, 10, 15, 30)]
        self.assertAlmostEqual(p[1], 0.5)
        self.assertAlmostEqual(p[2], 0.05)
        self.assertAlmostEqual(p[3], 0.01)
        self.assertTrue(all(a >= b for a, b in zip(p, p[1:])))
        self.assertGreater(p[4], 0)

    def test_evaluate_runs(self):
        rng = np.random.default_rng(1)
        y = rng.gamma(2, 3, 1000)
        pred = {"t_p50_ms": np.full(1000, 5.0), "t_p95_ms": np.full(1000, 14.0), "t_p99_ms": np.full(1000, 20.0),
                "p_miss": rng.random(1000)}
        r = M.evaluate(pred, y, 16.67)
        self.assertIn("pinball_p95", r["overall"])
        self.assertIn("pmiss_derived_pr_auc", r["overall"])


class TestDataHelpers(unittest.TestCase):
    def test_lag_ema(self):
        y = np.array([1, 2, 3, 10, 20], float)
        off = np.array([0, 3, 5])
        np.testing.assert_allclose(Dt.lag(y, off, 1), [0, 1, 2, 0, 10])
        e = Dt.ema_prev(y, off, 0.2)
        np.testing.assert_allclose(e, [0, 1, 0.8 * 1 + 0.2 * 2, 0, 10])

    def test_masks(self):
        flags = np.zeros(20, np.uint8)
        flags[5] = 2
        flags[10] = 2
        off = np.array([0, 10, 20])
        m = Dt.transition_mask(flags, off, radius=2)
        self.assertTrue(m[3:8].all() and not m[2] and not m[8])
        self.assertTrue(m[10:13].all() and not m[9])  # does not bleed into the previous session
        y = np.ones(40)
        y[20] = 5
        s = Dt.spike_mask(y, np.array([0, 40]))
        self.assertTrue(s[20] and s.sum() == 1)

    def test_feature_mask(self):
        cfg = common.load_config()
        m = Dt.feature_mask(cfg, "history")
        self.assertEqual(int(m.sum()), len(Dt.name_index(cfg["feature_groups"]["history"])))
        m2 = Dt.feature_mask(cfg, "history+nonexistent?")
        self.assertTrue(np.array_equal(m, m2))
        self.assertEqual(int(Dt.feature_mask(cfg, "-history").sum()), spkm.F - int(m.sum()))
        with self.assertRaises(SystemExit):
            Dt.feature_mask(cfg, "nonexistent")

    def test_norm_stats(self):
        A = np.array([[1, 5], [3, 5]], np.float32)
        mean, std = Dt.norm_stats(A)
        np.testing.assert_allclose(std, [1, 1])
        np.testing.assert_allclose(Dt.normalise(A, mean, std)[:, 0], [-1, 1])


class TestTargets(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(2)
        X = np.abs(rng.normal(1, 1, (50, spkm.F))).astype(np.float32) + 0.1
        y = np.abs(rng.normal(5, 2, 50)) + 0.1
        specs = {"none": {"type": "none"}, "resid": {"type": "residual", "feature": "naive_ms"},
                 "lognaive": {"type": "log_ratio", "feature": "naive_ms", "floor": 0.05},
                 "cycles": {"type": "log_ratio_inverse", "feature": "cpu_ghz", "floor": 0.4}}
        for name, s in specs.items():
            t = T.Target(name, s)
            ref = t.ref(X)
            np.testing.assert_allclose(t.decode(t.encode(y, ref), ref), y, rtol=1e-9, err_msg=name)
        self.assertEqual(T.Target("r", specs["resid"]).export(), ("identity", "naive_ms"))
        if "mulexp" in spkm.TRANSFORMS:
            self.assertEqual(T.Target("l", specs["lognaive"]).export(), ("mulexp", "naive_ms"))
        self.assertIsNone(T.Target("c", specs["cycles"]).export())


class TestFastpred(unittest.TestCase):
    def test_vs_runner_all_families(self):
        rng = np.random.default_rng(3)
        for fam in ("linear", "mlp", "gbt", "rls", "gru_stream", "tcn"):
            m = spkm.random_model(fam, rng, "small")
            X, H, y = random_frames(rng, 90, m)
            off = np.array([0, 40, 90])
            ref = runner_outputs(m, X, H, y, off)
            got = fastpred.predict_split(m, X, H, y, off)
            np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5, err_msg=fam)

    def test_step_api_matches_split(self):
        rng = np.random.default_rng(4)
        m = spkm.random_model("rls", rng, "small")
        X, H, y = random_frames(rng, 30, m)
        P = fastpred.StepPredictor(m, 2)
        out = []
        for k in range(15):
            act = np.array([0, 1])
            rows = np.array([k, 15 + k])
            P.begin_frame(act)
            out.append(P.predict(X[rows][:, None, :], act)[:, 0])
            P.end_frame(X[rows], H[rows], y[rows], act)
        out = np.array(out)  # (15, 2, n_out)
        ref = fastpred.predict_split(m, X, H, y, np.array([0, 15, 30]))
        np.testing.assert_allclose(out[:, 0], ref[:15], atol=1e-9)
        np.testing.assert_allclose(out[:, 1], ref[15:], atol=1e-9)


class TestTrees(unittest.TestCase):
    def test_f32_floor(self):
        for t in (0.1, 1 / 3, 2.5, -7.123456789, 1e-8):
            v = TR.f32_floor(t)
            self.assertLessEqual(v, t)
            self.assertGreater(float(np.nextafter(np.float32(v), np.float32(np.inf))), t)  # float(): NEP 50

    def test_dump_export(self):
        rng = np.random.default_rng(5)
        dumps = [TR.random_dump(rng, 7, 4, spkm.F) for _ in range(3)]
        outs = [("t_p50_ms", "identity", "-"), ("t_p95_ms", "identity", "-"), ("p_miss", "sigmoid", "-")]
        m = TR.export_gbt(dumps, outs, 16.67, {})
        X = rng.normal(0, 2, (200, spkm.F)).astype(np.float32)
        got = fastpred.predict_split(m, X, None, None, np.array([0, 200]))
        for o, d in enumerate(dumps):
            ref = np.array([sum(TR.eval_dump_tree(t["tree_structure"], x.astype(np.float64)) for t in d["tree_info"])
                            for x in X])
            if o == 2:
                ref = 1 / (1 + np.exp(-ref))
            np.testing.assert_allclose(got[:, o], ref, rtol=1e-5, atol=1e-5)

    def test_feat_map(self):
        d = {"tree_info": [{"tree_structure": {"split_feature": 1, "threshold": 0.5, "decision_type": "<=",
                                               "default_left": True, "missing_type": "None",
                                               "left_child": {"leaf_value": 1.0}, "right_child": {"leaf_value": 2.0}}}]}
        nodes = TR.tree_from_dump(d["tree_info"][0]["tree_structure"], feat_map=np.array([3, 7]))
        self.assertEqual(nodes[0][0], 7)
        self.assertEqual([n[0] for n in nodes[1:]], [-1, -1])


class TestControllers(unittest.TestCase):
    def test_reactive_panic(self):
        p = common.load_config()["controller"]
        st = C.CtlState(1)
        L = 8
        lv = []
        for t in [20, 20, 20, 20]:
            st.since_change[0] += 1
            new, _ = C.decide_reactive(st, 0, t, 16.67, 15.0, p, L)
            if new != st.level[0]:
                st.since_change[0] = 0
            st.level[0] = new
            lv.append(new)
        self.assertEqual(lv[0], 1)          # ema over target, first change allowed
        self.assertEqual(lv[2], 3)          # third consecutive miss: panic, two levels

    def test_model_decision(self):
        p = common.load_config()["controller"]
        st = C.CtlState(1)
        v = np.array([20.0, 18, 14, 9, 5, 4, 3, 2])
        new, action, ev = C.decide_model(st, 0, v, 15.0, p, 8)
        self.assertEqual((new, action, ev), (2, 1, 3))

    def test_simulator_identity(self):
        """At the session's own quality the simulator returns the logged frame time."""
        cfg = common.load_config()
        n = 50
        rng = np.random.default_rng(6)
        nlev = len(cfg["levers"])
        q = np.zeros((n, nlev), np.uint8)
        a = {"q": q, "render_scale": np.full(n, 2.0, np.float32), "px_clear": np.full(n, 1920 * 1080.0),
             "raster_px": rng.uniform(1e5, 1e6, n), "px_aa": rng.uniform(0, 1e5, n), "px_glow_req": rng.uniform(0, 1e6, n),
             "t_r_particles": rng.uniform(0, 0.2, n), "t_render": rng.uniform(1, 5, n), "t_steps": rng.uniform(0, 0.1, n)}
        a["y"] = (a["t_render"] + a["t_steps"]).astype(np.float32)

        class S:
            pass
        s = S()
        s.a, s.n = a, n
        s.header = {"session": {"logical_w": 960, "logical_h": 540, "out_scale": 2.0}}
        t = C.simulate_costs(s, cfg, [tuple([0] * nlev), tuple(lv["levels"] - 1 for lv in cfg["levers"])])
        np.testing.assert_allclose(t[0], a["y"], rtol=1e-5)
        self.assertTrue(np.all(t[1] < t[0]))


class TestCost(unittest.TestCase):
    def test_tcn_reference(self):
        m = cost.tcn_needed_macs(50, [32, 32, 32], 3, [1, 2, 4], 32, [False, True, True])
        self.assertEqual(m, 7 * 32 * 50 * 3 + 3 * 32 * 32 * 3 + 32 * 32 * 3)
        e = cost.estimate(spkm.random_model("tcn", np.random.default_rng(0), "bench"))
        self.assertAlmostEqual(e["per_frame_extra_us"],
                               25.0 * cost.tcn_needed_macs(spkm.D, [32] * 3, 3, [1, 2, 4], 32, [False, True, True])
                               / cost.REF["tcn_encode"][1], places=3)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestTorchExport(unittest.TestCase):
    """Random-init torch modules (no training) exported to .spkm must match their own forward pass."""

    def setUp(self):
        import nn as N
        self.N = N
        torch.manual_seed(0)
        self.rng = np.random.default_rng(7)

    def test_quantile_head_monotone_and_export(self):
        N = self.N
        head = N.QuantileHead(8, 3, pmiss=True)
        with torch.no_grad():
            head.lin.weight.normal_(0, 2)
        h = torch.randn(500, 8)
        Q, L = head(h)
        self.assertTrue(bool((Q[:, 1:] >= Q[:, :-1] - 1e-6).all()))
        layers = head.export_layers()
        v = h.double().numpy()
        for W, b, a in layers:
            v = v @ W.T + b
            if a == "relu":
                v = np.maximum(v, 0)
        np.testing.assert_allclose(v[:, :3], Q.detach().double().numpy(), atol=1e-5)
        np.testing.assert_allclose(v[:, 3], L.detach().double().numpy(), atol=1e-5)

    def _norm(self):
        return (self.rng.normal(0, 1, spkm.F).astype(np.float32), self.rng.uniform(0.5, 2, spkm.F).astype(np.float32))

    def test_mlp(self):
        N = self.N
        mask = np.ones(spkm.F, bool)
        mask[:5] = False
        net = N.MLPNet([16, 8], 3, True, mask).eval()
        norm = self._norm()
        outs = [("t_p50_ms", "identity", "-"), ("t_p95_ms", "identity", "-"), ("t_p99_ms", "identity", "-"),
                ("p_miss", "sigmoid", "-")]
        m = spkm.mlp(net.export_layers(), outs, norm=norm)
        X, H, y = random_frames(self.rng, 64, m)
        Xn = Dt.normalise(X, *norm)
        Q, L = net(torch.as_tensor(Xn))
        got = fastpred.predict_split(m, X, H, y, np.array([0, 64]))
        np.testing.assert_allclose(got[:, :3], Q.detach().numpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(got[:, 3], torch.sigmoid(L).detach().numpy(), rtol=1e-4, atol=1e-4)
        X2 = X.copy()
        X2[:, :5] += 100  # masked features have no effect
        np.testing.assert_allclose(fastpred.predict_split(m, X2, H, y, np.array([0, 64])), got, atol=1e-9)

    def test_gru_stream(self):
        N = self.N
        net = N.GRUNet(6, 8, 3, True).eval()
        norm, hnorm = self._norm(), (np.zeros(spkm.D, np.float32), np.ones(spkm.D, np.float32))
        outs = [("t_p50_ms", "identity", "-"), ("t_p95_ms", "identity", "-"), ("t_p99_ms", "identity", "-"),
                ("p_miss", "sigmoid", "-")]
        m = net.export(outs, norm, hnorm, 16.67, {})
        X, H, y = random_frames(self.rng, 40, m)
        sd = {"Xn": Dt.normalise(X, *norm), "Hn": Dt.normalise(H, *hnorm), "offsets": np.array([0, 25, 40])}
        Q, L = N.gru_stream_outputs(net, sd, torch.device("cpu"))
        ref = runner_outputs(m, X, H, y, sd["offsets"])
        np.testing.assert_allclose(ref[:, :3], Q, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(ref[:, 3], 1 / (1 + np.exp(-L)), rtol=1e-4, atol=1e-4)

    def test_tcn(self):
        N = self.N
        net = N.TCNNet(5, 3, [1, 2], 6, 8, 3, True).eval()
        norm, hnorm = self._norm(), (np.zeros(spkm.D, np.float32), np.ones(spkm.D, np.float32))
        outs = [("t_p50_ms", "identity", "-"), ("t_p95_ms", "identity", "-"), ("t_p99_ms", "identity", "-"),
                ("p_miss", "sigmoid", "-")]
        m = net.export(outs, norm, hnorm, 16.67, {})
        X, H, y = random_frames(self.rng, 30, m)
        off = np.array([0, 12, 30])
        Hn = torch.as_tensor(Dt.normalise(H, *hnorm))
        rs = torch.as_tensor(np.repeat(off[:-1], np.diff(off)))
        rows = torch.arange(30)
        Q, L = net(N.tcn_windows(Hn, rows, rs, net.L), torch.as_tensor(Dt.normalise(X, *norm)))
        ref = runner_outputs(m, X, H, y, off)
        np.testing.assert_allclose(ref[:, :3], Q.detach().numpy(), rtol=1e-4, atol=1e-4)


class TestResultRecords(unittest.TestCase):
    """common.clean(): every value the baselines and train.py put in a result record must serialise."""

    def test_clean_numpy_values(self):
        import json
        from pathlib import Path
        rec = {"a": np.zeros(3), "b": np.float32(1.5), "c": np.int64(3), "d": np.array([np.nan, 1.0, np.inf]),
               "e": np.bool_(True), "f": True, "g": [np.array([0.5, 0.25])], "h": Path("x/y"), "i": None,
               "j": np.float64("nan"), "k": (1, 2.5), "l": {"nested": np.arange(2, dtype=np.uint8)}}
        back = json.loads(json.dumps(common.clean(rec)))  # must not raise
        self.assertEqual(back["a"], [0.0, 0.0, 0.0])
        self.assertEqual(back["d"], [None, 1.0, None])
        self.assertIsInstance(back["c"], int)
        self.assertIs(back["e"], True)
        self.assertIs(back["f"], True)
        self.assertEqual(back["g"], [[0.5, 0.25]])
        self.assertEqual(back["h"], "x/y")
        self.assertIsNone(back["j"])
        self.assertEqual(back["l"]["nested"], [0, 1])

    def test_clean_rejects_unknown_types(self):
        with self.assertRaises(TypeError):
            common.clean({"x": object()})


class TestEmptySplit(unittest.TestCase):
    def test_empty_split_shapes(self):
        sd = Dt.SplitData("val", [], n_levers=4)
        self.assertEqual(sd.N, 0)
        self.assertEqual(sd.X.shape, (0, spkm.F))
        self.assertEqual(sd.hx.shape, (0, spkm.D - spkm.F))
        self.assertEqual(sd.H.shape, (0, spkm.D))
        self.assertEqual(sd.y.shape, (0,))
        self.assertEqual(sd.aux["q"].shape, (0, 4))
        self.assertEqual(sd.aux["t_steps"].shape, (0,))
        self.assertEqual(sd.offsets.tolist(), [0])
        self.assertEqual(len(sd.row_start), 0)


def _synthetic_manifest(n_train=(7, 15, 21, 21, 19), n_test=(2, 3, 4, 3, 8)):
    sess = []
    for k, (ntr, nte) in enumerate(zip(n_train, n_test)):
        for i in range(ntr):
            sess.append({"id": f"s{k}-tr{i:03d}", "split": "train", "stress": f"S{k}"})
        for i in range(nte):
            sess.append({"id": f"s{k}-te{i:03d}", "split": "test", "stress": f"S{k}"})
    return sess


class TestValidationCarve(unittest.TestCase):
    def test_carve_is_stratified_stable_and_leaves_test_alone(self):
        sess = _synthetic_manifest()
        out, info = Dt.carve_validation(sess)
        self.assertEqual(info["train_before"], 83)
        self.assertEqual(info["val"], sum(v["val"] for v in info["by_stress"].values()))
        self.assertAlmostEqual(info["val"] / info["train_before"], 0.15, delta=0.02)
        self.assertEqual([s for s in out if s["split"] == "test"], [s for s in sess if s["split"] == "test"])
        self.assertEqual(len({s["id"] for s in out}), len(sess))
        for st, v in info["by_stress"].items():  # every stress level with >= 2 train sessions has a val one
            self.assertGreaterEqual(v["val"], 1, st)
        out2, _ = Dt.carve_validation(list(reversed(sess)))  # stable: independent of the session order
        self.assertEqual(sorted(s["id"] for s in out if s["split"] == "val"),
                         sorted(s["id"] for s in out2 if s["split"] == "val"))
        self.assertTrue(all(s.get("carved_from") == "train" for s in out if s["split"] == "val"))

    def test_existing_val_is_kept(self):
        sess = [{"id": "a", "split": "train", "stress": "S1"}, {"id": "b", "split": "val", "stress": "S1"}]
        out, info = Dt.carve_validation(sess)
        self.assertIsNone(info)
        self.assertEqual(out, sess)


class TestNoFitGuard(unittest.TestCase):
    def test_guard_refuses_when_set(self):
        import trees as TR
        import linear as L
        with mock.patch.dict(os.environ, {common.NO_FIT_ENV: "1"}):
            self.assertTrue(common.no_fit())
            with self.assertRaises(RuntimeError):
                common.require_fitting("test")
            with self.assertRaises(RuntimeError):  # raised before lightgbm is imported
                TR.fit_lgb(np.zeros((4, 2)), np.zeros(4), np.zeros((2, 2)), np.zeros(2), "quantile", 0.5, {}, 1, 1, 1)
            with self.assertRaises(RuntimeError):
                L.rls_init(np.zeros((10, spkm.F), np.float32), np.zeros(10), np.array([0], np.uint32), 0.99)

    def test_freeze_online_for_dry_rls(self):
        m = spkm.random_model("rls", np.random.default_rng(11), "small")
        with mock.patch.dict(os.environ, {common.NO_FIT_ENV: "1"}):
            common.freeze_online(m)
        self.assertFalse(m.params["online"])

    def test_forward_only_rls_model(self):
        import linear as L
        n = 4
        m = L.rls_model(np.array([0, 1, 2], np.uint32), np.zeros(n), np.eye(n), (0.5, 0.95, 0.99), np.zeros(3), 1.0,
                        0.99, 0.005, 0.01, 1e6, -1, (np.zeros(spkm.F), np.ones(spkm.F)), 16.67, {}, online=False)
        self.assertFalse(m.params["online"])


class TestVersionRefusal(unittest.TestCase):
    def test_v1_manifest_and_config_refused(self):
        from pathlib import Path
        with self.assertRaises(SystemExit):
            Dt.check_supported_manifest(Path("data"), {"format_version": 1, "dataset": "laptop-v1"})
        Dt.check_supported_manifest(Path("data-v2"), {"format_version": 2, "dataset": "laptop-v2"})
        with self.assertRaises(SystemExit) as cm:
            common.load_config(str(common.CONFIG_DIR / "spark-ft-v1.json"))
        self.assertIn("NOT SUPPORTED", str(cm.exception))



if __name__ == "__main__":
    unittest.main(verbosity=2)
