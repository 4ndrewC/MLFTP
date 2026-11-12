"""Linear models: quantile regression (pinball), Lo et al.'s asymmetric squared loss, and the RLS
(online, in-engine) initialisation. Linear models export as the `linear` family (LINR): one
independent row per output, so monotone quantiles are not structural; training adds a
non-crossing penalty and every run reports its crossing rate.
"""
import numpy as np

import spkm


def rls_init(Xn, z, idx, lam, ridge=1e-3, q_levels=(0.5, 0.95, 0.99), base=None):
    """Offline start values for the engine's RLS (MODEL_CONTRACT section 5, family rls).

    Xn: normalised features (N, F) float32; z: target minus the base column (ms).
    w: ridge least squares on phi = [xn[idx], 1]. P: inverse of the information an RLS with forgetting
    lambda would hold, (sum over an effective window of 1/(1-lambda) frames of phi phi^T)^-1, so the online
    filter starts as confident as one window of data. q_offsets: residual quantiles; var0: residual variance.
    """
    from common import require_fitting
    require_fitting("linear.rls_init (ridge start values)")
    phi = np.concatenate([Xn[:, idx].astype(np.float64), np.ones((len(Xn), 1))], 1)
    A = phi.T @ phi
    n = A.shape[0]
    w = np.linalg.solve(A + ridge * len(phi) * np.eye(n), phi.T @ np.asarray(z, np.float64))
    e = np.asarray(z, np.float64) - phi @ w
    win = 1.0 / max(1e-6, 1.0 - lam)
    info = A / len(phi) * win
    P = np.linalg.inv(info + 1e-6 * np.eye(n))
    q_off = np.quantile(e, q_levels)
    return w, P, q_off, float(np.var(e))


def rls_model(idx, w, P, q_levels, q_off, var0, lam, gamma, var_alpha, max_trace, base_idx, norm, budget_ms,
              info, online=True):
    outs = [(f"t_p{int(round(q * 100))}_ms", "identity", "-") for q in q_levels] + [("p_miss", "identity", "-")]
    return spkm.rls(idx, w, outs, lam=lam, delta=1.0, max_trace=max_trace, online=online, P=P, base_idx=base_idx,
                    q_levels=q_levels, q_offsets=q_off, gamma=gamma, var0=var0, var_alpha=var_alpha, norm=norm,
                    budget_ms=budget_ms, info=info)


def lo_output_name(coverage):
    """Name the single Lo output by its validation coverage, using a name the engine controller can find
    (its fallback list is t_p99, t_p95, t_p90, t_mean, t_p50)."""
    if coverage >= 0.97:
        return "t_p99_ms"
    if coverage >= 0.925:
        return "t_p95_ms"
    if coverage >= 0.7:
        return "t_p90_ms"
    return "t_p50_ms"
