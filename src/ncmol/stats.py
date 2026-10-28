"""Model comparison statistics (W6): cluster bootstrap over molecules, paired comparisons,
multiplicity corrections, MDE, activity cliffs.

The resampling unit is the MOLECULE (cluster): the same molecule appears in the test set of several seeds
and in several models, so rows are not independent. All functions take vectors of equal length
(one row = a molecule in the test set of one seed) and a vector of molecule identifiers `mol_ids`.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm, rankdata, wilcoxon

from .metrics import frac_ceiling

# --------------------------------------------------------------------------------------------
# Cluster bootstrap
# --------------------------------------------------------------------------------------------


def cluster_index(cluster_ids) -> list[np.ndarray]:
    """List of row indices per cluster (cluster order = by first appearance)."""
    ids = np.asarray(cluster_ids)
    _, inv = np.unique(ids, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def cluster_bootstrap(cluster_ids, stat_fn: Callable[[np.ndarray], object], n_boot: int = 1000,
                      seed: int = 0) -> np.ndarray:
    """Cluster bootstrap with replacement. stat_fn(idx) -> number or vector; idx = row indices of the
    sample (clusters concatenated; repeated clusters are included repeatedly). Returns (n_boot, k).
    The same seed with the same cluster_ids gives the same resamples -> pairing."""
    groups = cluster_index(cluster_ids)
    g = len(groups)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, g, size=g)
        idx = np.concatenate([groups[k] for k in pick])
        out.append(np.atleast_1d(np.asarray(stat_fn(idx), float)))
    return np.vstack(out)


def percentile_ci(draws, alpha: float = 0.05):
    """Percentile (1-alpha) CI over the columns of draws. NaNs are ignored."""
    d = np.asarray(draws, float)
    lo, hi = np.nanpercentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return lo, hi


def _spearman(y, p):
    if len(y) < 3:
        return np.nan
    ry, rp = rankdata(y), rankdata(p)
    if ry.std() == 0 or rp.std() == 0:
        return np.nan
    return float(np.corrcoef(ry, rp)[0, 1])


def _metrics_vec(y, p):
    e = y - p
    mse = float(np.mean(e**2))
    v = float(np.var(y))
    return np.array([mse**0.5, float(np.mean(np.abs(e))), 1 - mse / v if v > 0 else np.nan, _spearman(y, p)])


METRIC_NAMES = ["rmse", "mae", "r2", "spearman"]


def metrics_ci(y, pred, mol_ids, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """RMSE, MAE, R², Spearman with a cluster (by molecule) CI. Returns {name: (estimate, lo, hi)}."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    est = _metrics_vec(y, pred)
    draws = cluster_bootstrap(mol_ids, lambda i: _metrics_vec(y[i], pred[i]), n_boot, seed)
    lo, hi = percentile_ci(draws, alpha)
    return {n: (float(est[k]), float(lo[k]), float(hi[k])) for k, n in enumerate(METRIC_NAMES)}


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def frac_ceiling_stat(y, pred, y_train_mean, n_docs, sigma, idx=None) -> float:
    """Fraction of the ceiling over rows idx: rmse_mean is the RMSE of the "train mean" prediction (y_train_mean
    for each row is the train mean of its own seed); RMSE_floor = sigma*sqrt(mean(1/n_docs))."""
    if idx is not None:
        y, pred, y_train_mean, n_docs = y[idx], pred[idx], y_train_mean[idx], n_docs[idx]
    floor = float(sigma) * float(np.sqrt(np.mean(1.0 / n_docs)))
    return frac_ceiling(_rmse(y, pred), _rmse(y, y_train_mean), floor)


def frac_ceiling_ci(y, pred, y_train_mean, n_docs, mol_ids, sigma, sigma_draws=None, n_boot: int = 1000,
                    alpha: float = 0.05, seed: int = 0):
    """Fraction of the ceiling with a cluster CI. sigma is fixed; if sigma_draws is given (a bootstrap sample of sigma from
    scripts/03), a random sigma is taken at each iteration, and the CI accounts for the uncertainty of the ceiling.
    Returns (estimate, lo, hi, draws)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ytm, nd = np.asarray(y_train_mean, float), np.asarray(n_docs, float)
    est = frac_ceiling_stat(y, pred, ytm, nd, sigma)
    rng = np.random.default_rng(seed + 1)
    sd = None if sigma_draws is None else np.asarray(sigma_draws, float)

    def f(i):
        s = sigma if sd is None else sd[rng.integers(len(sd))]
        return frac_ceiling_stat(y, pred, ytm, nd, s, i)

    draws = cluster_bootstrap(mol_ids, f, n_boot, seed)[:, 0]
    lo, hi = percentile_ci(draws, alpha)
    return est, float(lo), float(hi), draws


def paired_diff_ci(stat_a: Callable[[np.ndarray], float], stat_b: Callable[[np.ndarray], float], mol_ids,
                   n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired cluster bootstrap of the difference stat_a - stat_b: both are computed on the SAME resamples
    of molecules. Returns diff, lo, hi, p_boot (two-sided fraction of resamples on the other side of zero),
    inside_noise (the CI contains 0)."""
    n = len(np.asarray(mol_ids))
    full = np.arange(n)
    est = float(stat_a(full) - stat_b(full))
    draws = cluster_bootstrap(mol_ids, lambda i: stat_a(i) - stat_b(i), n_boot, seed)[:, 0]
    draws = draws[np.isfinite(draws)]
    lo, hi = percentile_ci(draws, alpha)
    p = 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)) if len(draws) else np.nan
    return {"diff": est, "lo": float(lo), "hi": float(hi), "p_boot": float(min(p, 1.0)),
            "inside_noise": bool(lo <= 0 <= hi), "draws": draws}


# --------------------------------------------------------------------------------------------
# Per-task comparison: Wilcoxon + multiplicity corrections
# --------------------------------------------------------------------------------------------


def wilcoxon_tasks(delta, min_tasks: int = 6) -> dict:
    """Wilcoxon signed-rank test over tasks for differences delta (model - baseline). Two-sided.
    For n < min_tasks or all zeros, p = NaN: at n=5 the minimum two-sided p = 0.0625 > 0.05,
    so significance is unattainable and the test is not reported."""
    d = np.asarray(delta, float)
    d = d[np.isfinite(d)]
    out = {"n_tasks": int(len(d)), "mean_diff": float(d.mean()) if len(d) else np.nan,
           "median_diff": float(np.median(d)) if len(d) else np.nan, "n_negative": int((d < 0).sum()),
           "p": np.nan}
    if len(d) >= min_tasks and np.any(d != 0):
        out["p"] = float(wilcoxon(d).pvalue)
    return out


def holm(pvals) -> np.ndarray:
    """Holm correction (FWER control). NaNs are preserved and do not count toward the number of comparisons."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg correction (FDR control). NaNs are preserved."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    adj = np.minimum.accumulate((m / (np.arange(m) + 1) * p[ok][order])[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


# --------------------------------------------------------------------------------------------
# MDE: minimum detectable difference
# --------------------------------------------------------------------------------------------


def mde_from_se(se, alpha: float = 0.05, power: float = 0.8):
    """MDE = (z_{1-alpha/2} + z_power) * SE of the difference (normal approximation)."""
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.asarray(se, float)


def var_sqerr_diff(v_err, rho, noise_var) -> float:
    """Variance of d_i = e_a,i^2 - e_b,i^2 under H0 (equal models) in a Gaussian model.
    e_k = u_k + eps; Var(u_a)=Var(u_b)=v_err, corr(u_a,u_b)=rho, eps is the shared label noise, Var(eps)=noise_var.
    d = (u_a - u_b)(u_a + u_b + 2 eps); the factors are independent -> Var(d) = Var(D) Var(S) =
    2 v (1 - rho) * (2 v (1 + rho) + 4 noise_var)."""
    return 2 * v_err * (1 - rho) * (2 * v_err * (1 + rho) + 4 * noise_var)


def mde_rmse_gaussian(n_test, rmse_floor, v_err, rho=0.8, alpha: float = 0.05, power: float = 0.8):
    """MDE of the RMSE difference between two models as a function of the number of test molecules and label noise (rmse_floor).
    v_err is the variance of the model's "true" error (RMSE_true^2); rho is the correlation of the two models' errors
    (usually high for ECFP models, but here it is a PARAMETER, not a measurement; estimate it from predictions.csv).
    Delta method: dRMSE ≈ dMSE / (2 RMSE_obs), RMSE_obs^2 = v_err + floor^2.
    Model-based estimate under the assumptions of Gaussianity and independence of noise and errors."""
    noise_var = np.asarray(rmse_floor, float) ** 2
    var_d = var_sqerr_diff(v_err, rho, noise_var)
    se_mse = np.sqrt(var_d / np.asarray(n_test, float))
    rmse_obs = np.sqrt(v_err + noise_var)
    return mde_from_se(se_mse, alpha, power) / (2 * rmse_obs)


def mde_frac_units(mde_rmse, rmse_mean, rmse_floor):
    """Convert the RMSE MDE into units of fraction of the ceiling: divide by (RMSE_mean - RMSE_floor)."""
    return np.asarray(mde_rmse, float) / (np.asarray(rmse_mean, float) - np.asarray(rmse_floor, float))


# --------------------------------------------------------------------------------------------
# Similarity and activity cliffs
# --------------------------------------------------------------------------------------------


def tanimoto_matrix(A, B) -> np.ndarray:
    """Tanimoto for binary fingerprints (rows). float32 (|A|,|B|)."""
    A = (np.asarray(A) > 0).astype(np.float32)
    B = (np.asarray(B) > 0).astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(union > 0, inter / union, 0.0)
    return s.astype(np.float32)


def max_tanimoto_to_train(fp_test, fp_train, chunk: int = 512) -> np.ndarray:
    """For each test molecule, the maximum Tanimoto to train."""
    out = np.empty(len(fp_test), np.float32)
    for s in range(0, len(fp_test), chunk):
        out[s:s + chunk] = tanimoto_matrix(fp_test[s:s + chunk], fp_train).max(axis=1)
    return out


def cliff_pairs(fp, y, sim_min: float = 0.8, delta_min: float = 1.5) -> np.ndarray:
    """Pairs (i<j) within the set: Tanimoto(ECFP4) >= sim_min and |y_i - y_j| >= delta_min.
    Thresholds are PARAMETERS (defaults 0.8 and 1.5 log units are nominal, not tuned). Returns (k,2)."""
    y = np.asarray(y, float)
    S = tanimoto_matrix(fp, fp)
    D = np.abs(y[:, None] - y[None, :])
    m = np.triu((S >= sim_min) & (D >= delta_min), k=1)
    return np.argwhere(m)


def cliff_mask_vs_train(fp_test, y_test, fp_train, y_train, sim_min: float = 0.8, delta_min: float = 1.5,
                        chunk: int = 512) -> np.ndarray:
    """Boolean vector over test molecules: there is a neighbour in train with Tanimoto >= sim_min and |Δy| >= delta_min
    ("cliff molecules": a model relying on the neighbour errs systematically)."""
    y_test, y_train = np.asarray(y_test, float), np.asarray(y_train, float)
    out = np.zeros(len(fp_test), bool)
    for s in range(0, len(fp_test), chunk):
        S = tanimoto_matrix(fp_test[s:s + chunk], fp_train)
        D = np.abs(y_test[s:s + chunk, None] - y_train[None, :])
        out[s:s + chunk] = ((S >= sim_min) & (D >= delta_min)).any(axis=1)
    return out


def cliff_pair_metrics(pairs, y, pred) -> dict:
    """Metrics on pairs: whether the sign of Δ is predicted correctly (sign_acc) and the RMSE of the difference Δ (rmse_delta)."""
    if len(pairs) == 0:
        return {"n_pairs": 0, "sign_acc": np.nan, "rmse_delta": np.nan}
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    i, j = pairs[:, 0], pairs[:, 1]
    dy, dp = y[i] - y[j], pred[i] - pred[j]
    return {"n_pairs": int(len(pairs)), "sign_acc": float(np.mean(np.sign(dy) == np.sign(dp))),
            "rmse_delta": float(np.sqrt(np.mean((dy - dp) ** 2)))}


def p_noise_exceeds(delta, sigma, n_i=1, n_j=1) -> float:
    """P(|label difference noise| >= delta) for a pair of molecules, noise N(0, sigma^2 (1/n_i + 1/n_j)).
    How many "cliffs" are expected from noise alone; the Gaussian assumption is an approximation."""
    s = float(sigma) * np.sqrt(1.0 / np.asarray(n_i, float) + 1.0 / np.asarray(n_j, float))
    return 2 * norm.sf(float(delta) / s)


def slice_flag(lo: float, hi: float, lower_is_better: bool = True) -> str:
    """Label for a slice by the CI of the difference (model - baseline): "inside noise" if the CI contains 0.
    lower_is_better=True for errors (RMSE: difference < 0 = model is better); False for fraction of the ceiling."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "no data"
    if lo <= 0 <= hi:
        return "within noise"
    better = (hi < 0) if lower_is_better else (lo > 0)
    return "model better" if better else "model worse"


# --------------------------------------------------------------------------------------------
# Loading predictions (schema of results/predictions.csv, see docs/experiment_design.md)
# --------------------------------------------------------------------------------------------
PRED_REQUIRED = ["task", "split", "seed", "model", "mol_id", "pred"]


def load_predictions(results_dir, processed_dir):
    """Reads results/predictions.(parquet|csv): task, split, seed, model, mol_id, pred
    [+ y, n_docs, y_train_mean -- if absent, they are reconstructed]. y and n_docs are taken from
    <processed>/<task>.molecules.csv; y_train_mean (mean y over the train of its split and seed)
    is recomputed with the same split (ncmol.splits; splits are deterministic by seed) -- requires
    molecules.csv to be the same file that scripts/04 read. If the file is missing -> None."""
    from pathlib import Path

    import pandas as pd

    from .splits import SPLITS

    r, pdir = Path(results_dir), Path(processed_dir)
    pq, csv = r / "predictions.parquet", r / "predictions.csv"
    if pq.exists():
        try:
            p = pd.read_parquet(pq)
        except Exception:  # no pyarrow/fastparquet
            p = pd.read_csv(csv) if csv.exists() else None
    elif csv.exists():
        p = pd.read_csv(csv)
    else:
        return None
    if p is None:
        return None
    miss = [c for c in PRED_REQUIRED if c not in p.columns]
    if miss:
        raise ValueError(f"predictions: missing columns {miss}")
    parts = []
    for t, g in p.groupby("task"):
        mol = pd.read_csv(pdir / f"{t}.molecules.csv")
        g = g.copy()
        for c in ("y", "n_docs"):
            if c not in g.columns:
                g[c] = g["mol_id"].map(mol.set_index("mol_id")[c])
        if "y_train_mean" not in g.columns:
            ytm = {}
            for sp, seed in g[["split", "seed"]].drop_duplicates().itertuples(index=False):
                tr, _ = SPLITS[sp](mol, int(seed))
                ytm[(sp, seed)] = float(mol["y"].iloc[tr].mean())
            g["y_train_mean"] = [ytm[(a, b)] for a, b in zip(g["split"], g["seed"])]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def wide_by_model(g, models):
    """g is the predictions of one (task, split). Returns (base_df, P) where base_df has columns
    seed, mol_id, y, n_docs, y_train_mean (one row = a molecule in the test set of a seed), P is (n, len(models))
    predictions; the intersection of (seed, mol_id) keys across all models is used."""
    piv = g.pivot_table(index=["seed", "mol_id"], columns="model", values="pred", aggfunc="first")
    models = [m for m in models if m in piv.columns]
    piv = piv[models].dropna()
    base = (g.drop_duplicates(["seed", "mol_id"]).set_index(["seed", "mol_id"])
            .loc[piv.index, ["y", "n_docs", "y_train_mean"]].reset_index())
    return base, piv.to_numpy(float), models


def frac_ceiling_multi(idx, y, P, ytm, nd, sigma) -> np.ndarray:
    """Fraction of the ceiling for all columns of P over rows idx (the same floor and rmse_mean)."""
    y, P, ytm, nd = y[idx], P[idx], ytm[idx], nd[idx]
    floor = float(sigma) * float(np.sqrt(np.mean(1.0 / nd)))
    rm = _rmse(y, ytm)
    rmse = np.sqrt(np.mean((y[:, None] - P) ** 2, axis=0))
    den = rm - floor
    return (rm - rmse) / den if den > 0 else np.full(P.shape[1], np.nan)
