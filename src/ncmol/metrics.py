import numpy as np
from scipy.stats import spearmanr


def regression_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    mse = float(np.mean((y - p) ** 2))
    return {"rmse": mse**0.5, "mae": float(np.mean(np.abs(y - p))),
            "r2": float(1 - mse / np.var(y)) if np.var(y) > 0 else np.nan,
            "spearman": float(spearmanr(y, p)[0])}


def fraction_of_ceiling(rmse, r2, rmse_floor, r2_max) -> dict:
    """frac_rmse = rmse_floor/rmse (1 = at the ceiling; >1 suggests noise fitting/leakage);
    frac_r2 = r2/r2_max."""
    return {"frac_rmse": rmse_floor / rmse if rmse > 0 else np.nan,
            "frac_r2": r2 / r2_max if r2_max and r2_max > 0 else np.nan}


def frac_ceiling(rmse, rmse_mean, rmse_floor) -> float:
    """PRIMARY metric: (RMSE_mean - RMSE)/(RMSE_mean - RMSE_floor). 0 = predicting the mean,
    1 = noise ceiling, >1 = suspected noise fitting/leakage, <0 = worse than the mean."""
    den = rmse_mean - rmse_floor
    return (rmse_mean - rmse) / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Ceiling for the test label and attenuation correction (W6)
# ---------------------------------------------------------------------------
def rmse_floor(sigma, mean_inv_ndocs) -> float:
    """RMSE_floor = sigma * sqrt(mean(1/n_docs)) over test molecules (label = mean over n_docs documents)."""
    return float(sigma) * float(np.sqrt(mean_inv_ndocs))


def reliability(var_y_test, noise_var) -> float:
    """Label reliability = fraction of the observed test-label variance not due to noise: 1 - noise/var_y."""
    return float(1.0 - noise_var / var_y_test) if var_y_test > 0 else float("nan")


def attenuation_corrected(rmse, rmse_mean, floor) -> dict:
    """Correction for a noisy test label (classical model: y_obs = t + e, e uncorrelated with the model error).
    Then MSE_obs = MSE_true + floor^2, and
        rmse_true      = sqrt(max(rmse^2      - floor^2, 0))
        rmse_mean_true = sqrt(max(rmse_mean^2 - floor^2, 0))
        frac_ceiling_true = (rmse_mean_true - rmse_true) / rmse_mean_true   (ceiling in "true" units = 0).
    Assumptions: homoscedastic noise independent of predictions; checked on data, not proven.
    Values are not clipped: rmse^2 < floor^2 (rmse_true = 0 after clipping) indicates noise overfitting/leakage."""
    r2 = rmse**2 - floor**2
    m2 = rmse_mean**2 - floor**2
    rt = float(np.sqrt(max(r2, 0.0)))
    mt = float(np.sqrt(max(m2, 0.0)))
    return {"rmse_true": rt, "rmse_mean_true": mt,
            "frac_ceiling_true": (mt - rt) / mt if mt > 0 else float("nan"),
            "below_floor": bool(r2 < 0)}


def r2_corrected(mse, var_y_test, floor) -> float:
    """R² relative to the "true" labels: 1 - (MSE - floor^2) / (Var(y_obs) - floor^2)."""
    den = var_y_test - floor**2
    return float(1.0 - (mse - floor**2) / den) if den > 0 else float("nan")


def disattenuated_corr(r_obs, rel) -> float:
    """r_true ≈ r_obs / sqrt(reliability) (Spearman: an approximation, since the formula is derived for Pearson).
    May exceed 1 due to sampling error; not clipped."""
    return float(r_obs / np.sqrt(rel)) if rel and rel > 0 else float("nan")
