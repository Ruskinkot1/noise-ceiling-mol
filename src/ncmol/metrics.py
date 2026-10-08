import numpy as np
from scipy.stats import spearmanr


def regression_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    mse = float(np.mean((y - p) ** 2))
    return {"rmse": mse**0.5, "mae": float(np.mean(np.abs(y - p))),
            "r2": float(1 - mse / np.var(y)) if np.var(y) > 0 else np.nan,
            "spearman": float(spearmanr(y, p)[0])}


def fraction_of_ceiling(rmse, r2, rmse_floor, r2_max) -> dict:
    """frac_rmse = rmse_floor/rmse (1 = на потолке; >1 — подозрение на подгонку шума/утечку);
    frac_r2 = r2/r2_max."""
    return {"frac_rmse": rmse_floor / rmse if rmse > 0 else np.nan,
            "frac_r2": r2 / r2_max if r2_max and r2_max > 0 else np.nan}
