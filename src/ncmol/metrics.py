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


def frac_ceiling(rmse, rmse_mean, rmse_floor) -> float:
    """ОСНОВНАЯ метрика: (RMSE_mean - RMSE)/(RMSE_mean - RMSE_floor). 0 = предсказание среднего,
    1 = потолок шума, >1 = подозрение на подгонку шума/утечку, <0 = хуже среднего."""
    den = rmse_mean - rmse_floor
    return (rmse_mean - rmse) / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Потолок под тестовую метку и поправка на аттенюацию (W6)
# ---------------------------------------------------------------------------
def rmse_floor(sigma, mean_inv_ndocs) -> float:
    """RMSE_floor = sigma * sqrt(mean(1/n_docs)) по тест-молекулам (метка = среднее по n_docs документам)."""
    return float(sigma) * float(np.sqrt(mean_inv_ndocs))


def reliability(var_y_test, noise_var) -> float:
    """Надёжность метки = доля дисперсии наблюдаемой метки теста, не связанная с шумом: 1 - noise/var_y."""
    return float(1.0 - noise_var / var_y_test) if var_y_test > 0 else float("nan")


def attenuation_corrected(rmse, rmse_mean, floor) -> dict:
    """Поправка на шумную тестовую метку (классическая модель: y_obs = t + e, e не коррелирует с ошибкой модели).
    Тогда MSE_obs = MSE_true + floor^2, и
        rmse_true      = sqrt(max(rmse^2      - floor^2, 0))
        rmse_mean_true = sqrt(max(rmse_mean^2 - floor^2, 0))
        frac_ceiling_true = (rmse_mean_true - rmse_true) / rmse_mean_true   (потолок в «истинных» единицах = 0).
    Допущения: гомоскедастичный шум, независимый от предсказаний; проверяется на данных, не доказано.
    Значения не обрезаются: rmse^2 < floor^2 (rmse_true = 0 по обрезке) означает переподгонку шума/утечку."""
    r2 = rmse**2 - floor**2
    m2 = rmse_mean**2 - floor**2
    rt = float(np.sqrt(max(r2, 0.0)))
    mt = float(np.sqrt(max(m2, 0.0)))
    return {"rmse_true": rt, "rmse_mean_true": mt,
            "frac_ceiling_true": (mt - rt) / mt if mt > 0 else float("nan"),
            "below_floor": bool(r2 < 0)}


def r2_corrected(mse, var_y_test, floor) -> float:
    """R² относительно «истинных» меток: 1 - (MSE - floor^2) / (Var(y_obs) - floor^2)."""
    den = var_y_test - floor**2
    return float(1.0 - (mse - floor**2) / den) if den > 0 else float("nan")


def disattenuated_corr(r_obs, rel) -> float:
    """r_true ≈ r_obs / sqrt(reliability) (Спирмен: приближение, т.к. формула выведена для Пирсона).
    Может превышать 1 при выборочной ошибке; не обрезается."""
    return float(r_obs / np.sqrt(rel)) if rel and rel > 0 else float("nan")
