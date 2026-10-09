"""Статистика сравнения моделей (W6): кластерный бутстрэп по молекулам, парные сравнения,
поправки на множественность, MDE, activity cliffs.

Единица ресэмплинга — МОЛЕКУЛА (кластер): одна и та же молекула встречается в тесте нескольких сидов
и в нескольких моделях, строки не независимы. Все функции принимают векторы одинаковой длины
(по строке = молекула в тесте одного сида) и вектор идентификаторов молекул `mol_ids`.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm, rankdata, wilcoxon

from .metrics import frac_ceiling

# --------------------------------------------------------------------------------------------
# Кластерный бутстрэп
# --------------------------------------------------------------------------------------------


def cluster_index(cluster_ids) -> list[np.ndarray]:
    """Список индексов строк по кластерам (порядок кластеров — по первому появлению)."""
    ids = np.asarray(cluster_ids)
    _, inv = np.unique(ids, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def cluster_bootstrap(cluster_ids, stat_fn: Callable[[np.ndarray], object], n_boot: int = 1000,
                      seed: int = 0) -> np.ndarray:
    """Бутстрэп кластеров с возвращением. stat_fn(idx) -> число или вектор; idx — индексы строк
    выборки (кластеры склеены, повторные кластеры входят повторно). Возвращает (n_boot, k).
    Один и тот же seed при одном и том же cluster_ids даёт те же ресэмплы -> парность."""
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
    """Процентильный (1-alpha) ДИ по столбцам draws. NaN игнорируются."""
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
    """RMSE, MAE, R², Spearman с кластерным (по молекулам) ДИ. Возвращает {name: (оценка, lo, hi)}."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    est = _metrics_vec(y, pred)
    draws = cluster_bootstrap(mol_ids, lambda i: _metrics_vec(y[i], pred[i]), n_boot, seed)
    lo, hi = percentile_ci(draws, alpha)
    return {n: (float(est[k]), float(lo[k]), float(hi[k])) for k, n in enumerate(METRIC_NAMES)}


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def frac_ceiling_stat(y, pred, y_train_mean, n_docs, sigma, idx=None) -> float:
    """Доля потолка по строкам idx: rmse_mean — RMSE предсказания «среднее по train» (y_train_mean
    для каждой строки — среднее train своего сида); RMSE_floor = sigma*sqrt(mean(1/n_docs))."""
    if idx is not None:
        y, pred, y_train_mean, n_docs = y[idx], pred[idx], y_train_mean[idx], n_docs[idx]
    floor = float(sigma) * float(np.sqrt(np.mean(1.0 / n_docs)))
    return frac_ceiling(_rmse(y, pred), _rmse(y, y_train_mean), floor)


def frac_ceiling_ci(y, pred, y_train_mean, n_docs, mol_ids, sigma, sigma_draws=None, n_boot: int = 1000,
                    alpha: float = 0.05, seed: int = 0):
    """Доля потолка с кластерным ДИ. sigma фиксирована; если задан sigma_draws (бутстрэп-выборка sigma из
    scripts/03), на каждой итерации берётся случайная sigma, и ДИ учитывает неопределённость потолка.
    Возвращает (оценка, lo, hi, draws)."""
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
    """Парный кластерный бутстрэп разности stat_a - stat_b: оба считаются на ОДНИХ И ТЕХ ЖЕ ресэмплах
    молекул. Возвращает diff, lo, hi, p_boot (двусторонняя доля ресэмплов по другую сторону нуля),
    inside_noise (ДИ содержит 0)."""
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
# Сравнение по задачам: Уилкоксон + поправки на множественность
# --------------------------------------------------------------------------------------------


def wilcoxon_tasks(delta, min_tasks: int = 6) -> dict:
    """Знаковых рангов Уилкоксона по задачам для разностей delta (модель - базовая). Двусторонний.
    При n < min_tasks или всех нулях p = NaN: при n=5 минимальный двусторонний p = 0.0625 > 0.05,
    значимость недостижима, поэтому тест не выдаётся."""
    d = np.asarray(delta, float)
    d = d[np.isfinite(d)]
    out = {"n_tasks": int(len(d)), "mean_diff": float(d.mean()) if len(d) else np.nan,
           "median_diff": float(np.median(d)) if len(d) else np.nan, "n_negative": int((d < 0).sum()),
           "p": np.nan}
    if len(d) >= min_tasks and np.any(d != 0):
        out["p"] = float(wilcoxon(d).pvalue)
    return out


def holm(pvals) -> np.ndarray:
    """Поправка Холма (контроль FWER). NaN сохраняются и в число сравнений не входят."""
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
    """Поправка Бенджамини-Хохберга (контроль FDR). NaN сохраняются."""
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
# MDE: минимальная обнаружимая разность
# --------------------------------------------------------------------------------------------


def mde_from_se(se, alpha: float = 0.05, power: float = 0.8):
    """MDE = (z_{1-alpha/2} + z_power) * SE разности (нормальное приближение)."""
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.asarray(se, float)


def var_sqerr_diff(v_err, rho, noise_var) -> float:
    """Дисперсия d_i = e_a,i^2 - e_b,i^2 при H0 (равные модели) в гауссовой модели.
    e_k = u_k + eps; Var(u_a)=Var(u_b)=v_err, corr(u_a,u_b)=rho, eps — общий шум метки, Var(eps)=noise_var.
    d = (u_a - u_b)(u_a + u_b + 2 eps); множители независимы -> Var(d) = Var(D) Var(S) =
    2 v (1 - rho) * (2 v (1 + rho) + 4 noise_var)."""
    return 2 * v_err * (1 - rho) * (2 * v_err * (1 + rho) + 4 * noise_var)


def mde_rmse_gaussian(n_test, rmse_floor, v_err, rho=0.8, alpha: float = 0.05, power: float = 0.8):
    """MDE разности RMSE двух моделей как функция числа тест-молекул и шума метки (rmse_floor).
    v_err — дисперсия «истинной» ошибки модели (RMSE_true^2); rho — корреляция ошибок двух моделей
    (у ECFP-моделей обычно высокая, но здесь это ПАРАМЕТР, не измерение; оценить из predictions.csv).
    Дельта-метод: dRMSE ≈ dMSE / (2 RMSE_obs), RMSE_obs^2 = v_err + floor^2.
    Модельная оценка при допущениях гауссовости и независимости шума и ошибок."""
    noise_var = np.asarray(rmse_floor, float) ** 2
    var_d = var_sqerr_diff(v_err, rho, noise_var)
    se_mse = np.sqrt(var_d / np.asarray(n_test, float))
    rmse_obs = np.sqrt(v_err + noise_var)
    return mde_from_se(se_mse, alpha, power) / (2 * rmse_obs)


def mde_frac_units(mde_rmse, rmse_mean, rmse_floor):
    """Перевод MDE по RMSE в единицы доли потолка: делим на (RMSE_mean - RMSE_floor)."""
    return np.asarray(mde_rmse, float) / (np.asarray(rmse_mean, float) - np.asarray(rmse_floor, float))


# --------------------------------------------------------------------------------------------
# Сходство и activity cliffs
# --------------------------------------------------------------------------------------------


def tanimoto_matrix(A, B) -> np.ndarray:
    """Танимото для бинарных отпечатков (строки). float32 (|A|,|B|)."""
    A = (np.asarray(A) > 0).astype(np.float32)
    B = (np.asarray(B) > 0).astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(union > 0, inter / union, 0.0)
    return s.astype(np.float32)


def max_tanimoto_to_train(fp_test, fp_train, chunk: int = 512) -> np.ndarray:
    """Для каждой тест-молекулы — максимальное Танимото к train."""
    out = np.empty(len(fp_test), np.float32)
    for s in range(0, len(fp_test), chunk):
        out[s:s + chunk] = tanimoto_matrix(fp_test[s:s + chunk], fp_train).max(axis=1)
    return out


def cliff_pairs(fp, y, sim_min: float = 0.8, delta_min: float = 1.5) -> np.ndarray:
    """Пары (i<j) внутри набора: Tanimoto(ECFP4) >= sim_min и |y_i - y_j| >= delta_min.
    Пороги — ПАРАМЕТРЫ (умолчания 0.8 и 1.5 лог-единиц — условные, не подбирались). Возвращает (k,2)."""
    y = np.asarray(y, float)
    S = tanimoto_matrix(fp, fp)
    D = np.abs(y[:, None] - y[None, :])
    m = np.triu((S >= sim_min) & (D >= delta_min), k=1)
    return np.argwhere(m)


def cliff_mask_vs_train(fp_test, y_test, fp_train, y_train, sim_min: float = 0.8, delta_min: float = 1.5,
                        chunk: int = 512) -> np.ndarray:
    """Булев вектор по тест-молекулам: есть сосед в train с Tanimoto >= sim_min и |Δy| >= delta_min
    («cliff-молекулы»: модель, опираясь на соседа, ошибается систематически)."""
    y_test, y_train = np.asarray(y_test, float), np.asarray(y_train, float)
    out = np.zeros(len(fp_test), bool)
    for s in range(0, len(fp_test), chunk):
        S = tanimoto_matrix(fp_test[s:s + chunk], fp_train)
        D = np.abs(y_test[s:s + chunk, None] - y_train[None, :])
        out[s:s + chunk] = ((S >= sim_min) & (D >= delta_min)).any(axis=1)
    return out


def cliff_pair_metrics(pairs, y, pred) -> dict:
    """Метрики на парах: верно ли предсказан знак Δ (sign_acc) и RMSE разности Δ (rmse_delta)."""
    if len(pairs) == 0:
        return {"n_pairs": 0, "sign_acc": np.nan, "rmse_delta": np.nan}
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    i, j = pairs[:, 0], pairs[:, 1]
    dy, dp = y[i] - y[j], pred[i] - pred[j]
    return {"n_pairs": int(len(pairs)), "sign_acc": float(np.mean(np.sign(dy) == np.sign(dp))),
            "rmse_delta": float(np.sqrt(np.mean((dy - dp) ** 2)))}


def p_noise_exceeds(delta, sigma, n_i=1, n_j=1) -> float:
    """P(|шум разности меток| >= delta) для пары молекул, шум N(0, sigma^2 (1/n_i + 1/n_j)).
    Сколько «обрывов» ожидается из-за одного шума; допущение гауссовости — приближение."""
    s = float(sigma) * np.sqrt(1.0 / np.asarray(n_i, float) + 1.0 / np.asarray(n_j, float))
    return 2 * norm.sf(float(delta) / s)


def slice_flag(lo: float, hi: float, lower_is_better: bool = True) -> str:
    """Пометка разреза по ДИ разности (модель - базовая): «внутри шума», если ДИ содержит 0.
    lower_is_better=True для ошибок (RMSE: разность < 0 = модель лучше); False для доли потолка."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "нет данных"
    if lo <= 0 <= hi:
        return "внутри шума"
    better = (hi < 0) if lower_is_better else (lo > 0)
    return "модель лучше" if better else "модель хуже"
