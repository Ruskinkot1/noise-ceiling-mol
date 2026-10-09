"""Потолок шума по повторным измерениям из РАЗНЫХ документов.

Модель: y_ij = t_i + e_ij, e_ij ~ (0, s2) — шум «между документами» (лаборатория, условия анализа,
ошибки записи). s2 оценивается как объединённая внутримолекулярная дисперсия:
    s2 = sum_i sum_j (y_ij - ybar_i)^2 / sum_i (n_i - 1).
Это оценка ВСЕЙ межисточниковой изменчивости, а не «чистой» ошибки прибора; предполагается
гомоскедастичность и что реплицированные молекулы типичны для набора (оба допущения проверяются
в scripts/03 и описываются в статье как ограничения).

Если метка для моделирования = среднее по n_i документам, то её дисперсия шума s2/n_i, и
    RMSE_floor = sqrt(s2 * mean(1/n_i)),   R2_max = 1 - s2*mean(1/n_i) / Var(y).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Ceiling:
    n_mols_rep: int          # молекул с >= min_docs документами
    n_meas_rep: int
    sigma: float             # sqrt(s2), единицы pChEMBL
    var_y: float             # дисперсия меток по всему набору
    rmse_floor: float
    r2_max: float
    mae_floor: float

    def to_dict(self):
        return asdict(self)


def doc_level(df: pd.DataFrame) -> pd.DataFrame:
    """Одно значение на пару (молекула, документ): среднее внутри документа.
    Столбцы: mol_id, doc_id, y."""
    return df.groupby(["mol_id", "doc_id"], as_index=False)["y"].mean()


def pooled_variance(d: pd.DataFrame, min_docs: int = 2, max_abs_pair_diff: float | None = None):
    """Возвращает (s2, n_mols, n_meas). d — результат doc_level."""
    g = d.groupby("mol_id")["y"]
    n = g.transform("size")
    d = d[n >= min_docs]
    if max_abs_pair_diff is not None and len(d):
        rng = d.groupby("mol_id")["y"].transform(lambda s: s.max() - s.min())
        d = d[rng <= max_abs_pair_diff]
    if d.empty:
        return np.nan, 0, 0
    dev = d["y"] - d.groupby("mol_id")["y"].transform("mean")
    dof = (d.groupby("mol_id").size() - 1).sum()
    return float((dev**2).sum() / dof), int(d["mol_id"].nunique()), int(len(d))


def label_noise_factor(d: pd.DataFrame) -> float:
    """mean(1/n_i) по молекулам набора (для метки = среднее по документам)."""
    n = d.groupby("mol_id").size()
    return float((1.0 / n).mean())


def estimate_ceiling(df: pd.DataFrame, min_docs: int = 2, max_abs_pair_diff: float | None = 3.0,
                     mean_label: bool = True) -> Ceiling:
    """df: столбцы mol_id, doc_id, y (сырые измерения)."""
    d = doc_level(df)
    s2, n_mols, n_meas = pooled_variance(d, min_docs, max_abs_pair_diff)
    mols = d.groupby("mol_id")["y"].mean()
    var_y = float(mols.var(ddof=1)) if mean_label else float(d["y"].var(ddof=1))
    k = label_noise_factor(d) if mean_label else 1.0
    s2_eff = s2 * k
    return Ceiling(
        n_mols_rep=n_mols, n_meas_rep=n_meas, sigma=float(np.sqrt(s2)), var_y=var_y,
        rmse_floor=float(np.sqrt(s2_eff)),
        r2_max=float(np.clip(1 - s2_eff / var_y, 0, 1)),
        mae_floor=float(np.sqrt(s2_eff) * np.sqrt(2 / np.pi)),
    )


def bootstrap_ceiling(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, min_docs: int = 2,
                      max_abs_pair_diff: float | None = 3.0, mean_label: bool = True) -> pd.DataFrame:
    """Бутстрэп по МОЛЕКУЛАМ (кластерный), векторный. Совпадает с estimate_ceiling на каждой выборке.
    Возвращает перцентили 2.5/50/97.5 для sigma, rmse_floor, r2_max."""
    d = doc_level(df)
    g = d.groupby("mol_id")["y"]
    mol = pd.DataFrame({"n": g.size(), "mean": g.mean(), "sum": g.sum(),
                        "sum2": (d["y"] ** 2).groupby(d["mol_id"]).sum(),
                        "rng": g.max() - g.min()})
    mol["ss"] = mol["sum2"] - mol["sum"] ** 2 / mol["n"]
    ok = (mol["n"] >= min_docs).to_numpy().copy()
    if max_abs_pair_diff is not None:
        ok = ok & (mol["rng"] <= max_abs_pair_diff).to_numpy()
    n, m, ss = mol["n"].to_numpy(float), mol["mean"].to_numpy(), mol["ss"].to_numpy()
    sy, sy2 = mol["sum"].to_numpy(), mol["sum2"].to_numpy()
    rng = np.random.default_rng(seed)
    k = len(mol)
    rows = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
        dof = (w * ok * (n - 1)).sum()
        s2 = (w * ok * ss).sum() / dof if dof > 0 else np.nan
        if mean_label:
            mu = (w * m).sum() / w.sum()
            var_y = (w * (m - mu) ** 2).sum() / (w.sum() - 1)
            kf = (w / n).sum() / w.sum()
        else:
            cnt = (w * n).sum()
            var_y = ((w * sy2).sum() - (w * sy).sum() ** 2 / cnt) / (cnt - 1)
            kf = 1.0
        s2e = s2 * kf
        rows.append((np.sqrt(s2), np.sqrt(s2e), float(np.clip(1 - s2e / var_y, 0, 1))))
    arr = np.array(rows)
    out = {}
    for j, name in enumerate(["sigma", "rmse_floor", "r2_max"]):
        lo, med, hi = np.nanpercentile(arr[:, j], [2.5, 50, 97.5])
        out.update({f"{name}_lo": lo, f"{name}_med": med, f"{name}_hi": hi})
    return pd.DataFrame([out])


def sensitivity(df: pd.DataFrame, min_docs_grid=(2, 3), diff_grid=(None, 3.0, 2.0, 1.0)) -> pd.DataFrame:
    """Устойчивость потолка к порогам фильтрации (min_docs, обрезка |Δ|)."""
    rows = []
    for md in min_docs_grid:
        for dg in diff_grid:
            c = estimate_ceiling(df, min_docs=md, max_abs_pair_diff=dg)
            rows.append({"min_docs": md, "max_abs_pair_diff": dg, **c.to_dict()})
    return pd.DataFrame(rows)
