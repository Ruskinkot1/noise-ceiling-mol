#!/usr/bin/env python
"""Шаг 5. «Доля потолка» и сравнение с базовой линией ECFP+GBM.

Вход:
  results/benchmark.csv   — scripts/04 (task, split, seed, model, rmse, mae, r2, spearman, rmse_mean,
                            n_test, test_mean_inv_ndocs, ...)
  results/ceilings.csv    — scripts/03 (task, sigma, var_y, sigma_lo, sigma_hi, ...)
  results/predictions.csv|parquet (НЕОБЯЗАТЕЛЬНО; схема: task, split, seed, model, mol_id, pred
                            [+ y, n_docs, y_train_mean]) — нужен для ДИ по молекулам. Без него
                            выдаются только точечные оценки + разброс по сидам/задачам.

ОСНОВНАЯ метрика: frac_ceiling = (RMSE_mean - RMSE_model) / (RMSE_mean - RMSE_floor),
RMSE_floor = sigma*sqrt(mean(1/n_docs)) по тест-молекулам. 0 = среднее по train, 1 = потолок.
Вспомогательные: frac_rmse = RMSE_floor/RMSE, frac_r2 = R2/R2_max, frac_ceiling_true (поправка на аттенюацию).

Выход (results/):
  benchmark_with_ceiling.csv   — строки benchmark + потолок и все доли
  per_task_fraction.csv        — task x split x model: доля потолка (+ ДИ по молекулам, если есть предсказания)
  leaderboard.csv              — split x model: среднее по задачам и ДИ (по молекулам; по задачам)
  paired_vs_baseline.csv       — модель vs ECFP+GBM: Δ доли потолка, Уилкоксон по задачам,
                                 p с поправкой Холма (основная) и BH (чувствительность), парный бутстрэп
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ncmol import metrics, stats
from ncmol.models import BASELINE

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ["rmse", "r2", "frac_ceiling", "frac_rmse", "frac_r2", "frac_ceiling_true"]


def add_ceiling(b: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    cols = [x for x in ["task", "sigma", "var_y", "sigma_lo", "sigma_hi"] if x in c.columns]
    b = b.merge(c[cols], on="task")
    b["rmse_floor"] = b["sigma"] * np.sqrt(b["test_mean_inv_ndocs"])
    b["frac_ceiling"] = (b["rmse_mean"] - b["rmse"]) / (b["rmse_mean"] - b["rmse_floor"])
    b["frac_rmse"] = b["rmse_floor"] / b["rmse"]
    b["r2_max"] = (1 - b["rmse_floor"] ** 2 / b["var_y"]).clip(lower=0)
    b["frac_r2"] = b["r2"] / b["r2_max"]
    att = [metrics.attenuation_corrected(r, m, f) for r, m, f in zip(b["rmse"], b["rmse_mean"], b["rmse_floor"])]
    b["frac_ceiling_true"] = [a["frac_ceiling_true"] for a in att]
    b["below_floor"] = [a["below_floor"] for a in att]
    for tag in ("lo", "hi"):  # чувствительность к неопределённости sigma (границы бутстрэпа из scripts/03)
        if f"sigma_{tag}" in b.columns:
            fl = b[f"sigma_{tag}"] * np.sqrt(b["test_mean_inv_ndocs"])
            b[f"frac_ceiling_at_sigma_{tag}"] = (b["rmse_mean"] - b["rmse"]) / (b["rmse_mean"] - fl)
    return b


def bootstrap_tables(pred: pd.DataFrame, ceil: pd.DataFrame, n_boot: int, alpha: float, seed: int):
    """Возвращает (per_task DataFrame, draws[(split,model)] -> (n_tasks, B), diffs[(split,model)])."""
    sig = ceil.set_index("task")["sigma"]
    rows, draws, ddraws = [], {}, {}
    for (t, sp), g in pred.groupby(["task", "split"]):
        if t not in sig.index:
            continue
        models = sorted(g["model"].unique(), key=lambda m: (m != BASELINE, m))
        base, P, models = stats.wide_by_model(g, models)
        if BASELINE not in models or len(base) < 10:
            continue
        y, ytm, nd = base["y"].to_numpy(float), base["y_train_mean"].to_numpy(float), base["n_docs"].to_numpy(float)
        s = float(sig[t])
        est = stats.frac_ceiling_multi(np.arange(len(y)), y, P, ytm, nd, s)
        dr = stats.cluster_bootstrap(base["mol_id"].to_numpy(), lambda i: stats.frac_ceiling_multi(i, y, P, ytm, nd, s),
                                     n_boot, seed)
        lo, hi = stats.percentile_ci(dr, alpha)
        b0 = models.index(BASELINE)
        for k, m in enumerate(models):
            rows.append({"task": t, "split": sp, "model": m, "n_test_rows": len(y), "n_mols": base["mol_id"].nunique(),
                         "frac_ceiling_pooled": est[k], "frac_ceiling_lo": lo[k], "frac_ceiling_hi": hi[k]})
            draws.setdefault((sp, m), {})[t] = dr[:, k]
            if m != BASELINE:
                ddraws.setdefault((sp, m), {})[t] = (est[k] - est[b0], dr[:, k] - dr[:, b0])
    return pd.DataFrame(rows), draws, ddraws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--processed", default=str(ROOT / "data/processed"))
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-tasks", type=int, default=6, help="минимум задач для Уилкоксона")
    ap.add_argument("--correction", choices=["holm", "bh"], default="holm", help="какая поправка идёт в столбец p_adj")
    a = ap.parse_args()
    r = Path(a.results)
    c = pd.read_csv(r / "ceilings.csv")
    b = add_ceiling(pd.read_csv(r / "benchmark.csv"), c)
    b.to_csv(r / "benchmark_with_ceiling.csv", index=False)

    cols = [x for x in SUMMARY if x in b.columns]
    per_task = b.groupby(["task", "split", "model"], as_index=False)[cols].mean()
    pred = stats.load_predictions(r, a.processed)
    boot = pd.DataFrame()
    draws = ddraws = {}
    if pred is None:
        print("results/predictions.* не найден: ДИ по молекулам не считаются (схема — docs/experiment_design.md).")
    else:
        boot, draws, ddraws = bootstrap_tables(pred, c, a.n_boot, a.alpha, a.seed)
        per_task = per_task.merge(boot, on=["task", "split", "model"], how="left")
    per_task.to_csv(r / "per_task_fraction.csv", index=False)

    # ---- лидерборд: среднее по задачам (равный вес задач)
    lb_rows = []
    rng = np.random.default_rng(a.seed + 7)
    for (sp, m), g in per_task.groupby(["split", "model"]):
        row = {"split": sp, "model": m, "n_tasks": g["task"].nunique(), **{k: g[k].mean() for k in cols}}
        # разброс по задачам: бутстрэп задач (учёт того, что набор задач — тоже выборка)
        v = g["frac_ceiling"].to_numpy()
        if len(v) >= 2:
            bt = rng.choice(v, size=(a.n_boot, len(v)), replace=True).mean(1)
            row["frac_ceiling_task_lo"], row["frac_ceiling_task_hi"] = np.percentile(bt, [100 * a.alpha / 2, 100 * (1 - a.alpha / 2)])
        for tag in ("lo", "hi"):
            if f"frac_ceiling_at_sigma_{tag}" in b.columns:
                row[f"frac_ceiling_at_sigma_{tag}"] = b[(b.split == sp) & (b.model == m)].groupby("task")[f"frac_ceiling_at_sigma_{tag}"].mean().mean()
        if "frac_ceiling_pooled" in g.columns:
            row["frac_ceiling_pooled"] = g["frac_ceiling_pooled"].mean()  # оценка, к которой относятся ДИ по молекулам
        d = draws.get((sp, m))
        if d:
            mat = np.vstack(list(d.values()))  # (n_tasks, B): ресэмплинг молекул внутри каждой задачи независим
            mm = mat.mean(0)
            row["frac_ceiling_mol_lo"], row["frac_ceiling_mol_hi"] = stats.percentile_ci(mm, a.alpha)
        lb_rows.append(row)
    lb = pd.DataFrame(lb_rows).sort_values(["split", "frac_ceiling"], ascending=[True, False])
    lb.to_csv(r / "leaderboard.csv", index=False)
    print(lb.to_string(index=False))

    # ---- парное сравнение с базовой линией: Уилкоксон по задачам + парный бутстрэп
    pv = per_task.pivot_table(index=["split", "task"], columns="model", values=["frac_ceiling", "rmse"])
    prow = []
    for (sp, m), g in per_task.groupby(["split", "model"]):
        if m == BASELINE:
            continue
        try:
            dfc = (pv["frac_ceiling"][m] - pv["frac_ceiling"][BASELINE]).xs(sp, level="split").dropna()
            drm = (pv["rmse"][m] - pv["rmse"][BASELINE]).xs(sp, level="split").dropna()
        except KeyError:
            continue
        w = stats.wilcoxon_tasks(dfc.to_numpy(), a.min_tasks)
        row = {"split": sp, "model": m, "n_tasks": w["n_tasks"], "mean_diff_frac_ceiling": w["mean_diff"],
               "median_diff_frac_ceiling": w["median_diff"], "n_better": int((dfc > 0).sum()),
               "mean_diff_rmse": float(drm.mean()), "wilcoxon_p": w["p"]}
        dd = ddraws.get((sp, m))
        if dd:
            mat = np.vstack([v[1] for v in dd.values()]).mean(0)
            row["diff_mol_lo"], row["diff_mol_hi"] = stats.percentile_ci(mat, a.alpha)
            row["inside_noise_mol"] = bool(row["diff_mol_lo"] <= 0 <= row["diff_mol_hi"])
        prow.append(row)
    pr = pd.DataFrame(prow)
    if len(pr):
        pr["p_holm"] = stats.holm(pr["wilcoxon_p"])
        pr["p_bh"] = stats.benjamini_hochberg(pr["wilcoxon_p"])
        pr["p_adj"] = pr["p_holm" if a.correction == "holm" else "p_bh"]
        pr["inside_noise_tasks"] = ~(pr["p_adj"] < a.alpha)  # NaN -> внутри шума (недостаточно задач/нет эффекта)
    pr.to_csv(r / "paired_vs_baseline.csv", index=False)
    print(pr.to_string(index=False))


if __name__ == "__main__":
    main()
