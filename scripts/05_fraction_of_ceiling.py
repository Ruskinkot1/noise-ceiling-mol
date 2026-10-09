#!/usr/bin/env python
"""Шаг 5. «Доля потолка» и сравнение с базовой линией ECFP+GBM.
Потолок пересчитывается под метку каждого тест-набора: RMSE_floor = sigma * sqrt(mean(1/n_docs)).
ОСНОВНАЯ метрика: frac_ceiling = (RMSE_mean - RMSE_model) / (RMSE_mean - RMSE_floor)  (0 = среднее по train, 1 = потолок).
Вспомогательные: frac_rmse = RMSE_floor/RMSE, frac_r2 = R2/R2_max (см. docs/experiment_design.md).
Выход: results/leaderboard.csv (модель x сплит), results/paired_vs_baseline.csv (Уилкоксон по задачам)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ncmol.models import BASELINE

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    r = Path(a.results)
    b = pd.read_csv(r / "benchmark.csv")
    c = pd.read_csv(r / "ceilings.csv")[["task", "sigma", "var_y"]]
    b = b.merge(c, on="task")
    b["rmse_floor"] = b["sigma"] * np.sqrt(b["test_mean_inv_ndocs"])
    b["frac_ceiling"] = (b["rmse_mean"] - b["rmse"]) / (b["rmse_mean"] - b["rmse_floor"])
    b["frac_rmse"] = b["rmse_floor"] / b["rmse"]
    b["r2_max"] = (1 - b["rmse_floor"] ** 2 / b["var_y"]).clip(lower=0)
    b["frac_r2"] = b["r2"] / b["r2_max"]
    b.to_csv(r / "benchmark_with_ceiling.csv", index=False)

    per_task = b.groupby(["task", "split", "model"], as_index=False)[["rmse", "r2", "frac_ceiling", "frac_rmse", "frac_r2"]].mean()
    lb = per_task.groupby(["split", "model"], as_index=False)[["rmse", "r2", "frac_ceiling", "frac_rmse", "frac_r2"]].mean()
    lb.sort_values(["split", "frac_ceiling"], ascending=[True, False]).to_csv(r / "leaderboard.csv", index=False)
    print(lb.sort_values(["split", "frac_ceiling"], ascending=[True, False]).to_string(index=False))

    rows = []
    for (sp, m), g in per_task.groupby(["split", "model"]):
        if m == BASELINE:
            continue
        base = per_task[(per_task.split == sp) & (per_task.model == BASELINE)].set_index("task")["rmse"]
        x = g.set_index("task")["rmse"]
        d = (x - base).dropna()
        p = wilcoxon(d).pvalue if len(d) >= 6 and (d != 0).any() else np.nan
        rows.append({"split": sp, "model": m, "n_tasks": len(d), "mean_rmse_diff_vs_baseline": d.mean(),
                     "n_better": int((d < 0).sum()), "wilcoxon_p": p})
    pd.DataFrame(rows).to_csv(r / "paired_vs_baseline.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
