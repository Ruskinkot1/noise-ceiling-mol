#!/usr/bin/env python
"""Step 5. "Fraction of ceiling" and comparison with the ECFP+GBM baseline.

Input:
  results/benchmark.csv   — scripts/04 (task, split, seed, model, rmse, mae, r2, spearman, rmse_mean,
                            n_test, test_mean_inv_ndocs, ...)
  results/ceilings.csv    — scripts/03 (task, sigma, var_y, sigma_lo, sigma_hi, ...)
  results/predictions.csv|parquet (OPTIONAL; schema: task, split, seed, model, mol_id, pred
                            [+ y, n_docs, y_train_mean]) — needed for per-molecule CIs. Without it
                            only point estimates + spread across seeds/tasks are produced.

PRIMARY metric: frac_ceiling = (RMSE_mean - RMSE_model) / (RMSE_mean - RMSE_floor),
RMSE_floor = sigma*sqrt(mean(1/n_docs)) over test molecules. 0 = train mean, 1 = ceiling.
Auxiliary: frac_rmse = RMSE_floor/RMSE, frac_r2 = R2/R2_max, frac_ceiling_true (attenuation correction).

Output (results/):
  benchmark_with_ceiling.csv   — benchmark rows + ceiling and all fractions
  per_task_fraction.csv        — task x split x model: fraction of ceiling (+ per-molecule CI if predictions exist)
  leaderboard.csv              — split x model: mean over tasks and CI (over molecules; over tasks)
  paired_vs_baseline.csv       — model vs ECFP+GBM: delta of fraction of ceiling, Wilcoxon over tasks,
                                 p with Holm correction (primary) and BH (sensitivity), paired bootstrap
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ncmol import metrics, stats
from ncmol.models import BASELINE

ROOT = Path(__file__).resolve().parents[1]
LOW_HEADROOM = 0.2  # parameter: (RMSE_mean - RMSE_floor)/RMSE_mean < 0.2 -> "fraction of ceiling is unstable"
SUMMARY = ["rmse", "r2", "frac_ceiling", "frac_rmse", "frac_r2", "frac_ceiling_true", "headroom"]


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
    # headroom to the ceiling: if small, the denominator of the fraction of ceiling is near 0 and the fraction is unstable
    b["headroom"] = (b["rmse_mean"] - b["rmse_floor"]) / b["rmse_mean"]
    b["low_headroom"] = b["headroom"] < LOW_HEADROOM
    for tag in ("lo", "hi"):  # sensitivity to sigma uncertainty (bootstrap bounds from scripts/03)
        if f"sigma_{tag}" in b.columns:
            fl = b[f"sigma_{tag}"] * np.sqrt(b["test_mean_inv_ndocs"])
            b[f"frac_ceiling_at_sigma_{tag}"] = (b["rmse_mean"] - b["rmse"]) / (b["rmse_mean"] - fl)
    return b


def bootstrap_tables(pred: pd.DataFrame, ceil: pd.DataFrame, n_boot: int, alpha: float, seed: int):
    """Returns (per_task DataFrame, draws[(split,model)] -> (n_tasks, B), diffs[(split,model)])."""
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
    ap.add_argument("--min-tasks", type=int, default=6, help="minimum number of tasks for Wilcoxon")
    ap.add_argument("--correction", choices=["holm", "bh"], default="holm", help="which correction goes into the p_adj column")
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
        print("results/predictions.* not found: per-molecule CIs are not computed (schema: docs/experiment_design.md).")
    else:
        boot, draws, ddraws = bootstrap_tables(pred, c, a.n_boot, a.alpha, a.seed)
        per_task = per_task.merge(boot, on=["task", "split", "model"], how="left")
    per_task.to_csv(r / "per_task_fraction.csv", index=False)

    # ---- leaderboard: mean over tasks (equal task weights)
    lb_rows = []
    rng = np.random.default_rng(a.seed + 7)
    for (sp, m), g in per_task.groupby(["split", "model"]):
        row = {"split": sp, "model": m, "n_tasks": g["task"].nunique(), **{k: g[k].mean() for k in cols}}
        # spread over tasks: task bootstrap (accounts for the task set also being a sample)
        v = g["frac_ceiling"].to_numpy()
        if len(v) >= 2:
            bt = rng.choice(v, size=(a.n_boot, len(v)), replace=True).mean(1)
            row["frac_ceiling_task_lo"], row["frac_ceiling_task_hi"] = np.percentile(bt, [100 * a.alpha / 2, 100 * (1 - a.alpha / 2)])
        for tag in ("lo", "hi"):
            if f"frac_ceiling_at_sigma_{tag}" in b.columns:
                row[f"frac_ceiling_at_sigma_{tag}"] = b[(b.split == sp) & (b.model == m)].groupby("task")[f"frac_ceiling_at_sigma_{tag}"].mean().mean()
        if "frac_ceiling_pooled" in g.columns:
            row["frac_ceiling_pooled"] = g["frac_ceiling_pooled"].mean()  # the estimate to which the per-molecule CIs refer
        d = draws.get((sp, m))
        if d:
            mat = np.vstack(list(d.values()))  # (n_tasks, B): molecule resampling within each task is independent
            mm = np.nanmean(mat, 0)  # resamples with RMSE_mean <= RMSE_floor give NaN and are skipped
            row["frac_ceiling_mol_lo"], row["frac_ceiling_mol_hi"] = stats.percentile_ci(mm, a.alpha)
        lb_rows.append(row)
    lb = pd.DataFrame(lb_rows).sort_values(["split", "frac_ceiling"], ascending=[True, False])
    lb.to_csv(r / "leaderboard.csv", index=False)
    print(lb.to_string(index=False))

    # ---- paired comparison with the baseline: Wilcoxon over tasks + paired bootstrap
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
            mat = np.nanmean(np.vstack([v[1] for v in dd.values()]), 0)
            row["diff_mol_lo"], row["diff_mol_hi"] = stats.percentile_ci(mat, a.alpha)
            row["inside_noise_mol"] = bool(row["diff_mol_lo"] <= 0 <= row["diff_mol_hi"])
        prow.append(row)
    pr = pd.DataFrame(prow)
    if len(pr):
        pr["p_holm"] = stats.holm(pr["wilcoxon_p"])
        pr["p_bh"] = stats.benjamini_hochberg(pr["wilcoxon_p"])
        pr["p_adj"] = pr["p_holm" if a.correction == "holm" else "p_bh"]
        pr["inside_noise_tasks"] = ~(pr["p_adj"] < a.alpha)  # NaN -> inside noise (too few tasks / no effect)
    pr.to_csv(r / "paired_vs_baseline.csv", index=False)
    print(pr.to_string(index=False))


if __name__ == "__main__":
    main()
