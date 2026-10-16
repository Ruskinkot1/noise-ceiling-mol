#!/usr/bin/env python
"""Шаг 9. Эмпирический MDE и рисунки для статьи по results/full_clean.
MDE на одном тесте (задача x сплит x сид) для пары модель vs ECFP+GBM:
d_i = e_m,i^2 - e_b,i^2 по тест-молекулам; SE = sd(d)/sqrt(n); MDE_MSE = (z_.975 + z_.8) SE;
MDE_RMSE = MDE_MSE / (2 RMSE_b) (дельта-метод, src/ncmol/stats.py); в долях потолка: / (RMSE_mean - RMSE_floor).
Выход: results/full_clean/mde.csv, paper/figures/{paired_diff,floor_vs_rmse}.pdf, paper/sections/mde_numbers.tex"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ncmol.stats import mde_from_se

R = "results/full_clean"
BASE = "ecfp_gbm"
MODELS = {"ecfp_rf": ("ECFP+RF", "#2a78d6", "o"), "emb-chemberta-77m-mlm_gbm": ("ChemBERTa (frozen)+GBM", "#eb6834", "s"),
          "emb-molformer-xl-10pct_gbm": ("MoLFormer (frozen)+GBM", "#1baf7a", "^")}
SPL = ["random", "scaffold", "temporal"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

pred = pd.read_csv(f"{R}/predictions.csv")
floor = pd.read_csv(f"{R}/ceilings.csv").set_index("task").rmse_floor
pred["se2"] = (pred.pred - pred.y) ** 2
rows = []
for (t, sp, s), g in pred.groupby(["task", "split", "seed"]):
    w = g.pivot_table(index="mol_id", columns="model", values="se2")
    yv = g.drop_duplicates("mol_id").set_index("mol_id")
    if BASE not in w: continue
    rmse_b = np.sqrt(w[BASE].mean())
    rmse_mean = np.sqrt(((yv.y - yv.y_train_mean) ** 2).mean())
    for m in MODELS:
        if m not in w: continue
        d = (w[m] - w[BASE]).dropna()
        mde_rmse = float(mde_from_se(d.std(ddof=1) / np.sqrt(len(d)))) / (2 * rmse_b)
        rows.append({"task": t, "split": sp, "seed": s, "model": m, "n_test": len(d), "mde_rmse": mde_rmse,
                     "mde_frac": mde_rmse / (rmse_mean - floor[t])})
mde = pd.DataFrame(rows)
mde.to_csv(f"{R}/mde.csv", index=False)
summ = mde.groupby(["model", "split"])[["mde_rmse", "mde_frac"]].median()
print(summ.round(3).to_string())

# числа для текста статьи
rf = summ.loc["ecfp_rf"]
with open("paper/sections/mde_numbers.tex", "w") as f:
    f.write("% сгенерировано scripts/09_mde_figures.py\n")
    for sp in SPL:
        f.write(f"\\newcommand{{\\mde{sp.capitalize()}Rmse}}{{{rf.loc[sp,'mde_rmse']:.3f}}}\n")
        f.write(f"\\newcommand{{\\mde{sp.capitalize()}Frac}}{{{rf.loc[sp,'mde_frac']:.3f}}}\n")

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
# Рис. 1: парная разность доли потолка с ECFP+GBM по задачам; серая полоса = медианный MDE (±) для ECFP+RF
ptf = pd.read_csv(f"{R}/per_task_fraction.csv")
base = ptf[ptf.model == BASE].set_index(["task", "split"]).frac_ceiling
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
for ax, sp in zip(axes, SPL):
    ax.axvline(0, color=MUTED, lw=0.8)
    for k, (m, (lab, col, mk)) in enumerate(MODELS.items()):
        band = summ.loc[(m, sp), "mde_frac"]  # своя полоса MDE для каждой модели
        ax.fill_between([-band, band], k - 0.42, k + 0.42, color=GRID, lw=0, zorder=1)
        q = ptf[(ptf.model == m) & (ptf.split == sp)].set_index("task").frac_ceiling
        diff = (q - base.xs(sp, level="split").reindex(q.index)).dropna()
        yj = k + np.linspace(-0.25, 0.25, len(diff))
        ax.scatter(diff.values, yj, s=14, color=col, marker=mk, edgecolor="white", linewidth=0.5, label=lab, zorder=3)
        ax.plot([diff.mean()] * 2, [k - 0.35, k + 0.35], color=INK, lw=2, solid_capstyle="round", zorder=4)
    ax.set_title(sp, color=INK, fontsize=9)
    ax.set_yticks(range(len(MODELS)), [v[0] for v in MODELS.values()])
    ax.grid(axis="x", color=GRID, lw=0.5)
axes[0].invert_yaxis()
fig.supxlabel("Δ fraction of ceiling vs ECFP+GBM (points: tasks; bar: mean; grey band: ± median single-test-set MDE of that model)", fontsize=8, color=INK)
fig.tight_layout()
fig.savefig("paper/figures/paired_diff.pdf")
fig.savefig("paper/figures/paired_diff.png", dpi=150)

# Рис. 2: RMSE_floor против RMSE ECFP+RF по задачам, три сплита
lb = ptf[ptf.model == "ecfp_rf"]
fig, ax = plt.subplots(figsize=(3.4, 3.0))
lim = [0.2, max(lb.rmse.max(), floor.max()) * 1.05]
ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
ax.text(lim[1] * 0.97, lim[1] * 0.93, "RMSE = floor", color=MUTED, ha="right", fontsize=7)
for sp, col, mk in zip(SPL, ["#2a78d6", "#eb6834", "#1baf7a"], ["o", "s", "^"]):
    q = lb[lb.split == sp]
    ax.scatter(floor.reindex(q.task), q.rmse, s=16, color=col, marker=mk, edgecolor="white", linewidth=0.5, label=sp, zorder=3)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("estimated noise floor, RMSE$_{floor}$")
ax.set_ylabel("RMSE of ECFP+RF")
ax.grid(color=GRID, lw=0.5)
ax.legend(frameon=False, fontsize=7, loc="lower right", scatterpoints=1, title="split", title_fontsize=7)
fig.tight_layout()
fig.savefig("paper/figures/floor_vs_rmse.pdf")
fig.savefig("paper/figures/floor_vs_rmse.png", dpi=150)
print("ok")
