#!/usr/bin/env python
"""Шаг 7. Разрезы разницы «модель − ECFP+GBM» и артефакты сравнения.

Метрика разреза: ΔRMSE = RMSE(модель) − RMSE(ECFP+GBM) на подмножестве тест-молекул (< 0 = модель лучше),
в единицах pActivity и в единицах sigma задачи (ΔRMSE/sigma, сопоставимо между задачами). Доля потолка в
разрезах НЕ используется: нулевая точка и потолок на подмножестве (например, узкий диапазон активности)
не определены. ДИ — парный кластерный бутстрэп по молекулам внутри подмножества задачи; для разрезов,
объединяющих задачи, берётся среднее по задачам с равным весом. Пометка «внутри шума» = ДИ содержит 0.

Разрезы: all (по сплиту), target, noise (n_docs молекулы: 1 / 2 / 3+; прокси шума метки, дисперсия ∝ 1/n_docs),
tanimoto (макс. Танимото ECFP4 к train: бины), cliff (у молекулы есть сосед в train с Tanimoto >= --cliff-sim
и |Δy| >= --cliff-delta), y_range (терцили активности в задаче), task_size (терцили числа молекул в задаче).

Вход: results/predictions.csv|parquet (task, split, seed, model, mol_id, pred), results/ceilings.csv
(необязательно: sigma), data/processed/*. Выход results/slices/:
  slices.csv                 — разрезы
  comparison_artifacts.csv   — по task x split x seed: размер теста, шум метки, дисбаланс диапазона,
                               распределение max Tanimoto test→train, дрейф по годам (не требует предсказаний)
  cliff_pairs.csv            — метрики на парах-обрывах внутри теста (знак Δ и RMSE Δ), если есть предсказания
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ncmol import stats
from ncmol.chem import ecfp
from ncmol.models import BASELINE
from ncmol.splits import SPLITS

ROOT = Path(__file__).resolve().parents[1]
SIM_BINS = [0, 0.4, 0.6, 0.8, 1.0001]
SIM_LABELS = ["<0.4", "0.4-0.6", "0.6-0.8", ">=0.8"]


def test_context(mol: pd.DataFrame, fp: np.ndarray, split: str, seed: int, sim_min: float, delta_min: float):
    """Контекст тест-молекул одного (сплит, сид): max Tanimoto к train и флаг cliff-молекулы."""
    tr, te = SPLITS[split](mol, seed)
    y = mol["y"].to_numpy(float)
    ms = stats.max_tanimoto_to_train(fp[te], fp[tr])
    cl = stats.cliff_mask_vs_train(fp[te], y[te], fp[tr], y[tr], sim_min, delta_min)
    ctx = pd.DataFrame({"seed": seed, "mol_id": mol["mol_id"].iloc[te].to_numpy(), "max_sim": ms, "cliff": cl})
    return tr, te, ctx


def artifacts(task, mol, fp, sigma, seeds, splits, sim_min, delta_min):
    rows = []
    y = mol["y"].to_numpy(float)
    for sp in splits:
        for seed in ([0] if sp == "temporal" else seeds):
            tr, te, ctx = test_context(mol, fp, sp, seed, sim_min, delta_min)
            yt, yr = y[te], y[tr]
            q5, q95 = np.percentile(yr, [5, 95])
            nd = mol["n_docs"].to_numpy(float)[te]
            floor = sigma * np.sqrt(np.mean(1 / nd)) if sigma is not None else np.nan
            rows.append({
                "task": task, "split": sp, "seed": seed, "n_train": len(tr), "n_test": len(te),
                "test_mean_n_docs": nd.mean(), "test_frac_ndocs_ge2": float((nd >= 2).mean()),
                "rmse_floor_test": floor, "floor_over_sd_y_test": floor / yt.std() if yt.std() > 0 else np.nan,
                "y_mean_train": yr.mean(), "y_mean_test": yt.mean(), "y_sd_train": yr.std(), "y_sd_test": yt.std(),
                "y_sd_ratio_test_train": yt.std() / yr.std() if yr.std() > 0 else np.nan,
                "y_ks_stat": ks_2samp(yr, yt).statistic, "frac_test_outside_train_q5_q95": float(((yt < q5) | (yt > q95)).mean()),
                "maxsim_mean": ctx["max_sim"].mean(), "maxsim_median": ctx["max_sim"].median(),
                "maxsim_q10": ctx["max_sim"].quantile(0.1), "maxsim_q90": ctx["max_sim"].quantile(0.9),
                "frac_maxsim_ge_0.8": float((ctx["max_sim"] >= 0.8).mean()), "frac_cliff_mols": float(ctx["cliff"].mean()),
                "year_median_train": float(mol["first_year"].iloc[tr].median()), "year_median_test": float(mol["first_year"].iloc[te].median()),
            })
    return rows


def slice_labels(t, sp, base: pd.DataFrame, mol: pd.DataFrame, ctx: pd.DataFrame, target: str, size_label):
    b = base.merge(ctx, on=["seed", "mol_id"], how="left")
    q1, q2 = np.quantile(mol["y"], [1 / 3, 2 / 3])
    return {
        "all": np.full(len(b), "all"),
        "target": np.full(len(b), target),
        "noise": np.where(b["n_docs"] >= 3, "3+", b["n_docs"].astype(int).astype(str)),
        "tanimoto": pd.cut(b["max_sim"], SIM_BINS, labels=SIM_LABELS, right=False).astype(str).to_numpy(),
        "cliff": np.where(b["cliff"].fillna(False).astype(bool), "cliff", "no_cliff"),
        "y_range": np.where(b["y"] < q1, "y_low", np.where(b["y"] < q2, "y_mid", "y_high")),
        "task_size": np.full(len(b), size_label),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--processed", default=str(ROOT / "data/processed"))
    ap.add_argument("--out", default=str(ROOT / "results/slices"))
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5, help="сиды для артефактов, если предсказаний нет")
    ap.add_argument("--cliff-sim", type=float, default=0.8, help="порог Tanimoto ECFP4 для cliff (параметр, не подбирался)")
    ap.add_argument("--cliff-delta", type=float, default=1.5, help="порог |ΔpActivity| для cliff (параметр, не подбирался)")
    ap.add_argument("--min-mols", type=int, default=30, help="минимум молекул в ячейке разреза (на задачу)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = Path(a.results)
    pdir = Path(a.processed)
    tasks = pd.read_csv(pdir / "tasks.csv")
    tmeta = tasks.set_index("task")
    sig = pd.read_csv(r / "ceilings.csv").set_index("task")["sigma"] if (r / "ceilings.csv").exists() else pd.Series(dtype=float)
    pred = stats.load_predictions(r, pdir)
    names = sorted(pred["task"].unique()) if pred is not None else tasks[tasks["passes"]]["task"].tolist()
    mols = {t: pd.read_csv(pdir / f"{t}.molecules.csv") for t in names}
    fps = {t: ecfp(list(mols[t]["smiles"]), counts=False) for t in names}  # бинарные ECFP4 2048 для Танимото
    nmol = pd.Series({t: len(mols[t]) for t in names})
    size_lab = {}
    if len(nmol) >= 3:
        lo, hi = nmol.quantile([1 / 3, 2 / 3])
        size_lab = {t: ("small" if n <= lo else "large" if n >= hi else "mid") for t, n in nmol.items()}

    art = []
    for t in names:
        art += artifacts(t, mols[t], fps[t], float(sig[t]) if t in sig.index else None, range(a.seeds), list(SPLITS), a.cliff_sim, a.cliff_delta)
    pd.DataFrame(art).to_csv(out / "comparison_artifacts.csv", index=False)
    print(f"артефакты сравнения: {len(art)} строк -> {out / 'comparison_artifacts.csv'}")
    if pred is None:
        print("results/predictions.* не найден: разрезы не посчитаны (схема — docs/experiment_design.md)")
        return

    cells = {}  # (split, slice_type, slice_value, model) -> list of per-task dicts
    cliff_rows = []
    for (t, sp), g in pred.groupby(["task", "split"]):
        models = sorted(g["model"].unique(), key=lambda m: (m != BASELINE, m))
        base, P, models = stats.wide_by_model(g, models)
        if BASELINE not in models or len(models) < 2:
            continue
        ctx = pd.concat([test_context(mols[t], fps[t], sp, int(s), a.cliff_sim, a.cliff_delta)[2] for s in sorted(base["seed"].unique())])
        labels = slice_labels(t, sp, base, mols[t], ctx, tmeta.loc[t, "target"], size_lab.get(t, "n/a"))
        y = base["y"].to_numpy(float)
        ids = base["mol_id"].to_numpy()
        b0 = models.index(BASELINE)
        s_t = float(sig[t]) if t in sig.index else np.nan
        for stype, lab in labels.items():
            for val in pd.unique(lab):
                if val == "nan":
                    continue
                mk = np.flatnonzero(lab == val)
                if len(np.unique(ids[mk])) < a.min_mols:
                    continue
                yy, PP, ii = y[mk], P[mk], ids[mk]

                def f(i):
                    rm = np.sqrt(np.mean((yy[i][:, None] - PP[i]) ** 2, axis=0))
                    return rm - rm[b0]

                est = f(np.arange(len(yy)))
                dr = stats.cluster_bootstrap(ii, f, a.n_boot, a.seed)
                for k, m in enumerate(models):
                    if m == BASELINE:
                        continue
                    cells.setdefault((sp, stype, val, m), []).append(
                        {"task": t, "n_mols": len(np.unique(ii)), "est": est[k], "draws": dr[:, k], "sigma": s_t})
        # пары-обрывы внутри теста (по сидам): способность предсказать знак и величину Δ
        for seed, gs in base.groupby("seed"):
            tr, te, _ = test_context(mols[t], fps[t], sp, int(seed), a.cliff_sim, a.cliff_delta)
            pairs = stats.cliff_pairs(fps[t][te], mols[t]["y"].to_numpy(float)[te], a.cliff_sim, a.cliff_delta)
            pos = {mid: j for j, mid in enumerate(mols[t]["mol_id"].iloc[te])}
            gk = gs.reset_index(drop=True)
            Pg = P[gs.index.to_numpy()]
            order = np.array([pos[mid] for mid in gk["mol_id"]])
            inv = np.empty(len(te), int)
            inv[order] = np.arange(len(order))
            for k, m in enumerate(models):
                met = stats.cliff_pair_metrics(pairs, mols[t]["y"].to_numpy(float)[te], Pg[inv, k]) if len(pairs) else {"n_pairs": 0}
                cliff_rows.append({"task": t, "split": sp, "seed": seed, "model": m, **met})

    rows = []
    for (sp, stype, val, m), lst in cells.items():
        est = np.mean([c["est"] for c in lst])
        dr = np.vstack([c["draws"] for c in lst]).mean(0)
        lo, hi = stats.percentile_ci(dr, a.alpha)
        row = {"slice_type": stype, "slice_value": val, "split": sp, "model": m, "n_tasks": len(lst),
               "n_mols": int(sum(c["n_mols"] for c in lst)), "delta_rmse": est, "lo": lo, "hi": hi,
               "se": float(np.std(dr, ddof=1)), "mde_80": float(stats.mde_from_se(np.std(dr, ddof=1))),
               "flag": stats.slice_flag(lo, hi, lower_is_better=True)}
        if all(np.isfinite(c["sigma"]) for c in lst):
            sg = np.array([c["sigma"] for c in lst])[:, None]
            drs = (np.vstack([c["draws"] for c in lst]) / sg).mean(0)
            row.update(delta_rmse_over_sigma=float(np.mean([c["est"] / c["sigma"] for c in lst])),
                       lo_over_sigma=float(np.percentile(drs, 100 * a.alpha / 2)), hi_over_sigma=float(np.percentile(drs, 100 * (1 - a.alpha / 2))))
        rows.append(row)
    sl = pd.DataFrame(rows).sort_values(["slice_type", "split", "model", "slice_value"])
    sl.to_csv(out / "slices.csv", index=False)
    pd.DataFrame(cliff_rows).to_csv(out / "cliff_pairs.csv", index=False)
    print(sl.to_string(index=False))


if __name__ == "__main__":
    main()
