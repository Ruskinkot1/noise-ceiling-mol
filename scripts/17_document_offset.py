#!/usr/bin/env python
"""Шаг 17. Разложение шума на постоянную поправку документа и остаток.

Модель раздела 4 пишет y_ij = t_i + e_ij и ничего не говорит о структуре e. Здесь
проверяется более подробная модель y_ij = t_i + d_j + e_ij, где d_j — постоянная
поправка публикации (лаборатория, протокол, калибровка), общая для всех её молекул.
Доля d в дисперсии шума — это верхняя граница того, сколько модель способна выиграть
от знакомой публикации одним лишь сдвигом.

Оцениватель. Для пары документов с k >= 2 общими молекулами разности δ = y_a − y_b
имеют среднее, равное разности поправок, и разброс, равный удвоенному остатку:

    σ_e² = ΣΣ(δ − δ̄)² / (2 Σ(k−1)),    2σ_d² = ⟨δ̄²⟩ − 2σ_e²⟨1/k⟩

Результат: results/<набор>/document_offset.csv — компоненты, устойчивость к порогу k,
верхняя граница выигрыша от знания документа и сравнение с наблюдаемым разрывом между
случайным разбиением и разбиением по документам.

Запуск: python scripts/17_document_offset.py [--results results/all_clean]
        [--processed data/all_clean] [--n-boot 1000]
"""
import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MAX_SPAN = 3.0  # то же отсечение по размаху, что и в оценке потолка (раздел 4)


def doc_level_frames(processed, tasks):
    """Таблицы «молекула, документ, y» по задачам, с теми же отборами, что в оценке потолка."""
    frames = {}
    for t in tasks:
        m = pd.read_csv(processed / f"{t}.measurements.csv")
        d = m.groupby(["mol_id", "doc_id"], as_index=False).y.mean()
        n = d.groupby("mol_id").y.transform("size")
        d = d[n >= 2]
        span = d.groupby("mol_id").y.transform(lambda s: s.max() - s.min())
        d = d[span <= MAX_SPAN]
        if not d.empty:
            frames[t] = d.reset_index(drop=True)
    return frames


def pair_deltas(frames):
    """Разности по парам документов: по строке на пару (задача, doc_a, doc_b)."""
    rows = []
    for t, d in frames.items():
        by_pair = defaultdict(list)
        for _, g in d.groupby("mol_id"):
            g = g.sort_values("doc_id")
            for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
                by_pair[(a.doc_id, b.doc_id)].append(a.y - b.y)
        for (da, db), deltas in by_pair.items():
            rows.append({"task": t, "doc_a": da, "doc_b": db,
                         "k": len(deltas), "deltas": np.asarray(deltas, float)})
    return pd.DataFrame(rows)


def decompose(pairs):
    """(σ_d, σ_e, доля дисперсии на документной составляющей) по набору пар с k >= 2."""
    p = pairs[pairs.k >= 2]
    ss = sum(float(((v - v.mean()) ** 2).sum()) for v in p.deltas)
    dof = float((p.k - 1).sum())
    sig_e2 = ss / (2 * dof)
    mbar2 = float(np.mean([v.mean() ** 2 for v in p.deltas]))
    sig_d2 = max(0.0, (mbar2 - 2 * sig_e2 * float((1.0 / p.k).mean())) / 2)
    return np.sqrt(sig_d2), np.sqrt(sig_e2), sig_d2 / (sig_d2 + sig_e2)


def synthetic_check(frames, sigma, levels, rng, n_rep=3):
    """Оцениватель на той же структуре молекул и документов с известной долей d."""
    out = []
    for true_frac in levels:
        est = [synthetic_once(frames, sigma, true_frac, rng) for _ in range(n_rep)]
        out.append((true_frac, float(np.mean(est))))
    return out


def synthetic_once(frames, sigma, true_frac, rng):
    syn = {}
    for t, d in frames.items():
        s = float(sigma[t])
        sd, se = s * np.sqrt(true_frac), s * np.sqrt(1 - true_frac)
        docs = {j: rng.normal(0, sd) for j in d.doc_id.unique()}
        f = d.copy()
        f["y"] = (d.groupby("mol_id").y.transform("mean")
                  + d.doc_id.map(docs) + rng.normal(0, se, len(d)))
        syn[t] = f
    return decompose(pair_deltas(syn))[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_clean")
    ap.add_argument("--processed", default="data/all_clean")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R, P = Path(a.results), Path(a.processed)

    ceil = pd.read_csv(R / "ceilings.csv").set_index("task")
    bench = pd.read_csv(R / "benchmark.csv")
    tasks = sorted(ceil.index)

    frames = doc_level_frames(P, tasks)
    pairs = pair_deltas(frames)
    n_any = len(pairs)
    pairs = pairs[pairs.k >= 2].reset_index(drop=True)
    sig_d, sig_e, frac = decompose(pairs)

    rng = np.random.default_rng(a.seed)
    boot = []
    for _ in range(a.n_boot):
        idx = rng.integers(0, len(pairs), len(pairs))
        boot.append(decompose(pairs.iloc[idx])[2])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    means = np.array([v.mean() for v in pairs.deltas])
    rows = [("components", "sigma_d", sig_d, np.nan),
            ("components", "sigma_e", sig_e, np.nan),
            ("components", "document_fraction", frac, np.nan),
            ("components", "document_fraction_lo", lo, np.nan),
            ("components", "document_fraction_hi", hi, np.nan),
            ("material", "pairs_sharing_any_molecule", n_any, np.nan),
            ("material", "pairs_used", len(pairs), np.nan),
            ("material", "median_abs_offset", float(np.median(np.abs(means))), np.nan),
            ("material", "share_offset_over_half", float((np.abs(means) > 0.5).mean()), np.nan)]

    # устойчивость к порогу k и к шкале: разности, делённые на σ своей задачи
    for kmin in (2, 3, 5, 10):
        sub = pairs[pairs.k >= kmin]
        rows.append(("robustness", f"fraction_k{kmin}", decompose(sub)[2], len(sub)))
    std = pairs.copy()
    std["deltas"] = [v / ceil.sigma[t] for v, t in zip(pairs.deltas, pairs.task)]
    rows.append(("robustness", "fraction_standardised", decompose(std)[2], len(std)))
    # пары с k >= 2 — смещённая выборка: насколько они шумнее типичной задачи
    all_d = np.concatenate(list(pairs.deltas))
    rows.append(("robustness", "sigma_on_used_pairs", float(np.sqrt((all_d**2).mean() / 2)), len(pairs)))
    rows.append(("robustness", "median_task_sigma", float(ceil.sigma.median()), len(ceil)))

    # верхняя граница выигрыша от знания документа против наблюдаемого разрыва
    shrink = np.sqrt(1 - frac)
    b = bench.groupby(["task", "split", "model"]).agg(
        rmse=("rmse", "mean"), kinv=("test_mean_inv_ndocs", "mean")).reset_index()
    floor_rand = (b[b.split == "random"].drop_duplicates("task")
                  .assign(floor=lambda x: np.sqrt(ceil.sigma[x.task].values**2 * x.kinv))
                  .set_index("task").floor)
    max_gain = floor_rand * (1 - shrink)
    rows.append(("bound", "floor_random_median", float(floor_rand.median()), len(floor_rand)))
    rows.append(("bound", "shrink_factor", float(shrink), np.nan))
    rows.append(("bound", "max_gain_median", float(max_gain.median()), len(max_gain)))
    for model in sorted(b.model.unique()):
        w = b[b.model == model].pivot_table(index="task", columns="split", values="rmse")
        if not {"random", "document"} <= set(w.columns):
            continue
        gap = (w["document"] - w["random"]).dropna()
        common = gap.index.intersection(max_gain.index)
        rows.append(("bound", f"observed_gap_median_{model}", float(gap.median()), len(gap)))
        rows.append(("bound", f"ratio_observed_over_max_{model}",
                     float(gap.median() / max_gain.median()), len(common)))
        rows.append(("bound", f"tasks_gap_over_max_{model}",
                     float((gap[common] > max_gain[common]).sum()), len(common)))

    for true_frac, est in synthetic_check(frames, ceil.sigma,
                                         (0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8), rng):
        rows.append(("synthetic", f"recovered_at_true_{true_frac:.2f}", est, np.nan))

    out = pd.DataFrame(rows, columns=["group", "quantity", "value", "n"])
    out.to_csv(R / "document_offset.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
