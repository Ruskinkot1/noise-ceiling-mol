#!/usr/bin/env python
"""Шаг 16. Три межзадачных объяснения разрыва f, которые раздел 8 проверяет и отвергает.

Числа этого абзаца считались вручную при написании статьи и нигде не сохранялись,
из-за чего две корреляции в тексте разошлись с данными. Скрипт считает все три заново
и кладёт результат в results/<набор>/rejected_mechanisms.csv.

  (i)   сосредоточенность задачи в немногих документах: f против числа документов
        и против доли измерений из крупнейшего документа;
  (ii)  межлабораторные сдвиги: f против разрыва между sigma на парах без общих
        авторов и с общими (по задачам, где shared-пар хватает);
  (iii) зависимость шума от активности: модуль междокументного отклонения против
        метки, внутри молекул с повторами.

Запуск: python scripts/16_rejected_mechanisms.py [--results results/all_clean]
        [--processed data/all_clean] [--model ecfp_gbm] [--split random]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

MIN_SHARED_PAIRS = 20  # задача попадает в проверку (ii), если shared-подмножество не меньше


def mechanism_i(frac, processed):
    """f против числа документов задачи и доли измерений из крупнейшего документа."""
    n_docs, largest = {}, {}
    for t in frac.index:
        m = pd.read_csv(processed / f"{t}.measurements.csv")
        n_docs[t] = m.doc_id.nunique()
        largest[t] = m.doc_id.value_counts().iloc[0] / len(m)
    out = []
    for name, d in (("n_documents", n_docs), ("largest_doc_share", largest)):
        r = spearmanr(frac.values, [d[t] for t in frac.index])
        out.append(("i_task_concentration", name, r.statistic, r.pvalue, len(frac)))
    return out


def mechanism_ii(frac, prov_files):
    """f против разрыва sigma(без общих авторов) - sigma(с общими) по задачам."""
    prov = pd.concat([pd.read_csv(f) for f in prov_files if Path(f).exists()], ignore_index=True)
    prov = prov[prov.task != "ALL_TASKS"]
    wide = prov.pivot_table(index="task", columns="subset", values=["sigma_pairs", "n_pairs"])
    gaps = {}
    for t in wide.index:
        try:
            n_shared = wide.loc[t, ("n_pairs", "shared_author")]
            if not np.isfinite(n_shared) or n_shared < MIN_SHARED_PAIRS:
                continue
            gap = wide.loc[t, ("sigma_pairs", "no_shared_author")] - wide.loc[t, ("sigma_pairs", "shared_author")]
        except KeyError:
            continue
        if np.isfinite(gap):
            gaps[t] = gap
    common = [t for t in frac.index if t in gaps]
    if len(common) < 3:
        return [("ii_lab_offsets", "author_gap", float("nan"), float("nan"), len(common))]
    r = spearmanr(frac.loc[common].values, [gaps[t] for t in common])
    return [("ii_lab_offsets", "author_gap", r.statistic, r.pvalue, len(common))]


def mechanism_iii(tasks, processed, trim=3.0):
    """Модуль отклонения от среднего по молекуле против самой метки, внутри задачи."""
    rhos = []
    for t in tasks:
        m = pd.read_csv(processed / f"{t}.measurements.csv")
        g = m.groupby(["mol_id", "doc_id"]).y.mean().reset_index()
        g = g[g.mol_id.isin(g.mol_id.value_counts().loc[lambda s: s >= 2].index)]
        if g.empty:
            continue
        mean = g.groupby("mol_id").y.transform("mean")
        rng = g.groupby("mol_id").y.transform(lambda s: s.max() - s.min())
        g = g[rng <= trim]
        if len(g) < 10:
            continue
        r = spearmanr((g.y - mean[g.index]).abs(), mean[g.index]).statistic
        if np.isfinite(r):
            rhos.append(r)
    pos = sum(r > 0 for r in rhos)
    p = binomtest(pos, len(rhos), 0.5).pvalue if rhos else float("nan")
    return [("iii_heteroscedasticity", "median_within_task_rho", float(np.median(rhos)), p, len(rhos)),
            ("iii_heteroscedasticity", "tasks_positive", float(pos), float("nan"), len(rhos))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_clean")
    ap.add_argument("--processed", default="data/all_clean")
    ap.add_argument("--model", default="ecfp_gbm")
    ap.add_argument("--split", default="random")
    ap.add_argument("--provenance", nargs="*",
                    default=["results/full_clean/provenance_pairs.csv", "results/ext/prov/provenance_pairs.csv"])
    a = ap.parse_args()
    R, P = Path(a.results), Path(a.processed)

    pt = pd.read_csv(R / "per_task_fraction.csv")
    frac = pt[(pt.split == a.split) & (pt.model == a.model)].set_index("task").frac_ceiling.sort_index()

    rows = mechanism_i(frac, P) + mechanism_ii(frac, a.provenance) + mechanism_iii(frac.index, P)
    out = pd.DataFrame(rows, columns=["mechanism", "quantity", "value", "p", "n"])
    out.insert(0, "split", a.split)
    out.insert(0, "model", a.model)
    out.to_csv(R / "rejected_mechanisms.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
