#!/usr/bin/env python
"""Шаг 19. Связь доли потолка с шумом по всем четырём разбиениям.

Разделы 8 и 9 приводят два края этой связи: ранговую корреляцию f с оценённым шумом
при случайном разбиении и при разбиении по документам. Если связь создаётся документной
опорой, она должна убывать вместе с опорой, а не переключаться между двумя состояниями.
Скрипт считает корреляцию для всех четырёх разбиений и всех моделей; доли тестовых
молекул без опоры берутся из results/document_support*/gradient.csv.

Выход: results/<набор>/confounding_gradient.csv.

Запуск: python scripts/19_confounding_gradient.py [--results results/all_clean]
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

SPLITS = ["random", "scaffold", "temporal", "document"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_clean")
    a = ap.parse_args()
    R = Path(a.results)
    sigma = pd.read_csv(R / "ceilings.csv").set_index("task").sigma
    pt = pd.read_csv(R / "per_task_fraction.csv")

    rows = []
    for model in sorted(pt.model.unique()):
        for sp in SPLITS:
            g = pt[(pt.model == model) & (pt.split == sp)].set_index("task").frac_ceiling.dropna()
            common = g.index.intersection(sigma.index)
            if len(common) < 3:
                continue
            r = spearmanr(g[common].values, sigma[common].values)
            rows.append((model, sp, r.statistic, r.pvalue, len(common)))
    out = pd.DataFrame(rows, columns=["model", "split", "rho_f_sigma", "p", "n_tasks"])
    out.to_csv(R / "confounding_gradient.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
