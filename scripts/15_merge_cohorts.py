#!/usr/bin/env python
"""Шаг 15. Объединяет две когорты задач в один набор, по которому считается статья.

results/all_clean и data/all_clean собираются из:
  results/full_clean  — 17 задач первой версии (benchmark, predictions, ceilings)
  results/ext         — 24 добавленные задачи, по одному файлу на задачу (merge_pilot.py ext)
  results/{full,ext}/doc_clean_*_cheap.csv — сплит по документам на обеих когортах
Файлы данных не копируются, а симлинкуются: per-task molecules/measurements лежат в
data/full_clean и data/ext_clean.

Предсказания (*.pred.csv) в git не хранятся, поэтому predictions.csv.gz собирается только
если они есть на диске; без них шаги 05 и 14 дадут точечные оценки без ДИ по молекулам.

Запуск: python scripts/15_merge_cohorts.py
Дальше: 05_fraction_of_ceiling.py --results results/all_clean --processed data/all_clean
        10_ceiling_robustness.py --processed data/all_clean --out results/all_clean
        09_mde_figures.py --results results/all_clean
        make_paper_tables.py
"""
import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def read_many(patterns):
    files = [f for p in patterns for f in sorted(glob.glob(p))]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-out", default="results/all_clean")
    ap.add_argument("--data-out", default="data/all_clean")
    a = ap.parse_args()
    R, D = Path(a.results_out), Path(a.data_out)
    R.mkdir(parents=True, exist_ok=True)
    D.mkdir(parents=True, exist_ok=True)

    bm = [pd.read_csv("results/full_clean/benchmark.csv")]
    ext = read_many(["results/ext/clean_*_cheap.csv"])
    if ext is None:
        raise SystemExit("нет results/ext/clean_*_cheap.csv — сначала scripts/run_ext.sh")
    bm.append(ext)
    doc = read_many(["results/full/doc_clean_*_cheap.csv", "results/ext/doc_clean_*_cheap.csv"])
    if doc is not None:
        bm.append(doc)
    else:
        print("сплит по документам не найден, собираю без него")
    benchmark = pd.concat(bm, ignore_index=True)
    # устойчивый порядок строк: иначе пересборка даёт файл, отличающийся от закоммиченного
    # только перестановкой, и git показывает тысячи строк мнимого диффа
    benchmark = benchmark.sort_values(["task", "split", "seed", "model"], kind="stable").reset_index(drop=True)
    benchmark.to_csv(R / "benchmark.csv", index=False)

    pd.concat([pd.read_csv("results/full_clean/ceilings.csv"),
               pd.read_csv("results/ext/ceil_clean/ceilings.csv")],
              ignore_index=True).to_csv(R / "ceilings.csv", index=False)

    preds = []
    if os.path.exists("results/full_clean/predictions.csv.gz"):
        preds.append(pd.read_csv("results/full_clean/predictions.csv.gz"))
    rest = read_many(["results/ext/clean_*_cheap.pred.csv",
                      "results/full/doc_clean_*_cheap.pred.csv", "results/ext/doc_clean_*_cheap.pred.csv"])
    if rest is not None:
        preds.append(rest)
    if len(preds) > 1:
        pd.concat(preds, ignore_index=True).to_csv(R / "predictions.csv.gz", index=False, compression="gzip")
        print("predictions.csv.gz собран")
    else:
        print("предсказаний по задачам нет на диске; ДИ по молекулам в шаге 05 будут пропущены")

    tasks = pd.concat([pd.read_csv("data/full_clean/tasks.csv"), pd.read_csv("data/ext_clean/tasks.csv")],
                      ignore_index=True)
    tasks[tasks.passes].to_csv(D / "tasks.csv", index=False)
    n = 0
    for src in ("data/full_clean", "data/ext_clean"):
        for f in glob.glob(f"{src}/*.csv"):
            b = os.path.basename(f)
            if b == "tasks.csv" or b.startswith("decontam"):
                continue
            dst = D / b
            if not dst.exists():
                # относительный линк: каталог переживает перенос или клонирование репозитория
                os.symlink(os.path.relpath(f, D), dst)
                n += 1
    print(f"задач {int(tasks.passes.sum())}, строк бенчмарка {len(benchmark)}, "
          f"сплитов {sorted(benchmark.split.unique())}, симлинков {n}")


if __name__ == "__main__":
    main()
