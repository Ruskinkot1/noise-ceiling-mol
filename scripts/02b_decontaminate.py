#!/usr/bin/env python
"""Шаг 2b. Деконтаминация датасета (после 02_standardize, до сплитов).

Вход : data/processed/*.molecules.csv, *.measurements.csv (+ tasks.csv)  [формат см. scripts/02_standardize.py]
Реестр: data/registry/*.csv.gz (scripts/build_registry.py), отсечки: configs/model_cutoffs.yaml

Шаги (лог потерь на каждом): exact InChIKey (полный) -> exact InChIKey14 -> близкие аналоги Tanimoto(ECFP4) > порог
  -> фильтр документов по году (doc year > год эффективной отсечки; отсечка по умолчанию = самая поздняя среди моделей).
Выход: data/processed_raw (копия без очистки), data/processed_clean (после очистки, тот же формат + tasks.csv),
  decontam_loss.csv (потери по задачам и шагам), decontam_models.csv (отсечка и доля пересечения по моделям)
  в data/processed_clean.
--pilot : ничего не пишет; печатает распределение максимума сходства к реестру и пересечения по источникам.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ncmol.decontam import (aggregate_molecules, annotate, apply_year_filter, decontaminate_task, load_cutoffs,
                            load_registry, model_overlap, model_registry_sources)

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load_tasks(proc: Path):
    tasks = {}
    for f in sorted(proc.glob("*.molecules.csv")):
        name = f.name[: -len(".molecules.csv")]
        mf = proc / f"{name}.measurements.csv"
        if mf.exists():
            tasks[name] = (pd.read_csv(mf), pd.read_csv(f))
    return tasks


def unique_molecules(tasks):
    allm = pd.concat([m[["mol_id", "smiles"]] for _, m in tasks.values()], ignore_index=True)
    return allm.drop_duplicates("mol_id").reset_index(drop=True)


def pilot(info: pd.DataFrame, thr: float):
    n = len(info)
    print(f"\n=== PILOT: {n} уникальных молекул датасета ===")
    print(f"точное совпадение ik14: {int(info['exact14'].sum())} ({info['exact14'].mean():.1%}); "
          f"полный InChIKey: {int(info['exact_full'].sum())} ({info['exact_full'].mean():.1%})")
    srcs = info["exact_sources"][info["exact_sources"] != ""].str.split(";").explode().value_counts()
    print("\nПересечение по источникам (число молекул датасета, ik14):")
    print(srcs.rename("n").to_frame().assign(share=lambda d: (d["n"] / n).round(4)).to_string())
    rest = info.loc[~info["exact14"], "max_sim"]
    print(f"\nМакс. Tanimoto ECFP4 к реестру у молекул БЕЗ точного совпадения (n={len(rest)}):")
    if len(rest):
        q = rest.quantile([.05, .25, .5, .75, .9, .95, .99])
        print("квантили: " + ", ".join(f"q{int(k * 100)}={v:.2f}" for k, v in q.items()))
        h, edges = np.histogram(rest, bins=np.linspace(0, 1, 11))
        for c, lo, hi in zip(h, edges[:-1], edges[1:]):
            print(f"  [{lo:.1f},{hi:.1f}) {c:7d} {c / len(rest):6.1%} {'#' * int(50 * c / max(h.max(), 1))}")
        print("\nПорог -> сколько молекул удалилось бы (сверх точных):")
        for t in THRESHOLDS:
            k = int((rest > t).sum())
            print(f"  >{t:<5} {k:7d}  ({k / n:.1%} от всех; суммарно с точными {(k + int(info['exact14'].sum())) / n:.1%})")
    print(f"\nВыбранный порог: {thr}. Обоснование: см. docs/decontamination.md (по данным этого пилота).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=str(ROOT / "data/processed"))
    ap.add_argument("--raw-out", default=str(ROOT / "data/processed_raw"))
    ap.add_argument("--clean-out", default=str(ROOT / "data/processed_clean"))
    ap.add_argument("--registry", default=str(ROOT / "data/registry"))
    ap.add_argument("--sources", nargs="*", default=None, help="подмножество файлов реестра (по умолчанию все)")
    ap.add_argument("--cutoffs", default=str(ROOT / "configs/model_cutoffs.yaml"))
    ap.add_argument("--models", nargs="*", default=None, help="модели из YAML (по умолчанию все)")
    ap.add_argument("--cutoff-year", type=int, default=None, help="явный год отсечки вместо max по моделям")
    ap.add_argument("--no-year", action="store_true", help="пропустить фильтр по году")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--endpoints", default=str(ROOT / "configs/endpoints.yaml"))
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-n", type=int, default=5000, help="размер выборки молекул для пилота")
    a = ap.parse_args()

    proc = Path(a.processed)
    tasks = load_tasks(proc)
    if not tasks:
        print(f"нет *.molecules.csv/*.measurements.csv в {proc}", file=sys.stderr)
        return 1
    reg = load_registry(a.registry, a.sources)
    if reg.empty:
        print(f"реестр пуст ({a.registry}); сначала scripts/build_registry.py", file=sys.stderr)
        return 1
    print(f"реестр: {len(reg)} строк, источников {reg['source'].nunique()}, ik14 уникальных {reg['inchikey14'].nunique()}")
    uniq = unique_molecules(tasks)
    if a.pilot:
        if len(uniq) > a.pilot_n:  # полный перебор дорог: распределение оцениваем по случайной выборке
            uniq = uniq.sample(a.pilot_n, random_state=0).reset_index(drop=True)
            print(f"пилот на случайной выборке {a.pilot_n} молекул (seed 0)")
        pilot(annotate(uniq, reg, a.threshold, a.n_jobs, exact_sim=True), a.threshold)
        return 0
    info = annotate(uniq, reg, a.threshold, a.n_jobs)

    cuts = load_cutoffs(a.cutoffs)
    names = a.models or list(cuts)
    cuts = {k: cuts[k] for k in names}
    msrc = {k: v for k, v in model_registry_sources(a.cutoffs).items() if k in names}
    cutoff_year = None if a.no_year else (a.cutoff_year or max(c.year for c in cuts.values()))
    print(f"год отсечки для clean: {cutoff_year}" + ("" if a.cutoff_year else
          " (max по моделям; не проверено у: " + ", ".join(k for k, c in cuts.items() if not c.verified) + ")"))

    # raw: копия без очистки
    raw_out, clean_out = Path(a.raw_out), Path(a.clean_out)
    raw_out.mkdir(parents=True, exist_ok=True)
    clean_out.mkdir(parents=True, exist_ok=True)
    for f in proc.glob("*.csv"):
        shutil.copy2(f, raw_out / f.name)

    ep = yaml.safe_load(open(a.endpoints)) if Path(a.endpoints).exists() else {}
    min_docs, min_rep = ep.get("min_docs", 2), ep.get("min_replicated_mols", 20)
    loss, summary, model_rows = [], [], []
    for name, (meas, mols) in tasks.items():
        cm, co, log = decontaminate_task(meas, mols, info, threshold=a.threshold, cutoff_year=cutoff_year)
        for r in log:
            loss.append(dict(task=name, **r))
        # справочно: сколько осталось бы при отсечке каждой модели по отдельности (после шагов 1-2)
        pre = [r for r in log if r["step"].startswith("near")][0]
        base = meas[meas["mol_id"].isin(set(mols["mol_id"]) & set(
            info.index[~(info["exact14"] | info["exact_full"] | (info["max_sim"] > a.threshold))]))]
        ov = model_overlap(info, mols["mol_id"], msrc)
        for k, c in cuts.items():
            y = aggregate_molecules(apply_year_filter(base, c.year))
            model_rows.append(dict(task=name, model=k, cutoff=str(c.date), basis=c.basis,
                                   verified=c.verified, mols_after_steps12=pre["mols_left"], mols_after_year=len(y),
                                   test_overlap_share=("н/д" if ov.get(k) is None else round(ov[k], 4))))
        co.to_csv(clean_out / f"{name}.molecules.csv", index=False)
        cm.to_csv(clean_out / f"{name}.measurements.csv", index=False)
        n_rep = int((co["n_docs"] >= min_docs).sum()) if len(co) else 0
        summary.append(dict(task=name, n_mols=len(co), n_rep_mols=n_rep, n_docs=cm["doc_id"].nunique(),
                            passes=n_rep >= min_rep))
    pd.DataFrame(summary).to_csv(clean_out / "tasks.csv", index=False)
    L = pd.DataFrame(loss)
    L.to_csv(clean_out / "decontam_loss.csv", index=False)
    M = pd.DataFrame(model_rows)
    M.to_csv(clean_out / "decontam_models.csv", index=False)

    tot = L.groupby("step", sort=False)[["mols_removed", "meas_removed"]].sum()
    last = L.groupby("task").tail(1)
    print("\n=== Потери по шагам (сумма по задачам) ===")
    print(tot.to_string())
    print(f"\nмолекул: {int(L[L.step == 'raw']['mols_left'].sum())} -> {int(last['mols_left'].sum())};"
          f" измерений: {int(L[L.step == 'raw']['meas_left'].sum())} -> {int(last['meas_left'].sum())}")
    print("\n=== По задачам ===")
    print(L.pivot(index="task", columns="step", values="mols_left").reindex(columns=list(tot.index)).to_string())
    print("\n=== Модели: отсечка и доля пересечения тест-молекул с известным обучающим набором ===")
    agg = M.groupby("model", sort=False).agg(cutoff=("cutoff", "first"), basis=("basis", "first"),
                                             verified=("verified", "first"),
                                             mols_after_year=("mols_after_year", "sum"),
                                             overlap=("test_overlap_share", "first"))
    print(agg.to_string())
    print("\nзадач проходит порог после очистки:", sum(s["passes"] for s in summary), "из", len(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
