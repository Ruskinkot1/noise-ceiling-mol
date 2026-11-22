#!/usr/bin/env python
"""Шаг 14. Поддержка документа: сколько молекул из документа тестовой молекулы попало в обучение.

Почему это нужно. На случайном и скэффолдном сплитах модели пробивают оценённый потолок шума
(f > 1). Здесь проверяется объяснение: ошибка падает с числом обучающих молекул из того же
документа, и при большой поддержке уходит ниже потолка. Контроль на конфаунд — тот же градиент
внутри страт по размеру документа (размер и поддержка почти коллинеарны на случайном сплите,
скэффолдный сплит их разводит).

Берутся только тестовые молекулы с n_docs == 1: у них метка происходит из одного документа,
поэтому «поддержка» определена однозначно.

Вход : results/<prefix>/predictions или clean_*_cheap.pred.csv, data/<prefix>_clean/*.measurements.csv
Выход: results/document_support/{gradient,strata,paired}.csv
Пример: python scripts/14_document_support.py --cohorts full:data/full_clean:results/full_clean/predictions.csv.gz \
                                                         ext:data/ext_clean:results/ext
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BINS = [-1, 0, 4, 19, 99, np.inf]
LABELS = ["0", "1-4", "5-19", "20-99", "100+"]


def load_preds(src: str, pattern: str = "clean_*_cheap.pred.csv") -> pd.DataFrame:
    """src — либо один файл predictions.csv.gz, либо каталог с файлами по маске pattern."""
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, pattern)))
        if not files:
            raise SystemExit(f"нет {pattern} в {src}")
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return pd.read_csv(src)


def support_table(preds: pd.DataFrame, datadir: str, model: str, splits) -> pd.DataFrame:
    """Для каждой тестовой молекулы с n_docs==1: сколько молекул её документа попало в обучение."""
    out = []
    for task, p0 in preds[preds["model"] == model].groupby("task"):
        f = Path(datadir) / f"{task}.measurements.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f)
        # пары (молекула, документ): обучающая молекула поддерживает документ, если она в нём измерена,
        # даже если её относит к другому документу разбиение (сплит группирует по самому раннему)
        md = m[["mol_id", "doc_id"]].drop_duplicates()
        only_doc = m.groupby("mol_id")["doc_id"].first()   # для n_docs==1 это и есть её документ
        doc_size = md.groupby("doc_id")["mol_id"].nunique()
        for sp in splits:
            p_split = p0[p0["split"] == sp]
            p = p_split[p_split["n_docs"] == 1]
            for seed, g in p.groupby("seed"):
                # из обучения исключается ВЕСЬ тест этого сплита и зерна, а не только молекулы
                # с одним документом: иначе многодокументные тестовые молекулы считаются
                # поддержкой, и на сплите по документам доля нулевой поддержки падает с 82% до 59%
                test_ids = set(p_split.loc[p_split["seed"] == seed, "mol_id"])
                train_docs = md[~md.mol_id.isin(test_ids)].groupby("doc_id")["mol_id"].nunique()
                doc = g["mol_id"].map(only_doc)
                out.append(pd.DataFrame({
                    "task": task, "split": sp, "seed": seed, "mol_id": g["mol_id"].to_numpy(),
                    "doc_size": doc.map(doc_size).to_numpy(),
                    "support": doc.map(train_docs).fillna(0).to_numpy(),
                    "sq_err": (g["y"].to_numpy() - g["pred"].to_numpy()) ** 2}))
    if not out:
        raise SystemExit("нет данных: проверьте --cohorts и имя модели")
    return pd.concat(out, ignore_index=True)


def rmse(x):
    return float(np.sqrt(np.mean(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", required=True,
                    help="name:datadir:predictions_src, напр. full:data/full_clean:results/full_clean/predictions.csv.gz")
    ap.add_argument("--model", default="ecfp_gbm")
    ap.add_argument("--splits", nargs="+", default=["random", "scaffold", "temporal"])
    ap.add_argument("--pred-glob", default="clean_*_cheap.pred.csv",
                    help="маска файлов предсказаний, если --cohorts указывает на каталог")
    ap.add_argument("--out", default="results/document_support")
    a = ap.parse_args()

    frames = []
    for spec in a.cohorts:
        name, datadir, src = spec.split(":", 2)
        t = support_table(load_preds(src, a.pred_glob), datadir, a.model, a.splits)
        frames.append(t.assign(cohort=name))
        print(f"{name}: {t.task.nunique()} задач, {len(t)} тестовых молекул (n_docs=1)")
    D = pd.concat(frames, ignore_index=True)
    D["bin"] = pd.cut(D["support"], BINS, labels=LABELS)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    grad = (D.groupby(["split", "bin"], observed=True)
              .agg(rmse=("sq_err", rmse), n=("sq_err", "size"), tasks=("task", "nunique")).reset_index())
    grad.to_csv(out / "gradient.csv", index=False)
    print("\n== RMSE по поддержке документа ==")
    print(grad.pivot(index="bin", columns="split", values="rmse").round(3).to_string())
    print(grad.pivot(index="bin", columns="split", values="n").to_string())

    # контроль на размер документа: доля документа, попавшая в обучение, внутри страт по размеру
    D["frac"] = D["support"] / D["doc_size"]
    D["size_bin"] = pd.cut(D["doc_size"], [0, 49, 199, np.inf], labels=["1-49", "50-199", "200+"])
    D["frac_bin"] = pd.cut(D["frac"], [-0.01, 0.2, 0.5, 0.8, 1.01], labels=["<20%", "20-50%", "50-80%", ">80%"])
    strata = (D.groupby(["split", "size_bin", "frac_bin"], observed=True)
                .agg(rmse=("sq_err", rmse), n=("sq_err", "size")).reset_index())
    strata.to_csv(out / "strata.csv", index=False)
    print("\n== внутри страт по размеру документа (scaffold) ==")
    s = strata[strata.split == "scaffold"]
    print(s.pivot(index="size_bin", columns="frac_bin", values="rmse").round(3).to_string())

    # парный тест: одни и те же молекулы, поддержка есть (random) против нет (temporal)
    if {"random", "temporal"} <= set(a.splits):
        L = D[D["doc_size"] >= 100]
        g = L.groupby(["task", "mol_id", "split"]).agg(sq_err=("sq_err", "mean"),
                                                       support=("support", "mean")).reset_index()
        w = g.pivot_table(index=["task", "mol_id"], columns="split", values=["sq_err", "support"]).dropna()
        sel = w[(w[("support", "random")] >= 100) & (w[("support", "temporal")] == 0)]
        if len(sel) > 20:
            r_with, r_without = rmse(sel[("sq_err", "random")]), rmse(sel[("sq_err", "temporal")])
            pv = stats.wilcoxon(sel[("sq_err", "temporal")], sel[("sq_err", "random")]).pvalue
            pd.DataFrame([{"n_molecules": len(sel), "rmse_with_support": r_with,
                           "rmse_without_support": r_without, "wilcoxon_p": pv}]).to_csv(out / "paired.csv", index=False)
            print(f"\n== парный тест: {len(sel)} молекул из документов >=100 молекул ==")
            print(f"  с поддержкой (random, >=100)   RMSE {r_with:.3f}")
            print(f"  без поддержки (temporal, 0)    RMSE {r_without:.3f}   Wilcoxon p={pv:.2e}")
    print(f"\nзаписано в {out}/")


if __name__ == "__main__":
    main()
