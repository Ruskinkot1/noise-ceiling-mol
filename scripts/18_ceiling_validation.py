#!/usr/bin/env python
"""Шаг 18. Проверка оценки потолка на настоящих данных: покрытие, однородность, охват.

Раздел 4 проверял оцениватель одним синтетическим прогоном. Здесь три проверки на
самих измерениях.

 (i)   Покрытие. У молекул с тремя и более документами по очереди прячется одно
       измерение; по оставшимся n−1 строится интервал полушириной z·σ·√(1+1/(n−1)),
       проверяется попадание. Доли попаданий сравниваются с номинальными 50/80/95 %.
       Дополнительно считается σ, построенная только по молекулам ровно с двумя
       документами, то есть не видевшая проверочный материал.
 (ii)  Однородность. Допущение «одна дисперсия шума на задачу» проверяется сравнением
       σ на молекулах с двумя документами и с тремя и более, парно по задачам.
 (iii) Охват. Доля документов, не делящих ни одной молекулы ни с кем: они поставляют
       тестовые метки, но в оценку шума не входят. Для остальных — связь модуля
       междокументной разности с числом партнёров документа.

Выход: results/<набор>/ceiling_validation.csv.

Запуск: python scripts/18_ceiling_validation.py [--results results/all_clean]
        [--processed data/all_clean] [--cohorts full:data/full_clean ext:data/ext_clean]
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, wilcoxon

MAX_SPAN = 3.0
LEVELS = (0.50, 0.80, 0.95)


def doc_level(processed, task):
    m = pd.read_csv(processed / f"{task}.measurements.csv")
    return m.groupby(["mol_id", "doc_id"], as_index=False).y.mean()


def trimmed(d, min_docs):
    n = d.groupby("mol_id").y.transform("size")
    d = d[n >= min_docs]
    if d.empty:
        return d
    span = d.groupby("mol_id").y.transform(lambda s: s.max() - s.min())
    return d[span <= MAX_SPAN]


def pooled_sigma(d):
    """σ по объединённой внутримолекулярной дисперсии на уже отобранном наборе."""
    if d.empty:
        return np.nan
    dev = d.y - d.groupby("mol_id").y.transform("mean")
    dof = (d.groupby("mol_id").size() - 1).sum()
    return float(np.sqrt((dev**2).sum() / dof)) if dof else np.nan


def loo_z(d3, sigma):
    """Стандартизованные остатки по спрятанным измерениям молекул с n >= 3."""
    out = []
    for _, g in d3.groupby("mol_id"):
        y = g.y.to_numpy()
        n = len(y)
        for i in range(n):
            rest = np.delete(y, i)
            se = sigma * np.sqrt(1 + 1 / (n - 1))
            out.append((y[i] - rest.mean()) / se)
    return np.asarray(out)


def coverage_rows(label, z, n_mols):
    rows = [(f"coverage_{label}", f"level_{int(L*100)}",
             float((np.abs(z) <= norm.ppf(0.5 + L / 2)).mean()), len(z)) for L in LEVELS]
    rows.append((f"coverage_{label}", "sd_z", float(z.std(ddof=1)), len(z)))
    rows.append((f"coverage_{label}", "share_abs_z_over_3", float((np.abs(z) > 3).mean()), len(z)))
    rows.append((f"coverage_{label}", "molecules", float(n_mols), len(z)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_clean")
    ap.add_argument("--processed", default="data/all_clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cohorts", nargs="*", default=["full:data/full_clean", "ext:data/ext_clean"])
    a = ap.parse_args()
    R, P = Path(a.results), Path(a.processed)
    ceil = pd.read_csv(R / "ceilings.csv").set_index("task")
    tasks = sorted(ceil.index)
    cohort_of = {}
    for spec in a.cohorts:
        name, path = spec.split(":", 1)
        for f in Path(path).glob("*.measurements.csv"):
            cohort_of[f.name.replace(".measurements.csv", "")] = name

    rng = np.random.default_rng(a.seed)
    z_all, z_oos, z_syn, by_cohort = [], [], [], {}
    n_mols3 = 0
    sig2, sig3, n_iso, n_doc, n_mol_iso, n_mol_all = [], [], 0, 0, 0, 0
    conn_rows = []
    for t in tasks:
        d = doc_level(P, t)
        n_doc += d.doc_id.nunique()
        n_mol_all += d.mol_id.nunique()
        multi = d[d.groupby("mol_id").y.transform("size") >= 2]
        linked = set(multi.doc_id)
        n_iso += d.doc_id.nunique() - len(linked)
        # молекула из изолированного документа по определению встречается только в нём:
        # иначе второй её документ делил бы с ним молекулу и оба были бы связными
        n_mol_iso += d[~d.doc_id.isin(linked)].mol_id.nunique()

        dt = trimmed(d, 2)
        if dt.empty:
            continue
        n = dt.groupby("mol_id").y.transform("size")
        d2, d3 = dt[n == 2], dt[n >= 3]
        s2, s3 = pooled_sigma(d2), pooled_sigma(d3)
        if np.isfinite(s2) and np.isfinite(s3):
            sig2.append(s2)
            sig3.append(s3)
        if d3.empty:
            continue
        n_mols3 += d3.mol_id.nunique()
        z = loo_z(d3, float(ceil.sigma[t]))
        z_all.append(z)
        by_cohort.setdefault(cohort_of.get(t, "?"), []).append(z)
        if np.isfinite(s2):
            z_oos.append(loo_z(d3, s2))
        # контроль машинерии: та же структура молекул, шум ровно σ по построению
        syn = d3.copy()
        syn["y"] = (d3.groupby("mol_id").y.transform("mean")
                    + rng.normal(0, float(ceil.sigma[t]), len(d3)))
        z_syn.append(loo_z(syn, float(ceil.sigma[t])))

        # связность: модуль междокументной разности против числа партнёров документа
        partners = {}
        for _, g in multi.groupby("mol_id"):
            for da, db in itertools.combinations(sorted(g.doc_id), 2):
                partners.setdefault(da, set()).add(db)
                partners.setdefault(db, set()).add(da)
        for _, g in trimmed(d, 2).groupby("mol_id"):
            g = g.sort_values("doc_id")
            for (_, x), (_, y) in itertools.combinations(g.iterrows(), 2):
                conn_rows.append((abs(x.y - y.y),
                                  min(len(partners.get(x.doc_id, ())), len(partners.get(y.doc_id, ())))))

    z = np.concatenate(z_all)
    rows = coverage_rows("published_sigma", z, n_mols3)
    rows += coverage_rows("out_of_sample_sigma", np.concatenate(z_oos), n_mols3)
    rows += coverage_rows("synthetic_control", np.concatenate(z_syn), n_mols3)
    for name, zs in sorted(by_cohort.items()):
        rows += coverage_rows(f"cohort_{name}", np.concatenate(zs), np.nan)

    sig2, sig3 = np.array(sig2), np.array(sig3)
    w = wilcoxon(sig3, sig2)
    rows += [("homogeneity", "median_sigma_2_docs", float(np.median(sig2)), len(sig2)),
             ("homogeneity", "median_sigma_3plus_docs", float(np.median(sig3)), len(sig3)),
             ("homogeneity", "tasks_higher_with_3plus", float((sig3 > sig2).sum()), len(sig2)),
             ("homogeneity", "wilcoxon_p", float(w.pvalue), len(sig2))]

    cr = pd.DataFrame(conn_rows, columns=["abs_delta", "partners"])
    sp = spearmanr(cr.abs_delta, cr.partners)
    rows += [("coverage_of_material", "documents", float(n_doc), np.nan),
             ("coverage_of_material", "isolated_documents_share", n_iso / n_doc, n_doc),
             ("coverage_of_material", "molecules", float(n_mol_all), np.nan),
             ("coverage_of_material", "molecules_only_in_isolated_share", n_mol_iso / n_mol_all, n_mol_all),
             ("coverage_of_material", "rho_abs_delta_partners", float(sp.statistic), len(cr)),
             ("coverage_of_material", "rho_p", float(sp.pvalue), len(cr))]

    out = pd.DataFrame(rows, columns=["group", "quantity", "value", "n"])
    out.to_csv(R / "ceiling_validation.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
