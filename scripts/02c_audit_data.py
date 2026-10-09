#!/usr/bin/env python
"""Шаг 2c. Аудит валидности данных по задачам (мишень x тип измерения). Только описание данных,
никаких выводов о моделях. Вход: data/processed/tasks.csv и <task>.measurements.csv (scripts/02),
необязательно data/raw/CHEMBL*.csv (для data_validity_comment).

Выход results/audit/:
  summary.csv             — по задаче: молекулы/измерения/документы/ассеи, доля молекул с >=2 документами,
                            sigma с фильтром |Δ| и без, связь |Δ| с разницей лет и с уровнем активности
  delta_distribution.csv  — квантили |Δ| между документами (kind=between_docs) и между ассеями ОДНОГО документа
                            (kind=within_doc); доли пар с |Δ|>=1,2,3
  big_pairs.csv           — пары документов с |Δ| >= --big-delta (по умолчанию 2): документы, ассеи, годы, значения
  doc_shift.csv           — сдвиг документа относительно консенсуса остальных документов (leave-one-out)
  assay_shift.csv         — то же на уровне assay_chembl_id + число общих молекул; доля пар с одинаковым assay_id
  type_compare.csv        — Ki vs IC50 для одной мишени: молекулы, измеренные в обоих типах, средний сдвиг pIC50-pKi
  duplicates.csv          — стереоизомеры/варианты: одна связность (InChIKey14 = mol_id) с >1 полным InChIKey
  not_checked.csv         — что в аудите НЕ проверено и почему
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr

from ncmol.ceiling import doc_level, pooled_variance

ROOT = Path(__file__).resolve().parents[1]

NOT_CHECKED = [
    ("общие авторы документов", "в raw/processed нет авторов и аффилиаций (доступны через ресурс document ChEMBL); независимость "
     "документов по авторам/лабораториям не проверена — ослабляет допущение независимости ошибок"),
    ("условия ассея (клетки/фермент, организм, субстрат, концентрация АТФ, описание)", "в RAW_COLUMNS scripts/01 нет assay_description / "
     "assay_organism / assay_cell_type / assay_category; сопоставимость проверена только косвенно (сдвиги ассеев, assay_id)"),
    ("перекрытие документов (один и тот же эксперимент, перепубликованные данные)", "не проверено; ChEMBL может содержать данные, "
     "перенесённые из статьи в статью, что занижает оценку шума"),
    ("точность единиц и порядков величины в исходных документах", "проверяются только косвенно большими |Δ|; ручной аудит пар из "
     "big_pairs.csv не выполнен"),
    ("таутомеры и прочие дубли помимо InChIKey14", "стандартизация scripts/02 не нормализует таутомеры; не проверено"),
    ("разброс по стандартным отклонениям внутри документа", "в данных одно значение на (молекула, документ), внутридокументные "
     "повторы усреднены в scripts/02"),
]


def full_inchikey(smiles: str):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)
    except Exception:
        return None


def pairs_of_docs(d: pd.DataFrame) -> pd.DataFrame:
    """Все пары документов внутри молекулы. d: mol_id, doc_id, y, year, assays(str)."""
    rows = []
    for mol, g in d.groupby("mol_id"):
        if len(g) < 2:
            continue
        for (_, a), (_, b) in combinations(g.iterrows(), 2):
            rows.append((mol, a.doc_id, b.doc_id, a.assays, b.assays, a.year, b.year, a.y, b.y))
    return pd.DataFrame(rows, columns=["mol_id", "doc_a", "doc_b", "assay_a", "assay_b", "year_a", "year_b", "y_a", "y_b"])


def loo_shift(d: pd.DataFrame, key: str) -> pd.DataFrame:
    """Сдвиг единицы key (doc_id или assay_id) = среднее (y_ед. - среднее остальных единиц той же молекулы).
    d: mol_id, key, y (одна строка на молекулу x единицу)."""
    g = d.groupby("mol_id")["y"]
    n = g.transform("size")
    d = d[n >= 2].copy()
    s = d.groupby("mol_id")["y"].transform("sum")
    n = d.groupby("mol_id")["y"].transform("size")
    d["resid"] = d["y"] - (s - d["y"]) / (n - 1)
    out = d.groupby(key).agg(n_overlap=("resid", "size"), mean_shift=("resid", "mean"), sd=("resid", "std")).reset_index()
    out["se"] = out["sd"] / np.sqrt(out["n_overlap"])
    out["z"] = out["mean_shift"] / out["se"]
    out["flag_shift"] = (out["n_overlap"] >= 5) & (out["z"].abs() > 3)
    return out


def qrow(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n_pairs": 0}
    q = np.quantile(x, [0.5, 0.75, 0.9, 0.95, 0.99])
    return {"n_pairs": len(x), "q50": q[0], "q75": q[1], "q90": q[2], "q95": q[3], "q99": q[4], "max": x.max(),
            "frac_ge1": (x >= 1).mean(), "frac_ge2": (x >= 2).mean(), "frac_ge3": (x >= 3).mean()}


def audit_task(task, m: pd.DataFrame, big_delta: float):
    m = m.dropna(subset=["y"]).copy()
    m["assay_id"] = m["assay_id"].astype(str)
    d = doc_level(m)
    meta = m.groupby(["mol_id", "doc_id"]).agg(year=("year", "first"), assays=("assay_id", lambda s: ";".join(sorted(set(s))))).reset_index()
    d = d.merge(meta, on=["mol_id", "doc_id"])
    nd = d.groupby("mol_id").size()
    res = {"task": task, "n_mols": len(nd), "n_meas": len(m), "n_docs": d["doc_id"].nunique(),
           "n_assays": m["assay_id"].nunique(), "n_mols_ge2docs": int((nd >= 2).sum()), "frac_mols_ge2docs": float((nd >= 2).mean()),
           "n_mols_ge3docs": int((nd >= 3).sum()), "year_min": m["year"].min(), "year_max": m["year"].max()}
    top = d.groupby("doc_id").size().sort_values(ascending=False)
    res["share_mols_in_top_doc"] = float(top.iloc[0] / len(nd)) if len(top) else np.nan
    for tag, thr in (("nofilter", None), ("diff3", 3.0), ("diff2", 2.0)):
        s2, nmol, _ = pooled_variance(d[["mol_id", "doc_id", "y"]], 2, thr)
        res[f"sigma_{tag}"] = float(np.sqrt(s2)) if nmol else np.nan
        res[f"n_rep_mols_{tag}"] = nmol
    pr = pairs_of_docs(d)
    dist, big = [], pd.DataFrame()
    if len(pr):
        pr["delta"] = pr["y_a"] - pr["y_b"]
        pr["abs_delta"] = pr["delta"].abs()
        pr["dyear"] = (pr["year_a"] - pr["year_b"]).abs()
        dist.append({"task": task, "kind": "between_docs", **qrow(pr["abs_delta"].to_numpy())})
        if pr["dyear"].nunique() > 1:
            res["spearman_absdelta_vs_dyear"] = float(spearmanr(pr["abs_delta"], pr["dyear"])[0])
        for tag, mk in (("le3", pr["dyear"] <= 3), ("gt3", pr["dyear"] > 3)):  # sigma по парам: sqrt(mean(Δ²)/2)
            res[f"sigma_pairs_dyear_{tag}"] = float(np.sqrt((pr.loc[mk, "delta"] ** 2).mean() / 2)) if mk.sum() else np.nan
            res[f"n_pairs_dyear_{tag}"] = int(mk.sum())
        mean_y = pr[["y_a", "y_b"]].mean(axis=1)
        res["spearman_absdelta_vs_meany"] = float(spearmanr(pr["abs_delta"], mean_y)[0]) if len(pr) > 2 else np.nan
        res["frac_pairs_same_assay_id"] = float((pr["assay_a"] == pr["assay_b"]).mean())
        big = pr[pr["abs_delta"] >= big_delta].copy()
        big.insert(0, "task", task)
        big["ge3"] = big["abs_delta"] >= 3
        big = big.merge(m.drop_duplicates("mol_id")[["mol_id", "smiles"]], on="mol_id", how="left")
    # внутри документа между разными ассеями
    da = m.groupby(["mol_id", "doc_id", "assay_id"], as_index=False)["y"].mean()
    wd = []
    for (_, _), g in da.groupby(["mol_id", "doc_id"]):
        if len(g) >= 2:
            wd += [abs(a - b) for a, b in combinations(g["y"].to_numpy(), 2)]
    dist.append({"task": task, "kind": "within_doc", **qrow(np.array(wd))})
    ds = loo_shift(d[["mol_id", "doc_id", "y"]], "doc_id")
    ds.insert(0, "task", task)
    res["n_docs_flag_shift"] = int(ds["flag_shift"].sum()) if len(ds) else 0
    res["n_docs_in_shift"] = len(ds)
    a_ = da.groupby(["mol_id", "assay_id"], as_index=False)["y"].mean()
    as_ = loo_shift(a_, "assay_id")
    as_.insert(0, "task", task)
    a2d = m.drop_duplicates("assay_id").set_index("assay_id")["doc_id"]
    as_["doc_id"] = as_["assay_id"].map(a2d)
    res["assays_per_doc_mean"] = float(m.groupby("doc_id")["assay_id"].nunique().mean())
    # стереоварианты: один mol_id (InChIKey14), разные полные InChIKey
    keys = {s: full_inchikey(s) for s in m["smiles"].unique()}
    m["fullkey"] = m["smiles"].map(keys)
    nk = m.groupby("mol_id")["fullkey"].nunique()
    res["n_mols_multi_fullkey"] = int((nk > 1).sum())
    res["frac_mols_multi_fullkey"] = float((nk > 1).mean())
    dup = {"task": task, "n_mols": len(nk), "n_mols_multi_fullkey": int((nk > 1).sum())}
    if (nk > 1).any() and len(pr):
        dk = m.groupby(["mol_id", "doc_id"])["fullkey"].agg(lambda s: ";".join(sorted(set(s.dropna()))))
        pr["same_variant"] = [dk.get((mm, a)) == dk.get((mm, b)) for mm, a, b in zip(pr["mol_id"], pr["doc_a"], pr["doc_b"])]
        multi = pr[pr["mol_id"].isin(nk[nk > 1].index)]
        for flag, name in ((True, "same_variant"), (False, "diff_variant")):
            x = multi.loc[multi["same_variant"] == flag, "abs_delta"].to_numpy()
            dup[f"n_pairs_{name}"] = len(x)
            dup[f"mean_absdelta_{name}"] = float(x.mean()) if len(x) else np.nan
            dup[f"frac_ge1_{name}"] = float((x >= 1).mean()) if len(x) else np.nan
    return res, pd.DataFrame(dist), big, ds, as_, d, dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=str(ROOT / "data/processed"))
    ap.add_argument("--raw", default=str(ROOT / "data/raw"))
    ap.add_argument("--out", default=str(ROOT / "results/audit"))
    ap.add_argument("--big-delta", type=float, default=2.0, help="порог |Δ| для списка подозрительных пар (2–3)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = pd.read_csv(Path(a.processed) / "tasks.csv")
    S, D, B, DS, AS, DUP, docs = [], [], [], [], [], [], {}
    for r in tasks.itertuples():
        f = Path(a.processed) / f"{r.task}.measurements.csv"
        if not f.exists():
            continue
        res, dist, big, ds, as_, d, dup = audit_task(r.task, pd.read_csv(f), a.big_delta)
        # data_validity_comment из raw (если есть)
        rawf = Path(a.raw) / f"{r.target}.csv"
        if rawf.exists():
            raw = pd.read_csv(rawf)
            raw = raw[raw["standard_type"] == r.type]
            res["n_raw_rows"] = len(raw)
            res["n_raw_with_validity_comment"] = int(raw["data_validity_comment"].notna().sum()) if "data_validity_comment" in raw else np.nan
        S.append(res); D.append(dist); B.append(big); DS.append(ds); AS.append(as_); DUP.append(dup); docs[r.task] = (r, d)
        print(f"{r.task}: mols={res['n_mols']} >=2docs={res['frac_mols_ge2docs']:.2f} sigma(no filter)={res['sigma_nofilter']:.2f} "
              f"flag_docs={res['n_docs_flag_shift']}/{res['n_docs_in_shift']}")
    pd.DataFrame(S).to_csv(out / "summary.csv", index=False)
    pd.concat(D).to_csv(out / "delta_distribution.csv", index=False)
    pd.concat(B).to_csv(out / "big_pairs.csv", index=False)
    pd.concat(DS).to_csv(out / "doc_shift.csv", index=False)
    pd.concat(AS).to_csv(out / "assay_shift.csv", index=False)
    pd.DataFrame(DUP).to_csv(out / "duplicates.csv", index=False)

    # Ki vs IC50 на одной мишени: молекулы, измеренные в обоих типах
    rows = []
    for tgt, g in tasks.groupby("target"):
        if {"Ki", "IC50"} <= set(g["type"]):
            tk, ti = g[g["type"] == "Ki"]["task"].iloc[0], g[g["type"] == "IC50"]["task"].iloc[0]
            if tk in docs and ti in docs:
                k = docs[tk][1].groupby("mol_id")["y"].mean()
                i = docs[ti][1].groupby("mol_id")["y"].mean()
                j = pd.concat([k.rename("pKi"), i.rename("pIC50")], axis=1, join="inner")
                row = {"target": tgt, "n_mols_both": len(j)}
                if len(j) >= 5:
                    dlt = j["pIC50"] - j["pKi"]
                    row.update(mean_pIC50_minus_pKi=dlt.mean(), sd_diff=dlt.std(), median_diff=dlt.median(),
                               pearson=j.corr().iloc[0, 1], spearman=float(spearmanr(j["pKi"], j["pIC50"])[0]))
                s = pd.DataFrame(S).set_index("task")
                row["sigma_Ki"], row["sigma_IC50"] = s.loc[tk, "sigma_nofilter"], s.loc[ti, "sigma_nofilter"]
                rows.append(row)
    pd.DataFrame(rows, columns=["target", "n_mols_both", "mean_pIC50_minus_pKi", "sd_diff", "median_diff", "pearson", "spearman",
                                "sigma_Ki", "sigma_IC50"]).to_csv(out / "type_compare.csv", index=False)
    pd.DataFrame(NOT_CHECKED, columns=["item", "reason"]).to_csv(out / "not_checked.csv", index=False)
    print(f"аудит записан в {out}")


if __name__ == "__main__":
    main()
