#!/usr/bin/env python
"""Step 2c. Data validity audit per task (target x measurement type). Data description only,
no conclusions about models. Input: data/processed/tasks.csv and <task>.measurements.csv (scripts/02),
optionally data/raw/CHEMBL*.csv (for data_validity_comment).

Output results/audit/:
  summary.csv             — per task: molecules/measurements/documents/assays, fraction of molecules with >=2 documents,
                            sigma with and without the |delta| filter, relation of |delta| to year difference and to activity level
  delta_distribution.csv  — quantiles of |delta| between documents (kind=between_docs) and between assays of the SAME document
                            (kind=within_doc); fractions of pairs with |delta|>=1,2,3
  big_pairs.csv           — document pairs with |delta| >= --big-delta (default 2): documents, assays, years, values
  doc_shift.csv           — shift of a document relative to the consensus of the other documents (leave-one-out)
  assay_shift.csv         — same at the assay_chembl_id level + number of shared molecules; fraction of pairs with the same assay_id
  type_compare.csv        — Ki vs IC50 for one target: molecules measured in both types, mean pIC50-pKi shift
  duplicates.csv          — stereoisomers/variants: one connectivity (InChIKey14 = mol_id) with >1 full InChIKey
  not_checked.csv         — what the audit does NOT check and why
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
    ("shared authors of documents", "raw/processed contain no authors or affiliations (available via the ChEMBL document resource); independence "
     "of documents by authors/laboratories is not checked — weakens the assumption of independent errors"),
    ("assay conditions (cells/enzyme, organism, substrate, ATP concentration, description)", "RAW_COLUMNS of scripts/01 has no assay_description / "
     "assay_organism / assay_cell_type / assay_category; comparability is checked only indirectly (assay shifts, assay_id)"),
    ("overlap of documents (the same experiment, republished data)", "not checked; ChEMBL may contain data "
     "carried over from paper to paper, which underestimates the noise"),
    ("correctness of units and orders of magnitude in the source documents", "checked only indirectly via large |delta|; manual audit of pairs from "
     "big_pairs.csv has not been performed"),
    ("tautomers and other duplicates beyond InChIKey14", "standardization in scripts/02 does not normalize tautomers; not checked"),
    ("spread via standard deviations within a document", "the data has one value per (molecule, document); within-document "
     "replicates are averaged in scripts/02"),
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
    """All document pairs within a molecule. d: mol_id, doc_id, y, year, assays(str)."""
    rows = []
    for mol, g in d.groupby("mol_id"):
        if len(g) < 2:
            continue
        for (_, a), (_, b) in combinations(g.iterrows(), 2):
            rows.append((mol, a.doc_id, b.doc_id, a.assays, b.assays, a.year, b.year, a.y, b.y))
    return pd.DataFrame(rows, columns=["mol_id", "doc_a", "doc_b", "assay_a", "assay_b", "year_a", "year_b", "y_a", "y_b"])


def loo_shift(d: pd.DataFrame, key: str) -> pd.DataFrame:
    """Shift of unit key (doc_id or assay_id) = mean of (y_unit - mean of the other units of the same molecule).
    d: mol_id, key, y (one row per molecule x unit)."""
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
        for tag, mk in (("le3", pr["dyear"] <= 3), ("gt3", pr["dyear"] > 3)):  # sigma from pairs: sqrt(mean(delta^2)/2)
            res[f"sigma_pairs_dyear_{tag}"] = float(np.sqrt((pr.loc[mk, "delta"] ** 2).mean() / 2)) if mk.sum() else np.nan
            res[f"n_pairs_dyear_{tag}"] = int(mk.sum())
        mean_y = pr[["y_a", "y_b"]].mean(axis=1)
        res["spearman_absdelta_vs_meany"] = float(spearmanr(pr["abs_delta"], mean_y)[0]) if len(pr) > 2 else np.nan
        res["frac_pairs_same_assay_id"] = float((pr["assay_a"] == pr["assay_b"]).mean())
        big = pr[pr["abs_delta"] >= big_delta].copy()
        big.insert(0, "task", task)
        big["ge3"] = big["abs_delta"] >= 3
        big = big.merge(m.drop_duplicates("mol_id")[["mol_id", "smiles"]], on="mol_id", how="left")
    # within a document between different assays
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
    # stereo variants: one mol_id (InChIKey14), different full InChIKeys
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
    ap.add_argument("--big-delta", type=float, default=2.0, help="|delta| threshold for the list of suspicious pairs (2-3)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = pd.read_csv(Path(a.processed) / "tasks.csv")
    if "target" not in tasks.columns:  # 02b не сохраняет target/type: task = "<CHEMBL_ID>_<тип>"
        tasks["target"] = tasks["task"].str.rsplit("_", n=1).str[0]
        tasks["type"] = tasks["task"].str.rsplit("_", n=1).str[1]
    S, D, B, DS, AS, DUP, docs = [], [], [], [], [], [], {}
    for r in tasks.itertuples():
        f = Path(a.processed) / f"{r.task}.measurements.csv"
        if not f.exists():
            continue
        res, dist, big, ds, as_, d, dup = audit_task(r.task, pd.read_csv(f), a.big_delta)
        # data_validity_comment from raw (if present)
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

    # Ki vs IC50 on one target: molecules measured in both types
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
    print(f"audit written to {out}")


if __name__ == "__main__":
    main()
