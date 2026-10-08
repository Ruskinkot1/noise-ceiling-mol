#!/usr/bin/env python
"""Шаг 2. Стандартизация структур, единицы (pChEMBL уже в -log10 M), пометка источников.

Выход для каждой задачи (мишень x тип измерения):
  data/processed/<target>_<type>.measurements.csv : mol_id, smiles, doc_id, assay_id, year, y   (по измерению)
  data/processed/<target>_<type>.molecules.csv    : mol_id, smiles, y (среднее по документам), n_docs, first_year
  data/processed/tasks.csv                        : сводка и флаг passes (>= min_replicated_mols)
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from ncmol.chem import standardize

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw"))
    ap.add_argument("--out", default=str(ROOT / "data/processed"))
    ap.add_argument("--config", default=str(ROOT / "configs/endpoints.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for f in sorted(Path(a.raw).glob("CHEMBL*.csv")):
        raw = pd.read_csv(f)
        if raw.empty:
            continue
        smap = {s: standardize(s) for s in raw["canonical_smiles"].unique()}
        raw["smiles"] = raw["canonical_smiles"].map(lambda s: smap[s][0])
        raw["mol_id"] = raw["canonical_smiles"].map(lambda s: smap[s][1])
        raw = raw.dropna(subset=["smiles", "mol_id"])
        for stype, g in raw.groupby("standard_type"):
            if stype not in cfg["standard_types"]:
                continue
            m = pd.DataFrame({"mol_id": g["mol_id"], "smiles": g["smiles"], "doc_id": g["doc_chembl_id"],
                              "assay_id": g["assay_chembl_id"], "year": g["doc_year"], "y": g["pchembl_value"]})
            # одно значение на (молекула, документ): среднее по повторам внутри документа
            d = m.groupby(["mol_id", "doc_id"], as_index=False).agg(
                smiles=("smiles", "first"), year=("year", "first"), y=("y", "mean"))
            mols = d.groupby("mol_id").agg(smiles=("smiles", "first"), y=("y", "mean"),
                                           n_docs=("doc_id", "nunique"), first_year=("year", "min")).reset_index()
            task = f"{f.stem}_{stype}"
            m.to_csv(out / f"{task}.measurements.csv", index=False)
            mols.to_csv(out / f"{task}.molecules.csv", index=False)
            n_rep = int((mols["n_docs"] >= cfg["min_docs"]).sum())
            summary.append({"task": task, "target": f.stem, "type": stype, "n_mols": len(mols),
                            "n_rep_mols": n_rep, "n_docs": d["doc_id"].nunique(),
                            "passes": n_rep >= cfg["min_replicated_mols"]})
    s = pd.DataFrame(summary)
    s.to_csv(out / "tasks.csv", index=False)
    print(s.to_string(index=False))
    print(f"\nзадач проходит порог: {int(s['passes'].sum())} (нужно >= 15, критерий остановки нед. 3)")


if __name__ == "__main__":
    main()
