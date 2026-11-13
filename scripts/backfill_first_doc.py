#!/usr/bin/env python
"""Добавляет столбец first_doc в уже построенные *.molecules.csv, не переизвлекая данные.

first_doc — документ молекулы с самым ранним годом (ties разрешаются по doc_id), ровно как в
scripts/02_standardize.py. Нужен для сплита по документам (ncmol.splits.document_split).
Пример: python scripts/backfill_first_doc.py data/full_clean data/ext_clean
"""
import sys
from pathlib import Path

import pandas as pd


def backfill(d: Path) -> None:
    for f in sorted(d.glob("*.molecules.csv")):
        task = f.name[: -len(".molecules.csv")]
        meas = d / f"{task}.measurements.csv"
        if not meas.exists():
            print(f"{task}: нет measurements.csv, пропуск")
            continue
        mols = pd.read_csv(f)
        if "first_doc" in mols.columns:
            print(f"{task}: уже есть")
            continue
        m = pd.read_csv(meas)
        pair = m.groupby(["mol_id", "doc_id"], as_index=False).agg(year=("year", "first"))
        pair = pair.sort_values(["mol_id", "year", "doc_id"], kind="stable")
        first = pair.groupby("mol_id")["doc_id"].first()
        mols["first_doc"] = mols["mol_id"].map(first)
        miss = int(mols["first_doc"].isna().sum())
        mols.to_csv(f, index=False)
        print(f"{task}: добавлено, без документа {miss} из {len(mols)}")


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["data/full_clean", "data/ext_clean"]
    for d in dirs:
        print(f"== {d}")
        backfill(Path(d))
