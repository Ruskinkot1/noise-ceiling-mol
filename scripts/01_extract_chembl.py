#!/usr/bin/env python
"""Шаг 1. Выгрузка измерений из локального SQLite ChEMBL.

  python scripts/01_extract_chembl.py --db data/chembl/chembl_35.db --discover --top 60 --min-mols 100
  python scripts/01_extract_chembl.py --db data/chembl/chembl_35.db --config configs/endpoints.yaml
  python scripts/01_extract_chembl.py --db ... --targets CHEMBL203 CHEMBL240
"""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/endpoints.yaml"))
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--out", default=str(ROOT / "data/raw"))
    ap.add_argument("--discover", action="store_true", help="ранжировать мишени по числу молекул с >=2 документами")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--min-mols", type=int, default=100)
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)

    if a.discover:
        q = (ROOT / "sql/discover_targets.sql").read_text()
        df = pd.read_sql_query(q, con, params={"min_mols": a.min_mols, "top": a.top})
        Path(a.out).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(a.out) / "discovered_targets.csv", index=False)
        print(df.to_string(index=False))
        return

    cfg = yaml.safe_load(open(a.config))
    targets = a.targets or [t["id"] for t in cfg["chembl_candidates"]]
    q = (ROOT / "sql/replicates.sql").read_text()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for t in targets:
        df = pd.read_sql_query(q, con, params={"target": t})
        df.to_csv(Path(a.out) / f"{t}.csv", index=False)
        name = df["target_name"].iloc[0] if len(df) else "-"
        print(f"{t}\t{name}\trows={len(df)}\tdocs={df['doc_chembl_id'].nunique() if len(df) else 0}")


if __name__ == "__main__":
    main()
