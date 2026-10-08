"""Pilot collector: molecules with replicate measurements from independent ChEMBL documents.

Usage:
  python collect_replicates.py --targets CHEMBL203 CHEMBL240 --types Ki IC50 --min-year 2023 \
      --exclude seen_inchikeys.txt --out replicates.csv

--exclude: optional text file, one InChIKey per line (full or first 14 chars),
molecules already present in benchmarks / pretraining corpora are dropped.
"""
import argparse
import sys
import time

import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
API = "https://www.ebi.ac.uk/chembl/api/data"
FIELDS = ["molecule_chembl_id", "canonical_smiles", "standard_type", "standard_relation",
          "pchembl_value", "document_chembl_id", "document_year", "assay_chembl_id",
          "target_chembl_id", "data_validity_comment"]


def fetch(target, std_type):
    url = f"{API}/activity.json"
    params = {"target_chembl_id": target, "standard_type": std_type, "standard_relation": "=",
              "pchembl_value__isnull": "false", "assay_type": "B", "limit": 1000}
    rows = []
    while url:
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    sys.exit(f"ChEMBL request failed: {e}")
                time.sleep(3)
        data = r.json()
        rows += [{k: a.get(k) for k in FIELDS} for a in data["activities"]]
        nxt = data["page_meta"]["next"]
        url = f"https://www.ebi.ac.uk{nxt}" if nxt else None
        params = None
    return rows


def standardize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["CHEMBL203", "CHEMBL240", "CHEMBL220"])
    ap.add_argument("--types", nargs="+", default=["Ki", "IC50"])
    ap.add_argument("--min-year", type=int, default=2023)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--out", default="replicates.csv")
    a = ap.parse_args()

    rows = []
    for t in a.targets:
        for st in a.types:
            got = fetch(t, st)
            print(f"{t} {st}: {len(got)} records", flush=True)
            rows += got
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no records")
    n0 = len(df)
    df = df[df["data_validity_comment"].isna()].dropna(subset=["canonical_smiles", "document_year"])
    df["pchembl_value"] = df["pchembl_value"].astype(float)
    df = df[df["document_year"].astype(int) >= a.min_year]
    print(f"after validity + year>={a.min_year}: {len(df)} of {n0}")

    std = df["canonical_smiles"].map(standardize)
    df["smiles"] = std.map(lambda x: x[0])
    df["inchikey"] = std.map(lambda x: x[1])
    df = df.dropna(subset=["inchikey"])

    if a.exclude:
        seen = {l.strip() for l in open(a.exclude) if l.strip()}
        seen14 = {s[:14] for s in seen}
        before = len(df)
        df = df[~df["inchikey"].isin(seen) & ~df["inchikey"].str[:14].isin(seen14)]
        print(f"excluded as already seen: {before - len(df)}")

    # one value per (molecule, target, type, document), then keep molecules seen in >=2 documents
    key = ["inchikey", "target_chembl_id", "standard_type"]
    per_doc = (df.groupby(key + ["document_chembl_id"], as_index=False)
                 .agg(pchembl=("pchembl_value", "mean"), year=("document_year", "first"),
                      smiles=("smiles", "first")))
    n_docs = per_doc.groupby(key)["document_chembl_id"].transform("nunique")
    rep = per_doc[n_docs >= 2].sort_values(key)
    rep.to_csv(a.out, index=False)

    g = rep.groupby(key)["pchembl"]
    summary = (rep.groupby(["target_chembl_id", "standard_type"])
                  .agg(molecules=("inchikey", "nunique"), documents=("document_chembl_id", "nunique")))
    summary["mean_abs_diff"] = (rep.assign(d=rep["pchembl"] - g.transform("mean"))
                                   .assign(d=lambda x: x["d"].abs())
                                   .groupby(["target_chembl_id", "standard_type"])["d"].mean())
    print("\nreplicates per endpoint (molecules with >=2 independent documents):")
    print(summary.to_string())
    print(f"\nsaved {len(rep)} rows to {a.out}")


if __name__ == "__main__":
    main()
