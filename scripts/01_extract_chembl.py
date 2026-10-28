#!/usr/bin/env python
"""Step 1. Extraction of measurements from the ChEMBL REST API (the 5.7 GB SQLite dump is not used).

  python scripts/01_extract_chembl.py --targets CHEMBL203 CHEMBL240
  python scripts/01_extract_chembl.py                      # all candidates from configs/endpoints.yaml

Filters (API side): standard_relation '=', pchembl_value not null, data_validity_comment empty,
assay_type B, standard_type in (Ki, IC50). The API has no confidence_score filter on activity, so
it is applied via a separate query to the assay resource (confidence_score__gte=8, assay_type=B, same target):
the set of assay_chembl_id is taken, and only activities from those assays are kept. Additionally,
potential_duplicate == 1 is discarded. If the API is unavailable after retries, the script FAILS (data are not simulated).
Output: data/raw/<CHEMBL_ID>.csv with columns RAW_COLUMNS (the format expected by scripts/02_standardize.py).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.ebi.ac.uk"
API = BASE + "/chembl/api/data"
RAW_COLUMNS = ["target_chembl_id", "target_name", "molecule_chembl_id", "canonical_smiles", "standard_type",
               "pchembl_value", "assay_chembl_id", "doc_chembl_id", "doc_year", "data_validity_comment"]
MIN_CONFIDENCE = 8
RETRY_PAUSES = (2, 4, 8, 16)


class ChemblUnavailable(RuntimeError):
    pass


def get_json(url, params=None, session=requests):
    last = None
    for pause in (*RETRY_PAUSES, None):
        try:
            r = session.get(url, params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last = e
            if pause is None:
                break
            time.sleep(pause)
    raise ChemblUnavailable(f"ChEMBL unavailable: {last}")


def paginate(url, params, key, session=requests):
    """Iterates over records page by page (page_meta.next is a relative path)."""
    while url:
        d = get_json(url, params, session)
        yield from d[key]
        nxt = d["page_meta"]["next"]
        url = BASE + nxt if nxt else None
        params = None  # next already contains the parameters


def parse_activities(records, assay_ok=None):
    """List of activity records -> DataFrame RAW_COLUMNS; filters are repeated client-side as a safeguard."""
    rows = []
    for a in records:
        if a.get("standard_relation") != "=" or a.get("pchembl_value") in (None, ""):
            continue
        if a.get("data_validity_comment"):
            continue
        if a.get("potential_duplicate") in (1, True, "1"):
            continue
        if not a.get("canonical_smiles") or a.get("document_year") is None:
            continue
        if assay_ok is not None and a.get("assay_chembl_id") not in assay_ok:
            continue
        rows.append({"target_chembl_id": a.get("target_chembl_id"), "target_name": a.get("target_pref_name"),
                     "molecule_chembl_id": a.get("molecule_chembl_id"), "canonical_smiles": a["canonical_smiles"],
                     "standard_type": a.get("standard_type"), "pchembl_value": float(a["pchembl_value"]),
                     "assay_chembl_id": a.get("assay_chembl_id"), "doc_chembl_id": a.get("document_chembl_id"),
                     "doc_year": int(a["document_year"]), "data_validity_comment": a.get("data_validity_comment")})
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def confident_assays(target, session=requests):
    recs = paginate(f"{API}/assay.json", {"target_chembl_id": target, "assay_type": "B",
                                          "confidence_score__gte": MIN_CONFIDENCE, "limit": 1000},
                    "assays", session)
    return {r["assay_chembl_id"] for r in recs}


def fetch_target(target, types, session=requests):
    ok = confident_assays(target, session)
    frames, n_api = [], 0
    for st in types:
        recs = list(paginate(f"{API}/activity.json",
                             {"target_chembl_id": target, "standard_type": st, "standard_relation": "=",
                              "pchembl_value__isnull": "false", "assay_type": "B",
                              "data_validity_comment__isnull": "true", "limit": 1000}, "activities", session))
        n_api += len(recs)
        frames.append(parse_activities(recs, ok))
    df = pd.concat(frames, ignore_index=True)
    return df, n_api, len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/endpoints.yaml"))
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--types", nargs="*")
    ap.add_argument("--out", default=str(ROOT / "data/raw"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    targets = a.targets or [t["id"] for t in cfg["chembl_candidates"]]
    types = a.types or cfg["standard_types"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    print("target\tpref_name\tapi_rows\tconf>=8_assays\tkept_rows\tdocs")
    for t in targets:
        try:
            df, n_api, n_as = fetch_target(t, types)
        except ChemblUnavailable as e:
            sys.exit(f"{t}: {e}. Step stopped; data are not simulated.")
        df.to_csv(Path(a.out) / f"{t}.csv", index=False)
        name = df["target_name"].iloc[0] if len(df) else "-"
        print(f"{t}\t{name}\t{n_api}\t{n_as}\t{len(df)}\t{df['doc_chembl_id'].nunique()}", flush=True)


if __name__ == "__main__":
    main()
