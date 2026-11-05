#!/usr/bin/env python
"""Step 12. Assay conditions of kinase IC50 tasks: ATP concentration and the Cheng-Prusoff relation.

For competitive inhibitors IC50 = Ki * (1 + [S]/Km), so documents that used different ATP concentrations
report systematically different IC50 values for the same molecule. This script
  1. downloads assay descriptions and assay parameters from the ChEMBL REST API (cached in --cache);
  2. parses the ATP concentration (uM) from them;
  3. reports how many assays and replicate pairs have a parsed ATP concentration on both sides;
  4. compares the pairwise noise sigma = sqrt(mean(delta^2/2)) for pairs with matched ATP (ratio <= --match)
     and mismatched ATP;
  5. fits one Km(ATP) per task by minimising the squared residual of
     delta = y_a - y_b = -log10((1 + A_a/Km) / (1 + A_b/Km)) over a grid and reports sigma after the correction.
Pairs are built from assay-level values of different documents. Output: results/<out>/assay_conditions.csv."""
import argparse
import re
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://www.ebi.ac.uk/chembl/api/data"
UNIT = {"nm": 1e-3, "um": 1.0, "mm": 1e3}   # factor to uM, keys are lower-case with micro sign replaced by "u"
NUM = r"(\d+(?:[.,]\d+)?)"
UN = r"(nM|[uμµ]M|mM)"
PATTERNS = [
    rf"ATP[^.;,]{{0,40}}?(?:at|of|=|:|was|were|with)?\s*{NUM}\s*{UN}",       # ATP at 10 uM / ATP concentration of 10 uM
    rf"{NUM}\s*{UN}\s*(?:of\s+)?(?:\[?(?:gamma|γ|g)\s*-?\s*(?:33|32)P\]?\s*-?\s*)?ATP",   # 10 uM ATP / 10 uM [gamma-33P]ATP
]


def parse_atp(text):
    """First ATP concentration in uM found in `text`, or NaN. 'Km' / 'apparent Km' statements return NaN (not a value)."""
    if not isinstance(text, str):
        return np.nan
    for pat in PATTERNS:
        m = re.search(pat, text, flags=re.I)
        if m:
            val = float(m.group(1).replace(",", "."))
            return val * UNIT[m.group(2).lower().replace("µ", "u").replace("μ", "u")]
    return np.nan


def fetch_assays(ids, cache):
    cache = Path(cache)
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["assay_id"])
    todo = sorted(set(ids) - set(have["assay_id"]))
    rows = []
    s = requests.Session()
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        for attempt in range(5):
            r = s.get(f"{API}/assay.json", params={"assay_chembl_id__in": ",".join(chunk), "limit": 1000,
                                                    "only": "assay_chembl_id,description,assay_parameters,assay_organism,assay_cell_type"}, timeout=60)
            if r.status_code == 200:
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        got = {x["assay_chembl_id"]: x for x in r.json()["assays"]}
        for c in chunk:
            x = got.get(c, {})
            params = " | ".join(f"{p.get('standard_type') or p.get('type')}={p.get('standard_value', p.get('value'))} {p.get('standard_units') or p.get('units') or ''}"
                                for p in (x.get("assay_parameters") or []))
            rows.append({"assay_id": c, "description": x.get("description"), "parameters": params,
                         "organism": x.get("assay_organism"), "cell_type": x.get("assay_cell_type")})
        print(f"assays {min(i + 50, len(todo))}/{len(todo)}", flush=True)
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def atp_from_row(r):
    for field in ("parameters", "description"):
        v = parse_atp(r[field])
        if not np.isnan(v):
            return v
    return np.nan


def sigma(delta):
    return float(np.sqrt(np.mean(np.asarray(delta) ** 2 / 2))) if len(delta) else np.nan


def fit_km(pairs, grid):
    best = None
    for km in grid:
        corr = -np.log10((1 + pairs.atp_a / km) / (1 + pairs.atp_b / km))
        sse = float(((pairs.delta - corr) ** 2).mean())
        if best is None or sse < best[1]:
            best = (km, sse)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/full_clean")
    ap.add_argument("--tasks", nargs="+", default=["CHEMBL203_IC50", "CHEMBL279_IC50", "CHEMBL2971_IC50", "CHEMBL4005_IC50", "CHEMBL5145_IC50"])
    ap.add_argument("--out", default="results/full_clean")
    ap.add_argument("--cache", default="data/raw/assays_kinase.csv")
    ap.add_argument("--max-pair-diff", type=float, default=3.0)
    ap.add_argument("--match", type=float, default=2.0, help="ATP concentrations within this ratio count as matched")
    a = ap.parse_args()
    meas = {t: pd.read_csv(Path(a.processed) / f"{t}.measurements.csv") for t in a.tasks}
    ids = sorted(set().union(*[set(m.assay_id) for m in meas.values()]))
    assays = fetch_assays(ids, a.cache).drop_duplicates("assay_id")
    assays["atp_uM"] = assays.apply(atp_from_row, axis=1)
    atp = assays.set_index("assay_id")["atp_uM"]
    grid = np.logspace(-1, 3, 81)
    rows = []
    for t, m in meas.items():
        m = m.assign(atp=m.assay_id.map(atp))
        rng_ok = m.groupby("mol_id")["y"].transform(lambda s: s.max() - s.min()) <= a.max_pair_diff
        pr = []
        for mol, x in m[rng_ok].groupby("mol_id"):
            if x.doc_id.nunique() < 2:
                continue
            for (d1, as1, y1, p1), (d2, as2, y2, p2) in combinations(zip(x.doc_id, x.assay_id, x.y, x.atp), 2):
                if d1 != d2:
                    pr.append((mol, d1, d2, y1 - y2, p1, p2))
        pr = pd.DataFrame(pr, columns=["mol_id", "doc_a", "doc_b", "delta", "atp_a", "atp_b"])
        known = pr.dropna(subset=["atp_a", "atp_b"])
        ratio = np.maximum(known.atp_a, known.atp_b) / np.minimum(known.atp_a, known.atp_b)
        matched, mism = known[ratio <= a.match], known[ratio > a.match]
        km, sse = fit_km(known, grid) if len(known) >= 30 else (np.nan, np.nan)
        corr = (-np.log10((1 + known.atp_a / km) / (1 + known.atp_b / km))) if len(known) >= 30 else 0
        rows.append({"task": t, "n_assays": m.assay_id.nunique(), "assays_atp_parsed": int(m.assay_id.drop_duplicates().map(atp).notna().sum()),
                     "n_pairs": len(pr), "pairs_both_atp": len(known), "sigma_all": sigma(pr.delta), "sigma_known": sigma(known.delta),
                     "n_matched": len(matched), "sigma_matched": sigma(matched.delta), "n_mismatched": len(mism), "sigma_mismatched": sigma(mism.delta),
                     "km_fit_uM": km, "sigma_known_after_cp": sigma(known.delta - corr) if len(known) >= 30 else np.nan})
    res = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "assay_conditions.csv", index=False)
    assays.to_csv(Path(a.cache).with_name("assays_kinase_atp.csv"), index=False)
    print(res.round(3).to_string(index=False))
