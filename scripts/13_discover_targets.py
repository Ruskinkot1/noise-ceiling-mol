#!/usr/bin/env python
"""Step 13. Discover candidate targets for the task pool.

Lists human single-protein targets in ChEMBL and counts, per target, binding-assay activities with a pChEMBL value
for IC50 and Ki together (ChEMBL REST API, one request per target). A target is a candidate when it has at least
--min-acts such activities: replicated molecules (>= 2 documents) are a small fraction of all molecules, so targets with few
activities cannot pass the replicate threshold of configs/endpoints.yaml.
The confidence_score >= 8 filter of 01_extract_chembl.py is applied later, at extraction time; counts here are upper bounds.
Output: data/raw/target_counts.csv (resumable: targets already in the file are skipped)."""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

API = "https://www.ebi.ac.uk/chembl/api/data"
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/raw/target_counts.csv")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--min-acts", type=int, default=1500)
a = ap.parse_args()
sess = requests.Session()


def get(url, params, tries=8):
    for k in range(tries):
        try:
            r = sess.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 ** k)
    raise RuntimeError(f"failed: {url} {params}")


def list_targets():
    out, off = [], 0
    while True:
        d = get(f"{API}/target.json", {"target_type": "SINGLE PROTEIN", "organism": "Homo sapiens", "limit": 1000, "offset": off,
                                       "only": "target_chembl_id,pref_name"})
        out += [(t["target_chembl_id"], t["pref_name"]) for t in d["targets"]]
        if not d["page_meta"]["next"]:
            return out
        off += 1000


def count(tid):
    d = get(f"{API}/activity.json", {"target_chembl_id": tid, "standard_type__in": "IC50,Ki", "pchembl_value__isnull": "false", "assay_type": "B",
                                     "limit": 1, "only": "activity_id"})
    return d["page_meta"]["total_count"]


def one(t):
    tid, name = t
    try:
        return {"target": tid, "pref_name": name, "n_acts": count(tid)}
    except RuntimeError:           # rate limit or server error: mark and retry on the next run
        return {"target": tid, "pref_name": name, "n_acts": -1}


out = Path(a.out)
done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["target", "pref_name", "n_acts"])
done = done[done["n_acts"] >= 0]
todo = [t for t in list_targets() if t[0] not in set(done["target"])]
print(f"{len(todo)} targets to count", flush=True)
rows = []
with ThreadPoolExecutor(a.threads) as ex:
    for i, r in enumerate(ex.map(one, todo), 1):
        rows.append(r)
        if i % 200 == 0 or i == len(todo):
            pd.concat([done, pd.DataFrame(rows)]).to_csv(out, index=False)
            print(f"{i}/{len(todo)}", flush=True)
res = pd.concat([done, pd.DataFrame(rows)])
res.to_csv(out, index=False)
cand = res[res.n_acts >= a.min_acts]
print(f"candidates with >= {a.min_acts} activities: {len(cand)} of {len(res)}")
