#!/usr/bin/env python
"""Step 11. Provenance audit of between-document pairs: shared authors and republication.

For every task in data/<processed>/tasks.csv the between-document pairs of replicated molecules are
collected (same trimming as in 03: molecules whose values span more than --max-pair-diff are dropped).
Pairs are labelled with
  * shared authors between the two documents (ChEMBL `authors` field, normalised to surname + first initial):
    any shared author / at least two shared authors / shared last author (usually the senior author);
  * likely republication: a document pair that shares at least --min-shared molecules of which at least
    --copy-frac have identical values (|delta| < 0.005).
The pairwise noise estimate sigma = sqrt(mean(delta^2 / 2)) is then recomputed on subsets of pairs.
Document metadata comes from the ChEMBL REST API and is cached in --cache.
Output: results/<out>/provenance_pairs.csv (per task and subset), provenance_summary.csv (all tasks pooled),
        provenance_doc_pairs.csv (document pairs flagged as republication)."""
import argparse
import re
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://www.ebi.ac.uk/chembl/api/data"

ap = argparse.ArgumentParser()
ap.add_argument("--processed", default="data/full_clean")
ap.add_argument("--out", default="results/full_clean")
ap.add_argument("--cache", default="data/raw/documents.csv")
ap.add_argument("--max-pair-diff", type=float, default=3.0)
ap.add_argument("--min-shared", type=int, default=5)
ap.add_argument("--copy-frac", type=float, default=0.5)
ap.add_argument("--n-boot", type=int, default=1000)
a = ap.parse_args()


def fetch_documents(ids, cache):
    cache = Path(cache)
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["doc_id"])
    todo = sorted(set(ids) - set(have["doc_id"]))
    rows = []
    s = requests.Session()
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        for attempt in range(5):
            r = s.get(f"{API}/document.json", params={"document_chembl_id__in": ",".join(chunk), "limit": 1000,
                                                       "only": "document_chembl_id,authors,journal,doc_type,year,doi,pubmed_id"}, timeout=60)
            if r.status_code == 200:
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        got = {d["document_chembl_id"]: d for d in r.json()["documents"]}
        for c in chunk:
            d = got.get(c, {})
            rows.append({"doc_id": c, "authors": d.get("authors"), "journal": d.get("journal"), "doc_type": d.get("doc_type"),
                         "year": d.get("year"), "doi": d.get("doi"), "pubmed_id": d.get("pubmed_id")})
        print(f"documents {min(i + 50, len(todo))}/{len(todo)}", flush=True)
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def author_list(s):
    """'Smith J, Jones AB, ...' -> ['smith j', 'jones a'] in order; empty list if unknown."""
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for part in s.split(","):
        toks = part.strip().split()
        if not toks:
            continue
        sur = " ".join(toks[:-1]) if len(toks) > 1 else toks[0]
        ini = toks[-1][0] if len(toks) > 1 else ""
        out.append(re.sub(r"[^a-z\- ]", "", f"{sur} {ini}".lower()).strip())
    return out


tasks = pd.read_csv(Path(a.processed) / "tasks.csv")
if "passes" in tasks:
    tasks = tasks[tasks["passes"]]
meas = {t: pd.read_csv(Path(a.processed) / f"{t}.measurements.csv") for t in tasks.task}
all_docs = sorted(set().union(*[set(m.doc_id) for m in meas.values()]))
docs = fetch_documents(all_docs, a.cache).drop_duplicates("doc_id").set_index("doc_id")
authors = {d: author_list(docs.loc[d, "authors"]) if d in docs.index else [] for d in all_docs}
print("documents without authors:", sum(1 for v in authors.values() if not v), "of", len(all_docs))


def shared_info(da, db):
    A, B = authors[da], authors[db]
    if not A or not B:
        return np.nan, np.nan, np.nan
    n = len(set(A) & set(B))
    return float(n >= 1), float(n >= 2), float(A[-1] == B[-1])


pair_rows, docpair_rows = [], []
for t, m in meas.items():
    d = m.groupby(["mol_id", "doc_id"], as_index=False)["y"].mean()
    g = d.groupby("mol_id")["y"]
    rep = d[(g.transform("size") >= 2) & (g.transform(lambda s: s.max() - s.min()) <= a.max_pair_diff)]
    for mol, x in rep.groupby("mol_id"):
        for (da, ya), (db, yb) in combinations(sorted(zip(x.doc_id, x.y)), 2):
            pair_rows.append((t, mol, da, db, ya - yb))
pairs = pd.DataFrame(pair_rows, columns=["task", "mol_id", "doc_a", "doc_b", "delta"])
pairs["ident"] = pairs["delta"].abs() < 0.005
dp = pairs.groupby(["task", "doc_a", "doc_b"]).agg(n_shared=("delta", "size"), frac_ident=("ident", "mean")).reset_index()
dp["copied"] = (dp.n_shared >= a.min_shared) & (dp.frac_ident >= a.copy_frac)
info = np.array([shared_info(x, y) for x, y in zip(dp.doc_a, dp.doc_b)])
dp["shared_any"], dp["shared_2plus"], dp["shared_last"] = info[:, 0], info[:, 1], info[:, 2]
dp["same_journal"] = [float(docs.loc[x, "journal"] == docs.loc[y, "journal"]) if x in docs.index and y in docs.index else np.nan
                      for x, y in zip(dp.doc_a, dp.doc_b)]
pairs = pairs.merge(dp[["task", "doc_a", "doc_b", "copied", "shared_any", "shared_2plus", "shared_last"]], on=["task", "doc_a", "doc_b"])


def sigma(p):
    return float(np.sqrt((p["delta"] ** 2 / 2).mean())) if len(p) else np.nan


def boot_ci(p, n_boot, rng):
    """Bootstrap over document pairs (clusters)."""
    if p["delta"].size < 2:
        return np.nan, np.nan
    grp = p.assign(h=p["delta"] ** 2 / 2).groupby(["doc_a", "doc_b"]).agg(h=("h", "sum"), n=("h", "size"))
    h, n = grp["h"].to_numpy(), grp["n"].to_numpy()
    k = len(h)
    idx = rng.integers(0, k, size=(n_boot, k))
    est = np.sqrt(h[idx].sum(1) / n[idx].sum(1))
    return tuple(np.percentile(est, [2.5, 97.5]))


subsets = {
    "all": lambda p: p,
    "no_shared_author": lambda p: p[p.shared_any == 0],
    "shared_author": lambda p: p[p.shared_any == 1],
    "shared_2plus": lambda p: p[p.shared_2plus == 1],
    "no_shared_2plus": lambda p: p[p.shared_2plus == 0],
    "shared_last_author": lambda p: p[p.shared_last == 1],
    "no_shared_last_author": lambda p: p[p.shared_last == 0],
    "no_copied_pairs": lambda p: p[~p.copied],
    "no_shared_author_no_copied": lambda p: p[(p.shared_any == 0) & (~p.copied)],
}
rng = np.random.default_rng(0)
rows = []
for scope, sub in [("ALL_TASKS", pairs)] + [(t, p) for t, p in pairs.groupby("task")]:
    for name, f in subsets.items():
        p = f(sub)
        lo, hi = boot_ci(p, a.n_boot, rng) if len(p) > 1 else (np.nan, np.nan)
        rows.append({"task": scope, "subset": name, "n_pairs": len(p), "n_doc_pairs": p[["doc_a", "doc_b"]].drop_duplicates().shape[0],
                     "sigma_pairs": sigma(p), "ci_lo": lo, "ci_hi": hi})
res = pd.DataFrame(rows)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
res.to_csv(out / "provenance_pairs.csv", index=False)
dp[dp.copied].to_csv(out / "provenance_doc_pairs.csv", index=False)
summ = res[res.task == "ALL_TASKS"]
summ.to_csv(out / "provenance_summary.csv", index=False)
print(summ.round(3).to_string(index=False))
print("pairs with unknown authors:", int(pairs.shared_any.isna().sum()), "of", len(pairs))
print("copied document pairs:", int(dp.copied.sum()), "of", len(dp), "; molecule pairs in them:", int(pairs.copied.sum()), "of", len(pairs))
