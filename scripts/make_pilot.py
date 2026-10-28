#!/usr/bin/env python
"""Prepares the pilot folders data/pilot_raw (uncleaned) and data/pilot_clean (after 02b) for 3 tasks."""
import argparse, os, shutil
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="+", default=["CHEMBL237_IC50", "CHEMBL206_IC50", "CHEMBL204_Ki"])
ap.add_argument("--name", default="pilot", help="folder prefix: data/<name>_raw and data/<name>_clean")
ap.add_argument("--versions", nargs="+", default=["raw", "clean"], choices=["raw", "clean"])
a = ap.parse_args()
for name, src in [(f"{a.name}_{v}", {"raw": "data/processed", "clean": "data/processed_clean"}[v]) for v in a.versions]:
    d = f"data/{name}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for t in a.tasks:
        for ext in ("molecules", "measurements"):
            shutil.copy(f"{src}/{t}.{ext}.csv", f"{d}/{t}.{ext}.csv")
    # take rows from the real tasks.csv (it has target, type, etc.); recompute n_mols/n_rep_mols from the folder
    full = pd.read_csv(f"{src}/tasks.csv")
    full = full[full["task"].isin(a.tasks)].copy()
    meta = pd.read_csv("data/processed/tasks.csv")[["task", "target", "type"]]  # 02b does not preserve target/type
    full = full.drop(columns=[c for c in ("target", "type") if c in full.columns]).merge(meta, on="task", how="left")
    full = full.reset_index(drop=True)
    for i, t in full["task"].items():
        m = pd.read_csv(f"{d}/{t}.molecules.csv")
        full.loc[i, "n_mols"], full.loc[i, "n_rep_mols"] = len(m), int((m.n_docs >= 2).sum())
    full["passes"] = True
    full.to_csv(f"{d}/tasks.csv", index=False)
    print(name, full[["task", "n_mols", "n_rep_mols"]].to_dict("records"))
