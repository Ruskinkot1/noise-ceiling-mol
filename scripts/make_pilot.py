#!/usr/bin/env python
"""Готовит папки пилота data/pilot_raw (неочищенные) и data/pilot_clean (после 02b) для 3 задач."""
import argparse, os, shutil
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="+", default=["CHEMBL237_IC50", "CHEMBL206_IC50", "CHEMBL204_Ki"])
a = ap.parse_args()
for name, src in [("pilot_raw", "data/processed"), ("pilot_clean", "data/processed_clean")]:
    d = f"data/{name}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    rows = []
    for t in a.tasks:
        for ext in ("molecules", "measurements"):
            shutil.copy(f"{src}/{t}.{ext}.csv", f"{d}/{t}.{ext}.csv")
        m = pd.read_csv(f"{d}/{t}.molecules.csv")
        rows.append({"task": t, "n_mols": len(m), "n_rep_mols": int((m.n_docs >= 2).sum()), "n_docs": 0, "passes": True})
    pd.DataFrame(rows).to_csv(f"{d}/tasks.csv", index=False)
    print(name, rows)
