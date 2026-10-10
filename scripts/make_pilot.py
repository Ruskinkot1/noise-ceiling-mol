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
    for t in a.tasks:
        for ext in ("molecules", "measurements"):
            shutil.copy(f"{src}/{t}.{ext}.csv", f"{d}/{t}.{ext}.csv")
    # строки берём из настоящего tasks.csv (там есть target, type и др.); n_mols/n_rep_mols пересчитываем по папке
    full = pd.read_csv(f"{src}/tasks.csv")
    full = full[full["task"].isin(a.tasks)].copy()
    meta = pd.read_csv("data/processed/tasks.csv")[["task", "target", "type"]]  # 02b не сохраняет target/type
    full = full.drop(columns=[c for c in ("target", "type") if c in full.columns]).merge(meta, on="task", how="left")
    full = full.reset_index(drop=True)
    for i, t in full["task"].items():
        m = pd.read_csv(f"{d}/{t}.molecules.csv")
        full.loc[i, "n_mols"], full.loc[i, "n_rep_mols"] = len(m), int((m.n_docs >= 2).sum())
    full["passes"] = True
    full.to_csv(f"{d}/tasks.csv", index=False)
    print(name, full[["task", "n_mols", "n_rep_mols"]].to_dict("records"))
