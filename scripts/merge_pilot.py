#!/usr/bin/env python
"""Склеивает результаты заданий пилота в results/pilot_{clean,raw}/ (benchmark, predictions, ceilings)."""
import glob, os
import pandas as pd

for v in ["clean", "raw"]:
    files = glob.glob(f"results/pilot/{v}_*_cheap.csv") + glob.glob(f"results/pilot/{v}_*_gnn.csv")
    pred = glob.glob(f"results/pilot/{v}_*.pred.csv")
    if not files:
        print(f"{v}: нет результатов, пропуск"); continue
    os.makedirs(f"results/pilot_{v}", exist_ok=True)
    pd.concat([pd.read_csv(f) for f in files]).to_csv(f"results/pilot_{v}/benchmark.csv", index=False)
    pd.concat([pd.read_csv(f) for f in pred]).to_csv(f"results/pilot_{v}/predictions.csv", index=False)
    pd.read_csv(f"results/pilot/ceil_{v}/ceilings.csv").to_csv(f"results/pilot_{v}/ceilings.csv", index=False)
    print(f"{v}: объединено {len(files)} файлов")
