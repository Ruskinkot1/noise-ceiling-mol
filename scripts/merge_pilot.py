#!/usr/bin/env python
"""Склеивает результаты заданий пилота в results/{P}_{clean,raw}/ (benchmark, predictions, ceilings)."""
import glob, os, sys
P = sys.argv[1] if len(sys.argv) > 1 else "pilot"   # префикс: results/<P>/ -> results/<P>_clean
import pandas as pd

for v in ["clean", "raw"]:
    files = glob.glob(f"results/{P}/{v}_*_cheap.csv") + glob.glob(f"results/{P}/{v}_*_gnn.csv") \
        + [f for f in glob.glob(f"results/{P}/ft_{v}_*.csv") if not f.endswith(".pred.csv")]
    pred = glob.glob(f"results/{P}/{v}_*.pred.csv") + glob.glob(f"results/{P}/ft_{v}_*.pred.csv")
    if not files:
        print(f"{v}: нет результатов, пропуск"); continue
    os.makedirs(f"results/{P}_{v}", exist_ok=True)
    pd.concat([pd.read_csv(f) for f in files]).to_csv(f"results/{P}_{v}/benchmark.csv", index=False)
    pd.concat([pd.read_csv(f) for f in pred]).to_csv(f"results/{P}_{v}/predictions.csv", index=False)
    pd.read_csv(f"results/{P}/ceil_{v}/ceilings.csv").to_csv(f"results/{P}_{v}/ceilings.csv", index=False)
    print(f"{v}: объединено {len(files)} файлов")
