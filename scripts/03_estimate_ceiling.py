#!/usr/bin/env python
"""Шаг 3. Потолок шума: sigma, RMSE_floor, R2_max + бутстрэп + чувствительность к порогам.
Выход: results/ceilings.csv, results/ceiling_sensitivity.csv.
Критерий остановки нед. 5: если R2_max у большинства задач близок к 1 — переформулировать вклад."""
import argparse
from pathlib import Path

import pandas as pd

from ncmol.ceiling import bootstrap_ceiling, estimate_ceiling, sensitivity

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=str(ROOT / "data/processed"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--min-docs", type=int, default=2)
    ap.add_argument("--max-pair-diff", type=float, default=3.0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tasks = pd.read_csv(Path(a.processed) / "tasks.csv")
    tasks = tasks[tasks["passes"]]
    rows, sens = [], []
    for t in tasks["task"]:
        m = pd.read_csv(Path(a.processed) / f"{t}.measurements.csv").rename(columns={"y": "y"})
        c = estimate_ceiling(m, a.min_docs, a.max_pair_diff)
        b = bootstrap_ceiling(m, n_boot=a.n_boot, min_docs=a.min_docs, max_abs_pair_diff=a.max_pair_diff)
        rows.append({"task": t, **c.to_dict(), **b.iloc[0].to_dict()})
        s = sensitivity(m)
        s.insert(0, "task", t)
        sens.append(s)
        print(f"{t}: sigma={c.sigma:.2f} RMSE_floor={c.rmse_floor:.2f} R2_max={c.r2_max:.2f} (n_rep={c.n_mols_rep})")
    pd.DataFrame(rows).to_csv(Path(a.out) / "ceilings.csv", index=False)
    pd.concat(sens).to_csv(Path(a.out) / "ceiling_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
