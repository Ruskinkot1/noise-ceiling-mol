#!/usr/bin/env python
"""Шаг 4. Модели x сплиты x сиды. Выход: results/benchmark.csv (одна строка = задача/сплит/сид/модель).
Эмбеддинги: --models ecfp_gbm emb-<name>_gbm --emb-dir data/embeddings (файл <name>.npz: mol_ids, X)."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ncmol.metrics import regression_metrics
from ncmol.models import DEFAULT_MODELS, featurize, tune_and_fit
from ncmol.splits import SPLITS

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=str(ROOT / "data/processed"))
    ap.add_argument("--out", default=str(ROOT / "results/benchmark.csv"))
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--emb-dir", default=str(ROOT / "data/embeddings"))
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"],
                    help="устройство torch для GNN (cpu по умолчанию; mps — Mac Apple Silicon). sklearn-модели всегда на CPU")
    ap.add_argument("--n-trials", type=int, default=6, help="одинаковый бюджет подбора на модель; 0 = без подбора")
    ap.add_argument("--pred-out", default=str(ROOT / "results/predictions.csv"))
    a = ap.parse_args()
    os.environ["NCMOL_DEVICE"] = a.device
    tasks = pd.read_csv(Path(a.processed) / "tasks.csv")
    names = a.tasks or tasks[tasks["passes"]]["task"].tolist()
    rows, preds = [], []
    for t in names:
        df = pd.read_csv(Path(a.processed) / f"{t}.molecules.csv")
        feats = {m: featurize(m, df["smiles"], df["mol_id"], a.emb_dir) for m in a.models}
        for sp in a.splits:
            for seed in range(a.seeds):
                tr, te = SPLITS[sp](df, seed)
                for m in a.models:
                    X = feats[m]
                    model, cfg = tune_and_fit(m, X[tr], df["y"].iloc[tr].to_numpy(), seed, a.n_trials)
                    pred = model.predict(X[te])
                    met = regression_metrics(df["y"].iloc[te], pred)
                    preds.append(pd.DataFrame({"task": t, "split": sp, "seed": seed, "model": m,
                                               "mol_id": df["mol_id"].iloc[te].to_numpy(), "y": df["y"].iloc[te].to_numpy(),
                                               "pred": pred, "n_docs": df["n_docs"].iloc[te].to_numpy(),
                                               "y_train_mean": float(df["y"].iloc[tr].mean())}))
                    # RMSE предсказателя «среднее по train» на том же тесте: нулевая точка шкалы доли потолка
                    met["rmse_mean"] = float(np.sqrt(np.mean((df["y"].iloc[te].to_numpy() - df["y"].iloc[tr].mean()) ** 2)))
                    # n_docs тест-молекул нужен для эффективного шума метки (шаг 5)
                    rows.append({"task": t, "split": sp, "seed": seed, "model": m, "n_train": len(tr),
                                 "n_test": len(te), "best_params": repr(cfg), "test_mean_inv_ndocs": float((1 / df["n_docs"].iloc[te]).mean()),
                                 **met})
                print(t, sp, seed, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    Path(a.pred_out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(preds).to_csv(a.pred_out, index=False)


if __name__ == "__main__":
    main()
