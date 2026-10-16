#!/usr/bin/env python
"""Шаг 8. Полное дообучение ChemBERTa-77M-MLM (mean-pooling + линейная голова) на одной задаче.
Те же сплиты и сиды, что в 04_run_benchmark.py; формат выхода тот же, модель называется ft-chemberta-77m-mlm.
Подбор: 2 значения lr (по RMSE на внутренней валидации = 10% train, сид сплита), ранняя остановка по этой же валидации.
Целевая переменная стандартизуется по train. Пример:
  python scripts/08_finetune.py --processed data/pilot_clean --tasks CHEMBL237_IC50 --seeds 5 --device mps \
      --out results/pilot/ft_clean_CHEMBL237_IC50.csv --pred-out results/pilot/ft_clean_CHEMBL237_IC50.pred.csv"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

from ncmol.device import resolve_device
from ncmol.metrics import regression_metrics
from ncmol.splits import SPLITS

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
REPO, REV = "DeepChem/ChemBERTa-77M-MLM", "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"  # как в 06_embed_foundation.py
NAME = "ft-chemberta-77m-mlm"


def canonical(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m is not None else s


class Regressor(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder
        self.head = torch.nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


def encode(tok, smiles, max_length=256):
    e = tok(list(smiles), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


@torch.no_grad()
def predict(model, tok, smiles, dev, bs=64):
    model.eval()
    out = []
    for i in range(0, len(smiles), bs):
        ids, am = encode(tok, smiles[i:i + bs])
        out.append(model(ids.to(dev), am.to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def fit_one(path, tok, smi_tr, y_tr, smi_va, y_va, lr, seed, dev, epochs, bs):
    from transformers import AutoModel
    torch.manual_seed(seed)
    model = Regressor(AutoModel.from_pretrained(path, add_pooling_layer=False)).to(dev)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": lr},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    rng = np.random.default_rng(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(smi_tr))
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            ids, am = encode(tok, [smi_tr[k] for k in j])
            loss = torch.nn.functional.mse_loss(model(ids.to(dev), am.to(dev)),
                                                torch.tensor(y_tr[j], dtype=torch.float32, device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        rmse = float(np.sqrt(np.mean((predict(model, tok, smi_va, dev) - y_va) ** 2)))
        if rmse < best - 1e-4:
            best, best_state, bad = rmse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= 5:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lrs", nargs="+", type=float, default=[2e-5, 1e-4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-out", required=True)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path = snapshot_download(REPO, revision=REV)
    tok = AutoTokenizer.from_pretrained(path)
    dev = resolve_device(a.device)
    rows, preds = [], []
    for t in a.tasks:
        df = pd.read_csv(Path(a.processed) / f"{t}.molecules.csv")
        smi = np.array([canonical(s) for s in df["smiles"]], dtype=object)
        y = df["y"].to_numpy(float)
        for sp in a.splits:
            for seed in range(a.seeds):
                t0 = time.time()
                tr, te = SPLITS[sp](df, seed)
                rng = np.random.default_rng(seed)
                p = rng.permutation(len(tr))
                nva = max(1, int(0.1 * len(tr)))
                va_i, fi_i = tr[p[:nva]], tr[p[nva:]]
                mu, sd = y[fi_i].mean(), y[fi_i].std() or 1.0
                cands = []
                for lr in a.lrs:
                    model, vr, ne = fit_one(path, tok, list(smi[fi_i]), (y[fi_i] - mu) / sd, list(smi[va_i]),
                                            (y[va_i] - mu) / sd, lr, seed, dev, a.epochs, a.batch_size)
                    cands.append((vr, lr, ne, model))
                vr, lr, ne, model = min(cands, key=lambda c: c[0])
                pred = predict(model, tok, list(smi[te]), dev) * sd + mu
                met = regression_metrics(y[te], pred)
                met["rmse_mean"] = float(np.sqrt(np.mean((y[te] - y[tr].mean()) ** 2)))
                rows.append({"task": t, "split": sp, "seed": seed, "model": NAME, "n_train": len(tr), "n_test": len(te),
                             "best_params": repr({"lr": lr, "epochs": ne, "val_rmse_std": round(vr, 4)}),
                             "test_mean_inv_ndocs": float((1 / df["n_docs"].iloc[te]).mean()), **met})
                preds.append(pd.DataFrame({"task": t, "split": sp, "seed": seed, "model": NAME,
                                           "mol_id": df["mol_id"].iloc[te].to_numpy(), "y": y[te], "pred": pred,
                                           "n_docs": df["n_docs"].iloc[te].to_numpy(),
                                           "y_train_mean": float(y[tr].mean())}))
                print(f"{t} {sp} {seed} rmse={met['rmse']:.3f} lr={lr} ep={ne} {time.time()-t0:.0f}s", flush=True)
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(a.out, index=False)       # сохраняем после каждого сида
                pd.concat(preds).to_csv(a.pred_out, index=False)
                del cands, model


if __name__ == "__main__":
    main()
