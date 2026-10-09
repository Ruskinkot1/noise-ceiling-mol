#!/usr/bin/env python
"""Шаг 6. Эмбеддинги предобученных (замороженных) моделей, CPU, mean-pooling с маской.
Вход: data/processed/*.molecules.csv (mol_id, smiles). Выход: data/embeddings/<name>.npz
(ключи: mol_ids, X; дополнительно: valid, revision, repo). Кэш: если файл есть и покрывает все mol_id с тем же
revision, модель пропускается (--force пересчитывает). Каждая ревизия закреплена (commit HF).
Пример: python scripts/06_embed_foundation.py --models chemberta-77m-mlm molformer-xl-10pct
        python scripts/06_embed_foundation.py --smoke 20     # быстрый тест на 20 молекулах"""
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]

# repo, revision (commit HF, проверен 2026-10-09), trust_remote_code, patterns (какие файлы качать: только нужные)
MODELS = {
    "chemberta-77m-mlm": dict(repo="DeepChem/ChemBERTa-77M-MLM", rev="ed8a5374f2024ec8da53760af91a33fb8f6a15ff"),
    "chemberta-77m-mtr": dict(repo="DeepChem/ChemBERTa-77M-MTR", rev="66b895cab8adebea0cb59a8effa66b2020f204ca"),
    "chemberta-5m-mlm": dict(repo="DeepChem/ChemBERTa-5M-MLM", rev="695672a601a8077ace5d0a6fc2529e93cc9cfa9c"),
    # код модели (modeling_molformer.py, configuration_molformer.py) прочитан перед запуском: только torch/transformers
    "molformer-xl-10pct": dict(repo="ibm-research/MoLFormer-XL-both-10pct",
                               rev="361063d0ad524ef77cf39b08469f6be770dc550f", remote=True),
    # 409 МБ: качаем только model.safetensors (без дублирующего pytorch_model.bin)
    "roberta-zinc-480m": dict(repo="entropy/roberta_zinc_480m", rev="05b00ec36b6d22e365cc06079516a5611c5f1d26",
                              patterns=["*.json", "*.txt", "model.safetensors"]),
}


def load_molecules(paths):
    frames = [pd.read_csv(p, usecols=["mol_id", "smiles"]) for p in paths]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("mol_id", keep="first").reset_index(drop=True)
    return df


def canonical(smiles):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToSmiles(m) if m is not None else None


@torch.no_grad()
def embed_smiles(smiles, tok, model, batch_size=32, max_length=256):
    """smiles: список валидных канонических SMILES -> (n, d) float32. mean-pooling по last_hidden_state с маской.
    Батчи по возрастанию длины (меньше паддинга), порядок результата сохраняется."""
    order = np.argsort([len(s) for s in smiles], kind="stable")
    out = [None] * len(smiles)
    for i in range(0, len(smiles), batch_size):
        idx = order[i:i + batch_size]
        enc = tok([smiles[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length)
        enc.pop("token_type_ids", None)
        h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = ((h * mask).sum(1) / mask.sum(1).clamp(min=1)).float().numpy()
        for k, j in enumerate(idx):
            out[j] = e[k]
    return np.stack(out) if out else np.zeros((0, 0), np.float32)


def load_model(spec):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    path = snapshot_download(spec["repo"], revision=spec["rev"], allow_patterns=spec.get("patterns"))
    remote = spec.get("remote", False)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=remote)
    model = AutoModel.from_pretrained(path, trust_remote_code=remote, **({} if "patterns" not in spec else {"add_pooling_layer": False})).eval()
    return tok, model


def run_model(name, mol_ids, smiles, a):
    spec = MODELS[name]
    out = Path(a.out_dir) / f"{name}.npz"
    if out.exists() and not a.force:
        z = np.load(out, allow_pickle=True)
        if str(z["revision"]) == spec["rev"] and set(map(str, mol_ids)) <= set(map(str, z["mol_ids"])):
            print(f"[{name}] кэш актуален: {out}", flush=True)
            return
    t0 = time.time()
    tok, model = load_model(spec)
    can = [canonical(s) for s in smiles]
    ok = np.array([c is not None for c in can])
    X = None
    if ok.any():
        E = embed_smiles([c for c in can if c is not None], tok, model, a.batch_size, a.max_length)
        X = np.zeros((len(can), E.shape[1]), np.float32)
        X[ok] = E
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mol_ids=np.asarray(mol_ids, dtype=object), X=X, valid=ok, revision=spec["rev"], repo=spec["repo"])
    print(f"[{name}] {len(can)} мол., dim={X.shape[1]}, невалидных SMILES={int((~ok).sum())}, {time.time()-t0:.1f} с -> {out}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=str(ROOT / "data/processed"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/embeddings"))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--smoke", type=int, default=0, help="взять только первые N молекул (проверка работоспособности)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    paths = sorted(glob.glob(str(Path(a.processed) / "*.molecules.csv")))
    if not paths:
        raise SystemExit(f"нет файлов {a.processed}/*.molecules.csv")
    df = load_molecules(paths)
    if a.smoke:
        df = df.head(a.smoke)
    for m in a.models:
        run_model(m, df["mol_id"].tolist(), df["smiles"].tolist(), a)


if __name__ == "__main__":
    main()
