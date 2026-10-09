"""Модели: sklearn-головы + собственный D-MPNN на torch (gnn.py). Эмбеддинги — предвычисленные
(scripts/06_embed_foundation.py), поверх них те же sklearn-головы.

Интерфейс: featurize(name, ...) возвращает 2D-матрицу признаков, а для 'gnn_*' — массив SMILES
(object), который модель разбирает сама; индексация X[tr] в scripts/04 работает одинаково."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .chem import ecfp

BASELINE = "ecfp_gbm"


def make_model(name: str, seed: int):
    base = name.split(":")[0]
    if base.startswith("gnn"):
        from .gnn import DMPNNRegressor
        return DMPNNRegressor(seed=seed)
    if base.endswith("_gbm"):
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
    if base.endswith("_rf"):
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
    if base.endswith("_mlp"):
        return make_pipeline(StandardScaler(), MLPRegressor((512, 128), alpha=1e-3, early_stopping=True,
                                                            max_iter=300, random_state=seed))
    if base.endswith("_ridge"):
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    raise ValueError(name)


def featurize(name: str, smiles, mol_ids, emb_dir=None):
    """'gnn_*' -> SMILES (object-массив); 'ecfp_*' -> ECFP4 counts 2048; 'emb-<model>_*' -> предвычисленный эмбеддинг из emb_dir/<model>.npz
    (ключи: mol_ids, X)."""
    if name.startswith("gnn"):
        return np.asarray(list(smiles), dtype=object)
    if name.startswith("ecfp"):
        return ecfp(list(smiles))
    if name.startswith("emb-"):
        model = name.split("_")[0][4:]
        z = np.load(f"{emb_dir}/{model}.npz", allow_pickle=True)
        lut = {m: i for i, m in enumerate(z["mol_ids"])}
        return z["X"][[lut[m] for m in mol_ids]]
    raise ValueError(name)


DEFAULT_MODELS = ["ecfp_gbm", "ecfp_rf", "ecfp_mlp"]
