"""Models: sklearn heads + a custom D-MPNN in torch (gnn.py). Embeddings are precomputed
(scripts/06_embed_foundation.py), with the same sklearn heads on top of them.

Interface: featurize(name, ...) returns a 2D feature matrix, and for 'gnn_*' an array of SMILES
(object) that the model parses itself; indexing X[tr] in scripts/04 works the same way."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .chem import ecfp

BASELINE = "ecfp_gbm"


# Hyperparameter grids. The tuning budget is the SAME for all families: n_trials random configurations
# from the grid, selected by RMSE on an inner validation set (20% of train), then refit on all of train.
PARAM_GRID = {
    "gbm": {"learning_rate": [0.03, 0.05, 0.1], "max_iter": [200, 400], "max_leaf_nodes": [15, 31, 63],
            "l2_regularization": [0.0, 1.0]},
    "rf": {"max_features": ["sqrt", 0.2, 0.5], "min_samples_leaf": [1, 2, 5]},
    "mlp": {"hidden": [(256,), (512, 128), (1024, 256)], "alpha": [1e-4, 1e-3, 1e-2], "lr": [1e-3, 3e-4]},
    "ridge": {"alpha": [1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0]},
    "gnn": {"hidden": [128, 200, 300], "depth": [2, 3, 4], "dropout": [0.0, 0.1, 0.2], "lr": [1e-3, 5e-4]},
}


def family(name: str) -> str:
    base = name.split(":")[0]
    if base.startswith("gnn"):
        return "gnn"
    for f in ("gbm", "rf", "mlp", "ridge"):
        if base.endswith("_" + f):
            return f
    raise ValueError(name)


def make_model(name: str, seed: int, params: dict | None = None):
    p = dict(params or {})
    f = family(name)
    if f == "gnn":
        from .gnn import DMPNNRegressor
        return DMPNNRegressor(seed=seed, **p)
    if f == "gbm":
        return HistGradientBoostingRegressor(random_state=seed, early_stopping=True, n_iter_no_change=10, validation_fraction=0.1,
                                             **{"max_iter": 300, "learning_rate": 0.05, **p})
    if f == "rf":
        return RandomForestRegressor(n_estimators=300, n_jobs=1, random_state=seed, **p)
    if f == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=tuple(p.get("hidden", (512, 128))), alpha=p.get("alpha", 1e-3), learning_rate_init=p.get("lr", 1e-3),
            early_stopping=True, max_iter=300, random_state=seed))
    if f == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=p.get("alpha", 10.0)))


def sample_configs(name: str, n_trials: int, seed: int):
    grid = PARAM_GRID[family(name)]
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    total = int(np.prod([len(v) for v in grid.values()]))
    while len(out) < min(n_trials, total):
        c = {k: v[rng.integers(len(v))] for k, v in grid.items()}
        key = repr(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def tune_and_fit(name: str, X_tr, y_tr, seed: int, n_trials: int = 6):
    """Returns (fitted model, best configuration). n_trials=0 means default hyperparameters."""
    y_tr = np.asarray(y_tr, float)
    if n_trials <= 0:
        return make_model(name, seed).fit(X_tr, y_tr), {}
    rng = np.random.default_rng(10_000 + seed)
    idx = rng.permutation(len(y_tr))
    k = max(1, int(0.2 * len(idx)))
    va, tr = idx[:k], idx[k:]
    best, best_c = np.inf, {}
    for c in sample_configs(name, n_trials, seed):
        m = make_model(name, seed, c).fit(X_tr[tr], y_tr[tr])
        e = float(np.sqrt(np.mean((y_tr[va] - m.predict(X_tr[va])) ** 2)))
        if e < best:
            best, best_c = e, c
    return make_model(name, seed, best_c).fit(X_tr, y_tr), best_c


def featurize(name: str, smiles, mol_ids, emb_dir=None):
    """'gnn_*' -> SMILES (object array); 'ecfp_*' -> ECFP4 counts 2048; 'emb-<model>_*' -> precomputed embedding from emb_dir/<model>.npz
    (keys: mol_ids, X)."""
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
