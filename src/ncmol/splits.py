"""Splits: random, scaffold (Murcko), temporal. Return (train_idx, test_idx)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .chem import scaffold


def random_split(df: pd.DataFrame, seed: int, test_frac: float = 0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    k = int(round(len(df) * test_frac))
    return np.sort(idx[k:]), np.sort(idx[:k])


def scaffold_split(df: pd.DataFrame, seed: int, test_frac: float = 0.2):
    """Scaffolds go entirely to train or to test; group order is shuffled by seed (large groups go to train)."""
    groups = defaultdict(list)
    for i, s in enumerate(df["smiles"]):
        groups[scaffold(s)].append(i)
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    n_test = int(round(len(df) * test_frac))
    train, test = [], []
    for k in keys:
        g = groups[k]
        # the group goes to test if it fits; otherwise to train (large scaffolds always go to train)
        (test if len(test) + len(g) <= n_test else train).extend(g)
    return np.sort(train), np.sort(test)


def temporal_split(df: pd.DataFrame, seed: int = 0, test_frac: float = 0.2):
    """Test = molecules with the latest year of first publication (column first_year). Deterministic."""
    order = np.argsort(df["first_year"].to_numpy(), kind="stable")
    k = int(round(len(df) * test_frac))
    return np.sort(order[:-k]), np.sort(order[-k:])


SPLITS = {"random": random_split, "scaffold": scaffold_split, "temporal": temporal_split}
