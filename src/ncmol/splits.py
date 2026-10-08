"""Сплиты: случайный, скэффолдный (Murcko), временной. Возвращают (train_idx, test_idx)."""
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
    """Скэффолды целиком в train или test; порядок групп перемешан по сиду (крупные группы — в train)."""
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
        # группа уходит в тест, если помещается; иначе в train (большие скэффолды всегда в train)
        (test if len(test) + len(g) <= n_test else train).extend(g)
    return np.sort(train), np.sort(test)


def temporal_split(df: pd.DataFrame, seed: int = 0, test_frac: float = 0.2):
    """Тест = самые поздние по году первой публикации молекулы (столбец first_year). Детерминирован."""
    order = np.argsort(df["first_year"].to_numpy(), kind="stable")
    k = int(round(len(df) * test_frac))
    return np.sort(order[:-k]), np.sort(order[-k:])


SPLITS = {"random": random_split, "scaffold": scaffold_split, "temporal": temporal_split}
