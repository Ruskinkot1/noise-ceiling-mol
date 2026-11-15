"""Сплиты: случайный, скэффолдный (Murcko), временной, по документам. Возвращают (train_idx, test_idx)."""
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


def document_split(df: pd.DataFrame, seed: int, test_frac: float = 0.2):
    """Документ целиком в train или test: тестовая молекула не имеет в обучении ни одной молекулы
    из своего документа. Снимает поддержку документа, из-за которой модели пробивают потолок шума
    на случайном и скэффолдном сплитах (см. раздел про поддержку документа в статье).
    Требует столбец first_doc (scripts/02_standardize.py). Крупные документы уходят в train,
    как и крупные скэффолды в scaffold_split, иначе один документ съедает весь тест."""
    if "first_doc" not in df.columns:
        raise KeyError("document_split требует столбец first_doc; пересоберите *.molecules.csv "
                       "через scripts/02_standardize.py или scripts/backfill_first_doc.py")
    groups = defaultdict(list)
    for i, doc in enumerate(df["first_doc"]):
        groups[doc].append(i)
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    n_test = int(round(len(df) * test_frac))
    train, test = [], []
    for k in keys:
        g = groups[k]
        (test if len(test) + len(g) <= n_test else train).extend(g)
    return np.sort(train), np.sort(test)


def component_split(df: pd.DataFrame, seed: int, test_frac: float = 0.2, meas: pd.DataFrame | None = None):
    """Строгая версия document_split. Молекулы и документы образуют двудольный граф; связная
    компонента целиком идёт в train или test, поэтому у тестовой молекулы в обучении нет молекул
    ни из одного её документа — даже если обучающая молекула измерена ещё где-то. document_split
    группирует только по самому раннему документу и оставляет часть поддержки (см. статью).
    meas: таблица измерений (mol_id, doc_id). Без неё падает обратно на first_doc."""
    if meas is None:
        return document_split(df, seed, test_frac)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    md = meas[["mol_id", "doc_id"]].drop_duplicates()
    md = md[md.mol_id.isin(set(df["mol_id"]))]
    mol_ix = {v: i for i, v in enumerate(df["mol_id"])}
    doc_ix = {v: len(mol_ix) + i for i, v in enumerate(md.doc_id.unique())}
    r = [mol_ix[x] for x in md.mol_id]
    c = [doc_ix[x] for x in md.doc_id]
    n = len(mol_ix) + len(doc_ix)
    _, lab = connected_components(coo_matrix(([1] * len(r), (r, c)), shape=(n, n)), directed=False)
    comp = pd.Series(lab[: len(mol_ix)], index=df.index)
    groups = defaultdict(list)
    for i, g in enumerate(comp.to_numpy()):
        groups[g].append(i)
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    n_test = int(round(len(df) * test_frac))
    train, test = [], []
    for k in keys:
        g = groups[k]
        (test if len(test) + len(g) <= n_test else train).extend(g)
    return np.sort(train), np.sort(test)


SPLITS = {"random": random_split, "scaffold": scaffold_split, "temporal": temporal_split,
          "document": document_split, "component": component_split}
