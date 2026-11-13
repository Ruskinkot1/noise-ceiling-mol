"""Сплит по документам: ни одна тестовая молекула не делит документ с обучающей."""
import numpy as np
import pandas as pd
import pytest

from ncmol.splits import SPLITS, document_split


def toy(n_docs=20, per_doc=25, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_docs):
        for j in range(per_doc):
            rows.append({"mol_id": f"M{d:03d}{j:03d}", "smiles": "CC", "y": rng.normal(),
                         "n_docs": 1, "first_year": 2000 + d % 10, "first_doc": f"DOC{d}"})
    return pd.DataFrame(rows)


def test_registered():
    assert SPLITS["document"] is document_split


def test_no_shared_document():
    df = toy()
    for seed in range(5):
        tr, te = document_split(df, seed)
        assert len(set(tr) & set(te)) == 0
        shared = set(df["first_doc"].iloc[tr]) & set(df["first_doc"].iloc[te])
        assert shared == set(), f"документ попал в обе части: {shared}"


def test_covers_all_rows_and_test_fraction():
    df = toy()
    tr, te = document_split(df, 0, test_frac=0.2)
    assert len(tr) + len(te) == len(df)
    assert 0.05 <= len(te) / len(df) <= 0.25


def test_seed_changes_assignment():
    df = toy()
    a = set(document_split(df, 0)[1])
    b = set(document_split(df, 7)[1])
    assert a != b


def test_large_document_goes_to_train():
    """Документ крупнее теста целиком уходит в train, а не съедает весь тест."""
    df = pd.concat([toy(n_docs=4, per_doc=10),
                    pd.DataFrame([{"mol_id": f"BIG{j}", "smiles": "CC", "y": 0.0, "n_docs": 1,
                                   "first_year": 2005, "first_doc": "HUGE"} for j in range(200)])],
                   ignore_index=True)
    for seed in range(5):
        tr, te = document_split(df, seed, test_frac=0.2)
        assert "HUGE" not in set(df["first_doc"].iloc[te])


def test_requires_column():
    df = toy().drop(columns=["first_doc"])
    with pytest.raises(KeyError):
        document_split(df, 0)
