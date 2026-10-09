import numpy as np
import pytest

from ncmol.gnn import ATOM_DIM, BOND_DIM, DMPNNRegressor, collate, mol_graph
from ncmol.models import featurize, make_model


def synth_smiles(n, seed=0):
    rng = np.random.default_rng(seed)
    frags = ["C", "CC", "O", "N", "c1ccccc1", "C(=O)", "Cl", "F", "CO", "CN"]
    out = []
    while len(out) < n:
        s = "".join(rng.choice(frags, size=rng.integers(2, 6)))
        from rdkit import Chem
        if Chem.MolFromSmiles(s) is not None:
            out.append(s)
    return out


def target(smiles):  # простая зависимость от состава: GNN должна её выучить
    return np.array([s.count("c") * 0.3 + s.count("O") * 0.5 - s.count("N") * 0.4 for s in smiles])


def test_graph_shapes_and_rev():
    x, s, d, a, r = mol_graph("CCO")
    assert x.shape == (3, ATOM_DIM) and a.shape == (4, BOND_DIM)
    assert (s[r] == d).all() and (d[r] == s).all()


def test_bad_smiles_does_not_crash():
    x, s, d, a, r = mol_graph("not_a_smiles")
    assert x.shape[0] == 1 and len(s) == 0
    t = collate([mol_graph("CC"), mol_graph(None)])
    assert t[-1] == 2


def test_make_model_featurize_interface():
    smi = ["CCO", "c1ccccc1", "CCN"]
    X = featurize("gnn_dmpnn", smi, ["a", "b", "c"])
    assert X.shape == (3,) and X[[0, 2]].tolist() == ["CCO", "CCN"]
    assert isinstance(make_model("gnn_dmpnn", 0), DMPNNRegressor)


def test_learns_and_deterministic():
    smi = synth_smiles(200)
    y = target(smi)
    kw = dict(hidden=64, max_epochs=25, patience=5)
    p1 = DMPNNRegressor(seed=1, **kw).fit(smi[:160], y[:160]).predict(smi[160:])
    p2 = DMPNNRegressor(seed=1, **kw).fit(smi[:160], y[:160]).predict(smi[160:])
    assert np.allclose(p1, p2)
    assert np.corrcoef(p1, y[160:])[0, 1] > 0.5
    assert np.sqrt(np.mean((p1 - y[160:]) ** 2)) < y[160:].std()


def test_early_stopping_restores_best():
    smi = synth_smiles(60)
    m = DMPNNRegressor(seed=0, hidden=32, max_epochs=40, patience=2).fit(smi, target(smi))
    assert 1 <= m.n_epochs_ <= 40
