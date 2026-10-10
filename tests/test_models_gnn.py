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


def _load_embed_script():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "06_embed_foundation.py"
    spec = importlib.util.spec_from_file_location("embed06", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_embed_masked_mean_pooling_offline():
    """Крошечная случайная RoBERTa + символьный токенайзер: паддинг не должен влиять на эмбеддинг."""
    torch = pytest.importorskip("torch")
    from transformers import RobertaConfig, RobertaModel
    mod = _load_embed_script()
    torch.manual_seed(0)
    model = RobertaModel(RobertaConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                       intermediate_size=32, max_position_embeddings=64, pad_token_id=1),
                         add_pooling_layer=False).eval()

    def tok(batch, return_tensors, padding, truncation, max_length):
        ids = [[2 + (ord(c) % 60) for c in s][:max_length] for s in batch]
        L = max(map(len, ids))
        x = torch.tensor([i + [1] * (L - len(i)) for i in ids])
        return {"input_ids": x, "attention_mask": (x != 1).long()}

    smi = ["CCO", "c1ccccc1C(=O)O", "CN", "CCN(CC)CC"]
    E = mod.embed_smiles(smi, tok, model, batch_size=2)
    alone = np.stack([mod.embed_smiles([s], tok, model, batch_size=1)[0] for s in smi])
    assert E.shape == (4, 16) and np.allclose(E, alone, atol=1e-5)


def test_embed_cache_roundtrip_and_featurize(tmp_path):
    ids = ["a", "b", "c"]
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez(tmp_path / "fake-model.npz", mol_ids=np.array(ids, dtype=object), X=X)
    out = featurize("emb-fake-model_gbm", ["C", "CC", "CCC"], ["c", "a"], emb_dir=str(tmp_path))
    assert np.array_equal(out, X[[2, 0]])


def test_resolve_device():
    import pytest
    from ncmol.device import resolve_device
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")
    with pytest.raises(ValueError):
        resolve_device("tpu")
    import torch
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        with pytest.raises(RuntimeError):
            resolve_device("mps")


def test_gnn_accepts_device_cpu():
    import numpy as np
    from ncmol.gnn import DMPNNRegressor
    smi = ["CCO", "CCN", "c1ccccc1", "CCC(=O)O", "CC(C)O", "c1ccncc1"] * 5
    y = np.arange(len(smi), dtype=float) % 5
    m = DMPNNRegressor(seed=0, max_epochs=2, device="cpu").fit(smi, y)
    assert m.predict(smi).shape == (len(smi),)
