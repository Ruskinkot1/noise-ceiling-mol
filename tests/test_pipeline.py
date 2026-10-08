import numpy as np
import pandas as pd

from ncmol.ceiling import estimate_ceiling
from ncmol.splits import SPLITS


def synth(n=2000, sigma=0.5, docs=3, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.normal(0, 1, n)
    rows = [(f"m{i}", f"d{j}", t[i] + rng.normal(0, sigma)) for i in range(n) for j in range(docs)]
    return pd.DataFrame(rows, columns=["mol_id", "doc_id", "y"])


def test_ceiling_recovers_sigma():
    c = estimate_ceiling(synth(), mean_label=False, max_abs_pair_diff=None)
    assert abs(c.sigma - 0.5) < 0.03
    # Var(y)=1+0.25 -> R2_max = 1-0.25/1.25 = 0.8
    assert abs(c.r2_max - 0.8) < 0.03


def test_mean_label_lowers_floor():
    d = synth()
    assert estimate_ceiling(d, mean_label=True).rmse_floor < estimate_ceiling(d, mean_label=False).rmse_floor


def test_splits_disjoint_and_complete():
    rings = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccsc1", "c1ccoc1", "C1CCNCC1"]
    smi = [rings[i % 6] + "C" * (i % 5) + "O" * (i % 3) for i in range(60)]
    df = pd.DataFrame({"smiles": smi, "first_year": np.arange(60) % 15 + 2000})
    for f in SPLITS.values():
        tr, te = f(df, 0)
        assert not set(tr) & set(te) and len(tr) + len(te) == len(df) and len(te) > 0
