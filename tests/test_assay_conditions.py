import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("assay_conditions", Path(__file__).resolve().parents[1] / "scripts/12_assay_conditions.py")
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)


def test_parse_atp_units_and_formats():
    assert ac.parse_atp("ATP at 10 uM") == 10.0
    assert ac.parse_atp("using 100 μM ATP") == 100.0
    assert ac.parse_atp("ATP concentration of 1 mM") == 1000.0
    assert ac.parse_atp("10 uM [gamma-33P]ATP") == 10.0
    assert abs(ac.parse_atp("ATP 20 nM") - 0.02) < 1e-12
    assert ac.parse_atp("ATP (5 µM)") == 5.0


def test_parse_atp_no_value():
    assert np.isnan(ac.parse_atp("at Km ATP"))
    assert np.isnan(ac.parse_atp(None))
    assert np.isnan(ac.parse_atp("kinase inhibition assay"))


def test_fit_km_recovers_synthetic_km():
    rng = np.random.default_rng(0)
    import pandas as pd
    a, b = rng.choice([1.0, 10.0, 100.0, 1000.0], 400), rng.choice([1.0, 10.0, 100.0, 1000.0], 400)
    km = 30.0
    delta = -np.log10((1 + a / km) / (1 + b / km)) + rng.normal(0, 0.05, 400)
    est, _ = ac.fit_km(pd.DataFrame({"atp_a": a, "atp_b": b, "delta": delta}), np.logspace(-1, 3, 81))
    assert 20 < est < 45
