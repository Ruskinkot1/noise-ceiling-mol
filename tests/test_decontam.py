"""Тесты деконтаминации на синтетике (без сети, без data/)."""
import datetime as dt
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from ncmol.chem import standardize
from ncmol.decontam import (aggregate_molecules, annotate, apply_year_filter, decontaminate_task, load_cutoffs,
                            load_registry, max_similarity, model_overlap, std_keys)

IBU = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
IBU_ANALOG = "CCC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Tanimoto ECFP4 ~0.71 к ибупрофену
IBU_STEREO = "C[C@H](C(=O)O)c1ccc(CC(C)C)cc1"  # тот же ik14, другой полный ключ
CAFF = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
NAPH = "c1ccc2ccccc2c1"
PARA = "CC(=O)Nc1ccc(O)cc1"
THR = 0.7


def make_registry(tmp_path):
    rows = []
    for smi, src in [(IBU, "moleculenet_x"), (CAFF, "tdc_y")]:
        s, k14, kf = std_keys(smi)
        rows.append(dict(inchikey14=k14, inchikey_full=kf, smiles=s, source=src, url="u", downloaded="2026-01-01"))
    d = tmp_path / "registry"
    d.mkdir()
    df = pd.DataFrame(rows)
    for src, g in df.groupby("source"):
        g.to_csv(d / f"{src}.csv.gz", index=False)
    return d


def make_data():
    """5 молекул: ibu_stereo (ik14 совпал, полный нет), caff (точно), analog (близкий), naph и para (чистые)."""
    smi = {"ibu_stereo": IBU_STEREO, "caff": CAFF, "analog": IBU_ANALOG, "naph": NAPH, "para": PARA}
    rows = []
    for name, s in smi.items():
        s_std, k14 = standardize(s)
        for doc, year in [("D1", 2015), ("D2", 2024)]:
            rows.append(dict(mol_id=k14, smiles=s_std, doc_id=doc, assay_id="A" + doc, year=year, y=5.0 + len(name) * 0.1))
    meas = pd.DataFrame(rows)
    return meas, aggregate_molecules(meas)


def test_std_keys_full_vs_14():
    s1, k1, f1 = std_keys(IBU)
    s2, k2, f2 = std_keys(IBU_STEREO)
    assert k1 == k2 and f1 != f2 and f1.startswith(k1)
    assert std_keys("not a smiles") == (None, None, None)


def test_max_similarity_basic():
    sim = max_similarity([IBU, IBU_ANALOG, NAPH], [IBU])
    assert sim[0] == pytest.approx(1.0)
    assert 0.7 < sim[1] < 0.8
    assert sim[2] < 0.3
    assert max_similarity([IBU], []).tolist() == [0.0]


def test_annotate_flags(tmp_path):
    reg = load_registry(make_registry(tmp_path))
    assert len(reg) == 2
    meas, mols = make_data()
    info = annotate(mols, reg, THR)
    k = lambda s: standardize(s)[1]  # noqa: E731
    assert info.loc[k(IBU_STEREO), "exact14"] and not info.loc[k(IBU_STEREO), "exact_full"]
    assert info.loc[k(CAFF), "exact14"] and info.loc[k(CAFF), "exact_full"]
    assert info.loc[k(CAFF), "exact_sources"] == "tdc_y"
    assert info.loc[k(IBU_ANALOG), "near"] and not info.loc[k(IBU_ANALOG), "exact14"]
    assert not info.loc[k(NAPH), "near"] and not info.loc[k(PARA), "exact14"]


def test_decontaminate_task_loss_log(tmp_path):
    reg = load_registry(make_registry(tmp_path))
    meas, mols = make_data()
    info = annotate(mols, reg, THR)
    cm, co, log = decontaminate_task(meas, mols, info, threshold=THR, cutoff_year=2020)
    steps = {r["step"]: r for r in log}
    assert steps["raw"]["mols_left"] == 5 and steps["raw"]["meas_left"] == 10
    assert steps["exact_inchikey_full"]["mols_removed"] == 1       # caffeine
    assert steps["exact_ik14(доп.)"]["mols_removed"] == 1          # ibuprofen stereo
    assert steps[f"near_tanimoto>{THR}"]["mols_removed"] == 1      # analog
    assert steps["year>2020"]["meas_removed"] == 2                 # D1(2015) у двух оставшихся молекул
    assert sorted(co["smiles"]) == sorted([standardize(NAPH)[0], standardize(PARA)[0]])
    assert set(cm["year"]) == {2024}
    # баланс: потери + остаток = исходное
    assert sum(r["meas_removed"] for r in log) + log[-1]["meas_left"] == 10
    assert (co["n_docs"] == 1).all() and (co["first_year"] == 2024).all()


def test_year_filter_boundary():
    meas, _ = make_data()
    assert set(apply_year_filter(meas, 2015)["year"]) == {2024}   # год отсечки включительно считается виденным
    assert len(apply_year_filter(meas, 2030)) == 0


def test_model_overlap(tmp_path):
    reg = load_registry(make_registry(tmp_path))
    _, mols = make_data()
    info = annotate(mols, reg, THR)
    ov = model_overlap(info, mols["mol_id"], {"m_known": ["tdc_y"], "m_unknown": []})
    assert ov["m_known"] == pytest.approx(1 / 5)
    assert ov["m_unknown"] is None


def test_cutoffs_rules(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("models:\n  a: {training_cutoff: 2021-05-01, paper_date: 2022-01-01}\n"
                 "  b: {paper_date: 2022-09-05}\n  c: {}\n")
    c = load_cutoffs(p, today=dt.date(2026, 10, 9))
    assert (c["a"].year, c["a"].verified, c["a"].basis) == (2021, True, "training_cutoff")
    assert (c["b"].year, c["b"].verified, c["b"].basis) == (2022, False, "paper_date")
    assert (c["c"].year, c["c"].verified, c["c"].basis) == (2026, False, "today")
    real = load_cutoffs(Path(__file__).resolve().parents[1] / "configs/model_cutoffs.yaml")
    assert real["chemberta2_77m"].year == 2022 and not real["chemberta2_77m"].verified


def test_script_end_to_end(tmp_path, monkeypatch, capsys):
    reg_dir = make_registry(tmp_path)
    proc = tmp_path / "processed"
    proc.mkdir()
    meas, mols = make_data()
    meas.to_csv(proc / "T1_IC50.measurements.csv", index=False)
    mols.to_csv(proc / "T1_IC50.molecules.csv", index=False)
    cut = tmp_path / "cut.yaml"
    cut.write_text("models:\n  m1: {paper_date: 2020-01-01, registry_sources: [tdc_y]}\n  m2: {}\n")
    script = Path(__file__).resolve().parents[1] / "scripts/02b_decontaminate.py"
    mod = runpy.run_path(str(script), run_name="not_main")
    base = ["02b", "--processed", str(proc), "--raw-out", str(tmp_path / "raw"), "--clean-out", str(tmp_path / "clean"),
            "--registry", str(reg_dir), "--cutoffs", str(cut), "--endpoints", str(tmp_path / "none.yaml"),
            "--threshold", str(THR), "--n-jobs", "1"]
    # pilot: ничего не пишет
    monkeypatch.setattr(sys, "argv", base + ["--pilot"])
    assert mod["main"]() == 0
    assert "PILOT" in capsys.readouterr().out and not (tmp_path / "clean").exists()
    # полный прогон; m2 без даты -> сегодня -> всё отфильтровано по году; явная отсечка 2020 -> остаются 2024
    monkeypatch.setattr(sys, "argv", base + ["--cutoff-year", "2020"])
    assert mod["main"]() == 0
    raw = pd.read_csv(tmp_path / "raw/T1_IC50.measurements.csv")
    clean = pd.read_csv(tmp_path / "clean/T1_IC50.measurements.csv")
    assert len(raw) == 10 and len(clean) == 2 and set(clean["year"]) == {2024}
    loss = pd.read_csv(tmp_path / "clean/decontam_loss.csv")
    assert list(loss["step"])[0] == "raw" and loss["task"].nunique() == 1
    mod_rows = pd.read_csv(tmp_path / "clean/decontam_models.csv").set_index("model")
    assert str(mod_rows.loc["m1", "test_overlap_share"]) == "0.2" and mod_rows.loc["m2", "test_overlap_share"] == "н/д"
    assert not mod_rows.loc["m2", "verified"]
