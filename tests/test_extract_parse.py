import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("extract", Path(__file__).resolve().parents[1] / "scripts/01_extract_chembl.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


def rec(**kw):
    base = dict(target_chembl_id="CHEMBL1", target_pref_name="T", molecule_chembl_id="CHEMBL9", canonical_smiles="CCO",
                standard_type="Ki", standard_relation="=", pchembl_value="6.5", assay_chembl_id="CHEMBL100",
                document_chembl_id="CHEMBL200", document_year=2010, data_validity_comment=None, potential_duplicate=0)
    base.update(kw)
    return base


def test_parse_filters_and_schema():
    recs = [rec(), rec(standard_relation=">"), rec(pchembl_value=None), rec(data_validity_comment="Outside typical range"),
            rec(assay_chembl_id="CHEMBL101"), rec(potential_duplicate=1), rec(document_year=None)]
    df = ex.parse_activities(recs, assay_ok={"CHEMBL100"})
    assert list(df.columns) == ex.RAW_COLUMNS
    assert len(df) == 1 and df.pchembl_value.iloc[0] == 6.5 and df.doc_chembl_id.iloc[0] == "CHEMBL200"


class FakeSession:
    def __init__(self):
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        pages = {0: {"activities": [rec()], "page_meta": {"next": "/p2"}},
                 1: {"activities": [rec(molecule_chembl_id="CHEMBL10")], "page_meta": {"next": None}}}
        p = pages[self.calls - 1]

        class R:
            def raise_for_status(self): pass
            def json(self): return p
        return R()


def test_pagination_follows_next():
    s = FakeSession()
    out = list(ex.paginate("http://x/a", {}, "activities", s))
    assert len(out) == 2 and s.calls == 2


def test_frac_ceiling_anchor_points():
    from ncmol.metrics import frac_ceiling
    assert frac_ceiling(1.0, 1.0, 0.4) == 0.0
    assert abs(frac_ceiling(0.4, 1.0, 0.4) - 1.0) < 1e-12
