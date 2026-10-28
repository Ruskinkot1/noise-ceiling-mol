#!/usr/bin/env python
"""Registry of "already seen" molecules from open benchmarks.

Sources (all without registration):
  moleculenet : deepchem files on S3 (deepchemdata.s3-us-west-1.amazonaws.com/datasets)
  tdc         : direct Harvard Dataverse files (api/access/datafile/<id>); ids taken from tdc/metadata.py (pytdc 1.1.15),
                PyTDC is not installed: the same files are downloaded directly
  polaris     : polaris-lib (if installed and load_dataset works without login); the direct parquet URL requires a signature

Model pretraining corpora (PubChem 77M, PubChem+ZINC 1.1B) are NOT downloaded: see configs/model_cutoffs.yaml.

Output: data/registry/<source>.csv.gz with columns inchikey14, inchikey_full, smiles, source, url, downloaded
(smiles = standardized by ncmol.chem.standardize; needed for finding close analogues) +
data/registry/_status.csv (what was downloaded, HTTP status, size, number of rows/unique structures).
Raw files go to data/registry/_raw/ (not tracked by git: data/* is in .gitignore).
"""
import argparse
import datetime as dt
import io
import sys
from pathlib import Path

import pandas as pd
import requests

from ncmol.decontam import REGISTRY_COLS, standardize_table

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 500 * 1024 * 1024  # hard limit per file

MN_BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"
MOLECULENET = {  # source name -> file
    "esol": "delaney-processed.csv", "freesolv": "SAMPL.csv", "lipophilicity": "Lipophilicity.csv",
    "bace": "bace.csv", "hiv": "HIV.csv", "bbbp": "BBBP.csv", "tox21": "tox21.csv.gz",
    "clintox": "clintox.csv.gz", "sider": "sider.csv.gz", "toxcast": "toxcast_data.csv.gz", "muv": "muv.csv.gz",
}
TDC_BASE = "https://dataverse.harvard.edu/api/access/datafile"
TDC = {  # name -> dataverse file id (pytdc 1.1.15 tdc/metadata.py, name2id)
    "lipophilicity_astrazeneca": 4259595, "solubility_aqsoldb": 4259610, "hydrationfreeenergy_freesolv": 4259594,
    "caco2_wang": 4259569, "pampa_ncats": 6695858, "approved_pampa_ncats": 6695857, "hia_hou": 4259591,
    "pgp_broccatelli": 4259597, "bioavailability_ma": 4259567, "vdss_lombardo": 4267387,
    "cyp2c19_veith": 4259576, "cyp2d6_veith": 4259580, "cyp3a4_veith": 4259582, "cyp1a2_veith": 4259573,
    "cyp2c9_veith": 4259577, "cyp2c9_substrate_carbonmangels": 4259584, "cyp2d6_substrate_carbonmangels": 4259578,
    "cyp3a4_substrate_carbonmangels": 4259581, "bbb_martins": 4259566, "b3db_classification": 7878566,
    "b3db_regression": 7878567, "ppbr_az": 6413140, "half_life_obach": 4266799,
    "clearance_hepatocyte_az": 4266187, "clearance_microsome_az": 4266186, "hlm": 10218426, "rlm": 10218425,
    "tox21": 4259612, "toxcast": 4259613, "clintox": 4259572, "herg_karim": 6822246, "herg": 4259588,
    "dili": 4259585, "skin_reaction": 4259609, "ames": 4259564, "carcinogens_lagunin": 4259570,
    "ld50_zhu": 4267146, "hiv": 4259593, "sarscov2_3clpro_diamond": 4259606, "sarscov2_vitro_touret": 4259607,
}
SMILES_COLS = ["smiles", "mol", "drug", "x", "canonical_smiles", "mol_smiles", "smiles_text"]  # "x": structure column in TDC single_pred
POLARIS_API = "https://polarishub.io/api/v1/dataset"


def fetch(url, dest: Path, session):
    """Downloads with a size-limit check. -> (http_status, nbytes, error)"""
    if dest.exists() and dest.stat().st_size > 0:
        return 200, dest.stat().st_size, "cache"
    try:
        r = session.get(url, stream=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        return None, 0, f"{type(e).__name__}: {e}"
    if r.status_code != 200:
        return r.status_code, 0, "http"
    n = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            n += len(chunk)
            if n > MAX_BYTES:
                f.close()
                dest.unlink()
                return 200, n, "too_big(>500MB), not downloaded"
            f.write(chunk)
    return 200, n, ""


def read_table(path: Path) -> pd.DataFrame:
    head = open(path, "rb").read(4096) if not str(path).endswith(".gz") else b""
    if str(path).endswith(".gz"):
        import gzip
        head = gzip.open(path, "rb").read(4096)
    first = head.split(b"\n")[0]
    sep = "\t" if first.count(b"\t") > first.count(b",") else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def smiles_column(df: pd.DataFrame):
    low = {c.lower(): c for c in df.columns}
    for c in SMILES_COLS:
        if c in low:
            return low[c]
    return None


def write_registry(source, smiles, url, out_dir, today, n_jobs):
    smiles = pd.Series(smiles).dropna().astype(str)
    st = standardize_table(smiles.unique(), n_jobs=n_jobs)
    st = st.drop_duplicates("inchikey_full")
    st["source"], st["url"], st["downloaded"] = source, url, today
    st = st[REGISTRY_COLS]
    st.to_csv(out_dir / f"{source}.csv.gz", index=False)
    return len(smiles), len(st), st["inchikey14"].nunique()


def build_polaris(out_dir, raw_dir, today, n_jobs, status, max_table_mb):
    """The Polaris metadata catalogue is open (GET /api/v1/dataset). The tables themselves are served via a storage token,
    for which polaris-lib requires an OAuth login (polaris/hub/storage.py, token-exchange) = registration -> not downloaded
    if polaris-lib is not installed or load_dataset fails without login."""
    try:
        meta = requests.get(POLARIS_API, timeout=60).json()["data"]
    except Exception as e:  # noqa: BLE001
        status.append(dict(source="polaris", ok=False, note=f"catalogue unavailable:{type(e).__name__}"))
        return
    cat = pd.DataFrame([dict(owner=(r.get("owner") or {}).get("slug"), slug=r["slug"], n_rows=r.get("nRows"),
                             table_bytes=(r.get("tableContent") or {}).get("size"), created=str(r.get("createdAt"))[:10],
                             year=(r.get("userAttributes") or {}).get("year"), source=r.get("source"),
                             license=r.get("license"), tags=";".join(r.get("tags") or []),
                             mol_cols=";".join(k for k, v in (r.get("annotations") or {}).items()
                                               if v.get("modality") == "MOLECULE")) for r in meta])
    cat["downloaded"] = today
    cat.to_csv(out_dir / "_polaris_catalog.csv", index=False)
    try:
        import polaris as po
    except Exception as e:  # noqa: BLE001
        status.append(dict(source="polaris", ok=False, n_rows=len(cat),
                           note=f"catalogue of {len(cat)} datasets saved; tables not downloaded: polaris-lib is not installed "
                                f"({type(e).__name__}) and downloading requires login"))
        return
    for r in meta:
        owner = (r.get("owner") or {}).get("slug")
        mol_cols = [k for k, v in (r.get("annotations") or {}).items() if v.get("modality") == "MOLECULE"]
        size = (r.get("tableContent") or {}).get("size") or 0
        slug = f"{owner}/{r['slug']}"
        if not mol_cols or owner == "tdcommons" or size > max_table_mb * 1e6:
            continue  # tdcommons mirrors duplicate TDC; large tables are skipped
        source = f"polaris_{owner}_{r['slug']}".replace("-", "_")
        try:
            ds = po.load_dataset(slug)
            n, nu, nk = write_registry(source, ds.table[mol_cols[0]], f"https://polarishub.io/datasets/{slug}",
                                       out_dir, today, n_jobs)
            status.append(dict(source=source, ok=True, http=200, bytes=size, n_rows=n, n_unique=nu, n_ik14=nk))
        except Exception as e:  # noqa: BLE001
            status.append(dict(source=source, ok=False, note=f"{type(e).__name__}: {str(e)[:150]}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/registry"))
    ap.add_argument("--only", nargs="*", default=None, help="moleculenet tdc polaris")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--skip-done", action="store_true", help="do not rebuild sources that already have a csv.gz")
    ap.add_argument("--polaris-max-mb", type=float, default=30.0)
    a = ap.parse_args()
    out = Path(a.out)
    raw = out / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    session = requests.Session()
    status = []
    groups = a.only or ["moleculenet", "tdc", "polaris"]

    jobs = []
    if "moleculenet" in groups:
        jobs += [(f"moleculenet_{k}", f"{MN_BASE}/{v}", raw / f"moleculenet_{v}") for k, v in MOLECULENET.items()]
    if "tdc" in groups:
        jobs += [(f"tdc_{k}", f"{TDC_BASE}/{v}", raw / f"tdc_{k}.tab") for k, v in TDC.items()]
    for source, url, dest in jobs:
        if a.skip_done and (out / f"{source}.csv.gz").exists():
            continue
        code, nbytes, note = fetch(url, dest, session)
        row = dict(source=source, url=url, http=code, bytes=nbytes, ok=False, note=note)
        if code == 200 and dest.exists():
            try:
                df = read_table(dest)
                col = smiles_column(df)
                if col is None:
                    row["note"] = f"no SMILES column:{list(df.columns)[:6]}"
                else:
                    n, nu, nk = write_registry(source, df[col], url, out, today, a.n_jobs)
                    row.update(ok=True, n_rows=n, n_unique=nu, n_ik14=nk)
            except Exception as e:  # noqa: BLE001
                row["note"] = f"{type(e).__name__}: {str(e)[:150]}"
        status.append(row)
        print(row, flush=True)
    if "polaris" in groups:
        build_polaris(out, raw, today, a.n_jobs, status, a.polaris_max_mb)
    st = pd.DataFrame(status)
    st["downloaded"] = today
    old = out / "_status.csv"
    if old.exists() and a.only:  # partial run: update only the rebuilt sources
        prev = pd.read_csv(old)
        st = pd.concat([prev[~prev["source"].isin(st["source"])], st], ignore_index=True)
    st.to_csv(old, index=False)
    ok = st[st["ok"] == True]  # noqa: E712
    print("\nSucceeded:", len(ok), "of", len(st))
    print(st.drop(columns=[c for c in ["url"] if c in st]).to_string(index=False))
    allr = pd.concat([pd.read_csv(f, usecols=["inchikey14", "inchikey_full"]) for f in sorted(out.glob("*.csv.gz"))])
    print(f"\nTotal rows: {len(allr)}, unique InChIKey14: {allr['inchikey14'].nunique()}, "
          f"full:{allr['inchikey_full'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
