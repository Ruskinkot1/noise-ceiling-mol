"""Деконтаминация: реестр «уже виденных» молекул, точные совпадения, близкие аналоги, фильтр по году.

Ключевые соглашения
- Структуры стандартизуются через ncmol.chem.standardize (та же функция, что в scripts/02_standardize.py).
  Полный InChIKey считается по стандартизованному SMILES (chem.standardize возвращает только 1-й блок).
- Близость: Tanimoto по битовому ECFP4 (Morgan r=2, 2048 бит).
- Год документа: keep только doc year > cutoff year (год отсечения включительно считается «виденным»).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from .chem import standardize

RDLogger.DisableLog("rdApp.*")
REGISTRY_COLS = ["inchikey14", "inchikey_full", "smiles", "source", "url", "downloaded"]
_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ---------------------------------------------------------------- стандартизация
def std_keys(smiles):
    """-> (std_smiles, inchikey14, inchikey_full) или (None, None, None)."""
    smi, ik14 = standardize(smiles)
    if smi is None or not ik14:
        return None, None, None
    try:
        full = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
    except Exception:
        full = None
    return smi, ik14, full or None


def standardize_table(smiles_list, n_jobs: int = 1) -> pd.DataFrame:
    """Стандартизует список SMILES; невалидные отбрасываются. Колонки: smiles, inchikey14, inchikey_full."""
    smiles_list = list(smiles_list)
    if n_jobs > 1 and len(smiles_list) > 5000:
        from multiprocessing import Pool
        with Pool(n_jobs) as p:
            res = p.map(std_keys, smiles_list, chunksize=2000)
    else:
        res = [std_keys(s) for s in smiles_list]
    df = pd.DataFrame(res, columns=["smiles", "inchikey14", "inchikey_full"])
    return df.dropna(subset=["smiles", "inchikey14"]).reset_index(drop=True)


# ---------------------------------------------------------------- реестр
def load_registry(reg_dir, sources=None) -> pd.DataFrame:
    """Читает data/registry/*.csv.gz (кроме служебных). sources — список имён или None."""
    reg_dir = Path(reg_dir)
    parts = []
    for f in sorted(reg_dir.glob("*.csv.gz")):
        name = f.name[: -len(".csv.gz")]
        if sources is not None and name not in sources:
            continue
        parts.append(pd.read_csv(f))
    if not parts:
        return pd.DataFrame(columns=REGISTRY_COLS)
    return pd.concat(parts, ignore_index=True)


def fingerprints(smiles_list):
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        fps.append(_FPGEN.GetFingerprint(m) if m is not None else DataStructs.ExplicitBitVect(2048))
    return fps


def _max_sim_chunk(args):
    q_fps, ref_fps, ref_counts, min_sim = args
    out = np.zeros(len(q_fps), dtype=np.float32)
    for i, q in enumerate(q_fps):
        a = q.GetNumOnBits()
        if a == 0:
            continue
        if min_sim:
            # Tanimoto <= min(a,b)/max(a,b): ref с числом бит вне [min_sim*a, a/min_sim] не могут дать sim >= min_sim
            lo = np.searchsorted(ref_counts, np.ceil(min_sim * a - 1e-9), side="left")
            hi = np.searchsorted(ref_counts, np.floor(a / min_sim + 1e-9), side="right")
            window = ref_fps[lo:hi]
        else:
            window = ref_fps
        if len(window):
            out[i] = max(DataStructs.BulkTanimotoSimilarity(q, window))
    return out


def max_similarity(query_smiles, ref_smiles, n_jobs: int = 1, min_sim: float | None = None) -> np.ndarray:
    """Максимум Tanimoto(ECFP4) каждого запроса к набору ref. Пустой ref -> нули.
    min_sim=None: точные значения. min_sim=t: значения >= t точны, остальные — нижняя оценка (часть ref отсечена по
    числу бит), достаточно для проверки «> порога»."""
    query_smiles, ref_smiles = list(query_smiles), list(ref_smiles)
    if not query_smiles or not ref_smiles:
        return np.zeros(len(query_smiles), dtype=np.float32)
    ref = fingerprints(ref_smiles)
    counts = np.array([f.GetNumOnBits() for f in ref])
    order = np.argsort(counts, kind="stable")
    ref = [ref[i] for i in order]
    counts = counts[order]
    q = fingerprints(query_smiles)
    if n_jobs > 1 and len(q) > 200:
        from multiprocessing import Pool
        chunks = [(q[i:i + 200], ref, counts, min_sim) for i in range(0, len(q), 200)]
        with Pool(n_jobs) as p:
            return np.concatenate(p.map(_max_sim_chunk, chunks))
    return _max_sim_chunk((q, ref, counts, min_sim))


# ---------------------------------------------------------------- отсечки моделей
@dataclass
class Cutoff:
    model: str
    date: dt.date
    verified: bool
    basis: str  # 'training_cutoff' | 'paper_date' | 'today'

    @property
    def year(self) -> int:
        return self.date.year


def load_cutoffs(path, today: dt.date | None = None) -> dict[str, Cutoff]:
    """Эффективная отсечка: training_cutoff -> иначе paper_date (верхняя граница) -> иначе сегодня («не проверено»)."""
    today = today or dt.date.today()
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    out = {}
    for name, m in (cfg.get("models") or {}).items():
        tc, pd_ = m.get("training_cutoff"), m.get("paper_date")
        if tc:
            out[name] = Cutoff(name, _d(tc), True, "training_cutoff")
        elif pd_:
            out[name] = Cutoff(name, _d(pd_), False, "paper_date")
        else:
            out[name] = Cutoff(name, today, False, "today")
    return out


def _d(x) -> dt.date:
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))


def model_registry_sources(path) -> dict[str, list[str]]:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    return {k: list(v.get("registry_sources") or []) for k, v in (cfg.get("models") or {}).items()}


# ---------------------------------------------------------------- агрегация как в 02_standardize
def aggregate_molecules(meas: pd.DataFrame) -> pd.DataFrame:
    """measurements -> molecules (mol_id, smiles, y, n_docs, first_year); та же логика, что scripts/02."""
    if meas.empty:
        return pd.DataFrame(columns=["mol_id", "smiles", "y", "n_docs", "first_year"])
    d = meas.groupby(["mol_id", "doc_id"], as_index=False).agg(
        smiles=("smiles", "first"), year=("year", "first"), y=("y", "mean"))
    return d.groupby("mol_id").agg(smiles=("smiles", "first"), y=("y", "mean"),
                                   n_docs=("doc_id", "nunique"), first_year=("year", "min")).reset_index()


# ---------------------------------------------------------------- шаги очистки
def annotate(mols: pd.DataFrame, registry: pd.DataFrame, threshold: float = 0.8, n_jobs: int = 1,
             exact_sim: bool = False) -> pd.DataFrame:
    """Для уникальных молекул (колонки mol_id, smiles) считает флаги относительно реестра.
    Индекс результата = mol_id. Колонки: ik14, ikfull, exact14, exact_full, exact_sources, max_sim, near.
    max_sim = 1.0 для точных совпадений; для остальных максимум Tanimoto ECFP4 к реестру.
    exact_sim=False: значения ниже порога — нижняя оценка (отсечение по числу бит), быстрее; True — точные (для пилота)."""
    mols = mols.drop_duplicates("mol_id").reset_index(drop=True)
    st = standardize_table_idx(mols["smiles"])
    out = pd.DataFrame({"mol_id": mols["mol_id"], "smiles": mols["smiles"],
                        "ik14": st["inchikey14"].fillna(mols["mol_id"]), "ikfull": st["inchikey_full"]})
    if registry.empty:
        out["exact14"] = out["exact_full"] = False
        out["exact_sources"] = ""
    else:
        src14 = registry.groupby("inchikey14")["source"].agg(lambda x: ";".join(sorted(set(x))))
        out["exact14"] = out["ik14"].isin(src14.index)
        out["exact_full"] = out["ikfull"].isin(set(registry["inchikey_full"].dropna()))
        out["exact_sources"] = out["ik14"].map(src14).fillna("")
    out["max_sim"] = np.float32(0.0)
    out.loc[out["exact14"], "max_sim"] = 1.0
    todo = np.where(~out["exact14"].values)[0]
    if len(todo) and not registry.empty:
        ref = registry.drop_duplicates("inchikey14")["smiles"]
        out.loc[out.index[todo], "max_sim"] = max_similarity(out["smiles"].values[todo], ref, n_jobs=n_jobs,
                                                          min_sim=None if exact_sim else threshold)
    out["near"] = (out["max_sim"] > threshold) & ~out["exact14"]
    return out.set_index("mol_id")


def standardize_table_idx(smiles: pd.Series) -> pd.DataFrame:
    """Как standardize_table, но сохраняет выравнивание по строкам (NaN для невалидных)."""
    res = [std_keys(s) for s in smiles]
    return pd.DataFrame(res, columns=["smiles", "inchikey14", "inchikey_full"], index=smiles.index)


def apply_year_filter(meas: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    """Оставляет измерения из документов с year > cutoff_year."""
    return meas[meas["year"] > cutoff_year]


def decontaminate_task(meas: pd.DataFrame, mols: pd.DataFrame, info: pd.DataFrame, *,
                       threshold: float = 0.8, cutoff_year: int | None = None):
    """info = annotate(...) (индекс mol_id). Шаги: exact_full -> exact_ik14 -> near -> year.
    Возвращает (clean_meas, clean_mols, log_rows); log_rows: step, mols_removed, meas_removed, mols_left, meas_left."""
    log = [dict(step="raw", mols_removed=0, meas_removed=0, mols_left=len(mols), meas_left=len(meas))]
    inf = info.reindex(mols["mol_id"])
    cur_meas, cur_mols = meas, mols

    def drop(ids, step):
        nonlocal cur_meas, cur_mols
        m_new = cur_meas[~cur_meas["mol_id"].isin(ids)]
        o_new = cur_mols[~cur_mols["mol_id"].isin(ids)]
        log.append(dict(step=step, mols_removed=len(cur_mols) - len(o_new), meas_removed=len(cur_meas) - len(m_new),
                        mols_left=len(o_new), meas_left=len(m_new)))
        cur_meas, cur_mols = m_new, o_new

    near = (inf["max_sim"].fillna(0) > threshold) & ~inf["exact14"].fillna(False).astype(bool)
    drop(set(inf.index[inf["exact_full"].fillna(False).astype(bool)]), "exact_inchikey_full")
    drop(set(inf.index[inf["exact14"].fillna(False).astype(bool)]), "exact_ik14(доп.)")
    drop(set(inf.index[near]), f"near_tanimoto>{threshold}")
    if cutoff_year is not None:
        n_m0, n_x0 = len(cur_mols), len(cur_meas)
        cur_meas = apply_year_filter(cur_meas, cutoff_year)
        cur_mols = aggregate_molecules(cur_meas)
        log.append(dict(step=f"year>{cutoff_year}", mols_removed=n_m0 - len(cur_mols), meas_removed=n_x0 - len(cur_meas),
                        mols_left=len(cur_mols), meas_left=len(cur_meas)))
    return cur_meas, cur_mols, log


def model_overlap(info: pd.DataFrame, mol_ids, model_sources: dict[str, list[str]]) -> dict[str, float | None]:
    """Доля молекул (mol_ids из raw) с точным совпадением ik14 с известным обучающим набором модели.
    None = обучающий набор модели не входит в реестр (не проверено / корпус не загружался)."""
    sub = info.reindex(list(mol_ids))
    out = {}
    for model, srcs in model_sources.items():
        if not srcs or sub.empty:
            out[model] = None
            continue
        hit = sub["exact_sources"].fillna("").apply(lambda x: bool(set(x.split(";")) & set(srcs)) if x else False)
        out[model] = float(hit.mean())
    return out
