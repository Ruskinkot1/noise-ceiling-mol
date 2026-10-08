"""Стандартизация структур и признаки."""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
_uncharger = rdMolStandardize.Uncharger()
_chooser = rdMolStandardize.LargestFragmentChooser()


def standardize(smiles: str):
    """Самый большой фрагмент -> нейтрализация -> канонический SMILES.
    Возвращает (smiles, inchikey_connectivity) или (None, None)."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None, None
    try:
        mol = _uncharger.uncharge(_chooser.choose(mol))
        smi = Chem.MolToSmiles(mol)
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None, None
    # первый блок InChIKey = связность (без стереохимии): стереоизомеры считаем одной молекулой
    return smi, (key.split("-")[0] if key else None)


def scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:
        return ""


def ecfp(smiles_list, radius: int = 2, n_bits: int = 2048, counts: bool = True) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out[i] = gen.GetCountFingerprintAsNumPy(m) if counts else gen.GetFingerprintAsNumPy(m)
    return out
