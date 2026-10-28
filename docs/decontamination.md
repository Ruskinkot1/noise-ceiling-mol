# Decontamination (W5)

Scripts: `scripts/build_registry.py` (registry), `scripts/02b_decontaminate.py` (cleaning), code `src/ncmol/decontam.py`,
model cutoffs `configs/model_cutoffs.yaml`, tests `tests/test_decontam.py`. Registry build date: 2026-10-09.

## 1. Registry of "already seen" molecules (`data/registry/`, not tracked in git)
Format `data/registry/<source>.csv.gz`: `inchikey14, inchikey_full, smiles, source, url, downloaded`.
Structures are standardized with `ncmol.chem.standardize` (same as the dataset); the full InChIKey is computed from the standardized SMILES.
`_status.csv` — HTTP status, size, row count for each source; `_raw/` — downloaded files; `_polaris_catalog.csv` — Polaris metadata.

| Group | Source and check | Files | Unique structures (full key, sum over files) |
|---|---|---|---|
| MoleculeNet | `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/<file>`; all 11 returned HTTP 200: ESOL (delaney-processed), FreeSolv (SAMPL), Lipophilicity, BACE, HIV, BBBP, Tox21, ClinTox, SIDER, ToxCast, MUV | 11 | 161958 |
| TDC | Harvard Dataverse `https://dataverse.harvard.edu/api/access/datafile/<id>`; ids taken from `tdc/metadata.py` (pytdc 1.1.15, downloaded from PyPI, not installed); all 40 returned HTTP 200: all ADME and Tox sets + HIV + 2 SARS-CoV-2 | 40 | 200524 |
| Polaris | The metadata catalog is open (`https://polarishub.io/api/v1/dataset`, 96 datasets). Tables are served via a storage token, polaris-lib requires an OAuth login (code `polaris/hub/storage.py`), pip installation of the lib on py3.13 failed to build (pyarrow) | 0 | not downloaded |

Total 362482 rows, 192953 unique InChIKey14, 202019 unique full keys (files overlap: MoleculeNet and TDC duplicate each other).

Deliberately not included: TDC HTS Butkiewicz (~100-340 thousand structures each), `herg_central`, BindingDB, DrugComb, QM, `chembl`/`zinc`/`moses` (corpora, not benchmarks);
MoleculeNet PCBA/QM9; MoleculeACE and Polaris sets (mostly ChEMBL, Polaris tables are behind a login) — **not verified** whether they are needed and directly available (the direct MoleculeACE URL was not checked).
The Polaris mirrors of TDC (owner `tdcommons`) duplicate the TDC already included.

## 2. Model pretraining corpora
Not downloaded (PubChem 77M, PubChem+ZINC 1.1B). Recorded in `configs/model_cutoffs.yaml` (model, corpus, date, source, verification flags).
Verified (arXiv API, 2026-10-09): ChemBERTa-2 — preprint 2022-09-05, "up to 77M compounds from PubChem"; MoLFormer — arXiv 2106.09553, v1 2021-06-17, last updated 2022-12-14, "1.1 billion molecules from PubChem and ZINC".
**Not verified**: the actual date of the PubChem/ZINC snapshot and the training-data cutoff for all models (field `training_cutoff: null`).
Effective cutoff rule: `training_cutoff` → otherwise the preprint date (training could not have ended later) → otherwise today's date (the latest possible, "not verified").
A document is kept if `doc year` is strictly greater than the cutoff year (the cutoff year itself is counted as seen, since a year has no month).
For `chemberta2_77m` the cutoff is 2022, for `molformer_xl` 2022; the clean set by default takes the maximum over the selected models.
Limitation: filtering by document year does not prove that a molecule is absent from PubChem (ChEMBL molecules and substances from old documents could have entered PubChem independently); it only removes documents published before the preprint. No direct check of overlap with PubChem was done.

## 3. Cleaning steps (`02b_decontaminate.py`)
1. Exact match on the full InChIKey, then on InChIKey14 (stereoisomers) — all measurements of the molecule are removed.
2. Close analogs: Tanimoto ECFP4 (Morgan r=2, 2048 bits, bit vectors) > threshold (0.8 by default) to any registry molecule. Speed-up: the registry is sorted by bit count, candidates outside the window `[t*a, a/t]` are not computed (for t=threshold the values ≥ t are exact, below that — a lower bound).
3. Document filter by year (see above); molecules are recomputed from the remaining measurements (mean over documents, `n_docs`, `first_year`).
Output: `data/processed_raw` (copy without cleaning), `data/processed_clean` (+ `tasks.csv`, `decontam_loss.csv` — losses by task and step, `decontam_models.csv` — cutoff, verification flag and the share of test-molecule overlap with the model's known training set; "unavailable" if the training set is not in the registry; currently "unavailable" for all models, since the lists of fine-tuning sets in the papers were not cross-checked).
Leakage = the difference in metrics between raw and clean.

## 4. Threshold pilot (random 5000 of 130113 unique molecules of data/processed at the time of the run, 58 tasks, seed 0)
Exact matches ik14: 403 (8.1%), full key 345 (6.9%); the main source is `tdc_herg_karim` (281, 5.6%), followed by BACE, Lipophilicity, TDC HLM/RLM, ToxCast/Tox21.
Maximum Tanimoto for molecules without an exact match (n=4597): median 0.46, q90 0.67, q95 0.74, q99 0.84.

| threshold | removed beyond exact | total with exact |
|---|---|---|
| >0.9 | 0.3% | 8.4% |
| >0.8 | 1.9% | 9.9% |
| >0.7 | 7.1% | 15.1% |
| >0.6 | 17.3% | 25.4% |
| >0.5 | 35.0% | 43.0% |

Rationale for 0.8: the dense bulk of the distribution lies at 0.3-0.6 (random proximity of drug-like molecules in ECFP4), the tail grows too fast from 0.7 (>0.7 already loses 7%, and typical "single substitution" analogs in ECFP4 give ~0.6-0.75, i.e. this is already a series, not a copy); 0.8 is the value from the plan, at a moderate cost (1.9% of molecules). This rationale is heuristic: the threshold was not calibrated against leakage on a real model. It is recommended to show the sensitivity of the results to thresholds 0.7 / 0.8 / 0.9 (flag `--threshold`). The sample of 5000 is random, not stratified by task; the overlap differs across tasks.

## 5. Known limitations
- The registry does not contain the PubChem/ZINC corpora or ChEMBL feeds (the dataset itself is from ChEMBL; sets like `herg_karim` are ChEMBL-derived, so the overlap with them is high).
- Standardization: largest fragment + neutralization, no tautomers; differences in tautomers/charges may cause matches to be missed.
- The document-year filter is an approximation (see section 2).

## 6. Trial run on data/processed (58 tasks, 154866 task-molecule pairs; results were not saved to data/processed_clean — ChEMBL was still being downloaded)
| step | molecules removed (sum over tasks) | measurements removed |
|---|---|---|
| exact InChIKey (full) | 12351 | 19108 |
| exact InChIKey14 (additional, stereoisomers) | 1472 | 2656 |
| Tanimoto ECFP4 > 0.8 | 3061 | 4135 |
| doc year > 2022 (max of model cutoffs from YAML) | 129733 | 158887 |

Result: 154866 -> 8249 task-molecule pairs; after cleaning, none of the 58 tasks passes the threshold on replicated molecules (`decontam_loss.csv`, `tasks.csv`). The main source of losses is the year filter: most ChEMBL documents are older than 2023. If restricted to overlap with benchmarks only (`--no-year`) or if a single model's cutoff is used (`--models`, `--cutoff-year`), the losses are much smaller (~10% of molecules). The decision on whether the year filter is needed for all models is left to the plan owner; the cutoffs were chosen conservatively (the preprint date, not an actually verified cutoff).
Time: ~16 min on 4 cores with a loaded machine (standardization + InChI of all molecules, then the bit-count window).
