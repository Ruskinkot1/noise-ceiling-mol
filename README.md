# noise-ceiling-mol

**The noise ceiling: how much of the difference between molecular-property models lies inside label noise, and do pretrained embeddings still beat ECFP once scores are expressed relative to the measurement limit?**

Preprint: <https://doi.org/10.26434/chemrxiv.15011172/v1>

Molecular property benchmarks are scored against experimental labels, yet the same compound measured in different laboratories routinely disagrees. This project estimates a **noise ceiling per endpoint from replicate measurements of the same molecules in independent ChEMBL documents** (instead of assuming an error size), converts model scores into a **fraction of the ceiling**, and re-tests the claimed advantage of pretrained embeddings over ECFP baselines under random, scaffold and temporal splits.

## Ideas

- **Ceiling from replicates.** Model `y_ij = t_i + e_ij`: molecule `i`, document `j`, `e_ij` is between-document noise. Values within one document are averaged first, so replicates are strictly cross-source. Noise variance `s²` is the pooled within-molecule variance over molecules with ≥2 documents.
  - `RMSE_floor = sqrt(s² · mean(1/n_docs))`, `R²_max = 1 − s²·mean(1/n_docs) / Var(y)`, `MAE_floor = RMSE_floor·sqrt(2/π)` (Gaussian). For evaluation, `mean(1/n_docs)` is taken over the test molecules.
  - Cluster bootstrap over molecules; sensitivity to the minimum number of documents (2, 3) and to trimming pairs with large `|Δ|` (guards against unit errors).
- **Fraction of ceiling.** `f = (RMSE_mean − RMSE_model) / (RMSE_mean − RMSE_floor)`, where `RMSE_mean` is the RMSE of predicting the training mean. `f = 0`: no better than the mean; `f = 1`: at the noise floor; `f > 1`: suspected noise fitting or leakage; `f < 0`: worse than the mean. Auxiliary: `RMSE_floor/RMSE` and `R²/R²_max`, plus an attenuation-corrected version (noisy test labels).
- **Honest comparison.** Same data, splits and seeds for every model; same hyperparameter-search budget for ECFP baselines, GNN and embedding heads; foundation models are frozen (embedding + trained head, no fine-tuning).
- **Reporting protocol.** Paired comparison against the ECFP+GBM baseline with bootstrap CIs and a minimum detectable effect (MDE), so that differences inside the noise are labelled as such.

**Limitations.** The ceiling is *total* cross-source variability, including systematic assay shifts, not pure instrument error. A single homoscedastic `s²` per task is assumed. Replicated molecules (reference compounds, popular series) may be atypical. Pretraining corpora and cutoff dates of some foundation models are unknown, so decontamination reduces but cannot exclude overlap. Scope is potency endpoints (Ki, IC50) from ChEMBL only.

## What is implemented

| Stage | Code | What it does |
|---|---|---|
| 01 Extraction | `scripts/01_extract_chembl.py` | ChEMBL REST API export (Ki/IC50, pChEMBL, confidence filter). Candidate targets in `configs/endpoints.yaml` (29 candidates, IDs verified, one rejected) |
| 02 Standardization | `scripts/02_standardize.py`, `src/ncmol/chem.py` | RDKit standardization, one value per (molecule, document), `tasks.csv`. A task is a (target, measurement type) pair; Ki and IC50 are never pooled |
| Registry | `scripts/build_registry.py` | Registry of molecules from 11 MoleculeNet and 40 TDC sets (Polaris tables are behind a login and are not included) |
| 02b Decontamination | `scripts/02b_decontaminate.py`, `src/ncmol/decontam.py`, `configs/model_cutoffs.yaml` | Removes exact matches (InChIKey, InChIKey14) and close analogs (ECFP4 Tanimoto > 0.8), optional document-year filter against model pretraining cutoffs. Raw and clean copies are kept so leakage can be measured as the raw − clean difference |
| 02c Audit | `scripts/02c_audit_data.py` | Replicate share, outlier pairs, Ki vs IC50, stereoisomer check, same-assay pairs. Results in `docs/AUDIT.md` |
| 03 Ceiling | `scripts/03_estimate_ceiling.py`, `src/ncmol/ceiling.py` | `σ`, floors, `R²_max` with bootstrap and sensitivity grid |
| 04 Benchmark | `scripts/04_run_benchmark.py`, `src/ncmol/models.py`, `src/ncmol/splits.py` | ECFP4 (counts, 2048) + GBM/RF/MLP/Ridge; random / scaffold (Bemis–Murcko) / temporal splits; seeds; equal tuning budget; per-molecule predictions saved |
| D-MPNN | `src/ncmol/gnn.py` | Our pure-PyTorch D-MPNN following Yang et al. 2019 (not the Chemprop package, no direct comparison yet) |
| 06 Embeddings | `scripts/06_embed_foundation.py` | Frozen embeddings (mean-pooled) from ChemBERTa-77M MLM/MTR, ChemBERTa-5M, MoLFormer-XL-10pct, roberta-zinc-480m. Cached, resumable by chunks, `--device {cpu,mps,cuda,auto}`. Details in `docs/models.md` |
| 05 Fraction of ceiling | `scripts/05_fraction_of_ceiling.py`, `src/ncmol/metrics.py`, `src/ncmol/stats.py` | Fraction of ceiling with molecule-level CIs, leaderboard, paired comparison with ECFP+GBM, Wilcoxon with Holm and BH correction, empirical and model-based MDE |
| 07 Slices | `scripts/07_slices.py` | Difference to ECFP+GBM by target, split, molecule noise, Tanimoto-to-train bin, activity cliffs, activity range; comparison artifacts (range imbalance, year) |
| Pilot / full runs | `scripts/make_pilot.py`, `run_pilot.sh`, `merge_pilot.py`, `run_all.sh`, `run_full.sh`, `run_full_gnn.sh`, `run_rest.sh` | Resumable runners (job = task × split; finished jobs are skipped) |
| Metadata | `scripts/make_croissant.py` | Draft Croissant file (core + RAI) for the released tasks |
| Paper | `paper/` | LaTeX draft: ceiling, data, benchmark, recommendations, limitations, datasheet, related work |
| Tests | `tests/` | `pytest -q`: 42 tests (ceiling recovery on synthetic data, splits, statistics, decontamination, GNN and embedding interfaces, ChEMBL parsing) |

Data version: ChEMBL_37 (release 2026-05-01). Data, embeddings and results are not tracked in git (`data/`, `results/`).

## Results so far

The only results committed to the repository are the **3-task pilot** in [docs/PILOT_RESULTS.md](docs/PILOT_RESULTS.md) (CHEMBL237 IC50, CHEMBL206 IC50, CHEMBL204 Ki). The paper draft in `paper/` still contains placeholders for the full-benchmark numbers.

- Noise ceilings: σ = 0.69 / 0.53 / 0.31, i.e. `R²_max` = 0.63 / 0.85 / 0.96.
- On random and scaffold splits, ECFP+RF and ECFP+GBM scored higher than frozen ChemBERTa-77M and MoLFormer-XL heads (about 0.09–0.11 of the ceiling; molecule-level CIs exclude 0). ECFP+RF was slightly better than ECFP+GBM.
- On the temporal split the order reverses, but every model is far from the ceiling (fraction 0.2–0.3) and the result is unstable across the 3 tasks.
- At task level no difference is distinguishable from noise (n = 3 gives no power for a Wilcoxon test).
- Raw vs cleaned data: no leakage detected on these tasks (changes within ±0.01–0.04).

These numbers do not generalize to other tasks, to fine-tuned models or to other embeddings. The full benchmark on the cleaned set runs under a reduced protocol (3 seeds, 2 tuning configurations per model, no ECFP+MLP); see `scripts/run_full.sh`.

## Setup and usage

Python 3.10–3.13 (3.11–3.12 if RDKit wheels are missing for your version); about 3 GB of disk (data ~0.5 GB, HF model cache ~0.6 GB).

```bash
git clone -b cloud-data https://github.com/Ruskinkot1/noise-ceiling-mol.git && cd noise-ceiling-mol
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
export OMP_NUM_THREADS=1            # otherwise sklearn GBM can be hundreds of times slower with parallel processes
python -m pytest -q
```

Step by step:

```bash
python scripts/01_extract_chembl.py && python scripts/02_standardize.py
python scripts/build_registry.py && python scripts/02b_decontaminate.py --no-year --n-jobs 4
python scripts/02c_audit_data.py
python scripts/make_pilot.py
python scripts/06_embed_foundation.py --processed data/pilot_raw --out-dir data/pilot_emb --models chemberta-77m-mlm molformer-xl-10pct
scripts/run_pilot.sh                # resumable
python scripts/merge_pilot.py && python scripts/05_fraction_of_ceiling.py --results results/pilot_clean --processed data/pilot_clean
python scripts/07_slices.py --results results/pilot_clean --processed data/pilot_clean
```

One command: `DEVICE=cpu bash scripts/run_all.sh` (Apple Silicon: `DEVICE=mps NJOBS=1 bash scripts/run_all.sh`). Completed stages are skipped; `FORCE=1` recomputes. The stages were verified separately, `run_all.sh` has not been run end-to-end in one pass.

The torch device is chosen with `--device {cpu,mps,cuda,auto}` in `06_embed_foundation.py` and `04_run_benchmark.py`, or with `DEVICE` for the runners. MPS has not been tested (on an Apple Silicon Mac set `PYTORCH_ENABLE_MPS_FALLBACK=1` if an operation is unsupported). Only embeddings and the GNN use the GPU; sklearn models always run on CPU.

Colab: [notebooks/run_on_colab.ipynb](https://colab.research.google.com/github/Ruskinkot1/noise-ceiling-mol/blob/cloud-data/notebooks/run_on_colab.ipynb) runs the pipeline on CPU (not verified on Colab; the full pilot takes hours).

## Documentation

| File | Content |
|---|---|
| `docs/experiment_design.md` | Full design: metric, tasks, splits, models, hypotheses with refutation criteria, statistics, sensitivity, slices, fairness requirements |
| `docs/decontamination.md` | Registry, cutoff rules, cleaning steps, threshold pilot |
| `docs/models.md` | Model table with revisions, licenses, sizes and what was verified |
| `docs/AUDIT.md` | Data validity audit (stereoisomers, replicate share, outliers) |
| `docs/PILOT_RESULTS.md` | 3-task pilot results |
| `docs/NOVELTY.md`, `docs/literature.md` | Novelty check and literature table |
| `docs/VENUE.md` | NeurIPS Evaluations & Datasets requirements (the 2026 deadlines have passed) |

## Novelty

Partly taken. The closest works are "Are we fitting data or noise?" (Crusius et al., 2025; noise of an assumed size is added to labels) and Landrum & Riniker (2024; they measure ChEMBL inter-assay noise but do not turn it into an accuracy limit). The contribution here is the combination of an empirical cross-document ceiling, a ceiling-relative metric, and a split-resolved test of pretrained embeddings against ECFP. Related work was checked from abstracts and search excerpts; full texts still need to be read before submission.

## Status

Research code. Verified: unit tests (42 passing), the ceiling recovers σ on synthetic data (0.42 for a true 0.4), a 3-task pilot. Not verified: the licenses of some ChemBERTa checkpoints, pretraining cutoff dates of all foundation models, equivalence of our D-MPNN to Chemprop, the full run of `run_all.sh`, MPS and Colab execution.
