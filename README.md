# noise-ceiling-mol

How much of the difference between molecular property models is just label noise?

This project estimates a **noise ceiling** for each ChEMBL endpoint from the same molecules measured in different publications, and re-scores models as a **fraction of that ceiling**. It uses this to re-test whether pretrained molecular embeddings really beat plain ECFP fingerprints.

Preprint: <https://doi.org/10.26434/chemrxiv.15011172/v1>

## The idea

A molecular ML benchmark ranks models by how close they get to measured labels. But the same compound measured in two labs often gets different values. If two models differ by less than that disagreement, the ranking says little.

- **Replicates.** In ChEMBL a *document* is one source publication. A molecule reported in several documents is a replicate, and the spread of its values across documents estimates the noise `s²`.
- **Ceiling.** `RMSE_floor = sqrt(s² · mean(1/n_docs))` and `R²_max = 1 − s²·mean(1/n_docs) / Var(y)`: the best any model can do on those labels.
- **Fraction of ceiling.** `f = (RMSE_mean − RMSE_model) / (RMSE_mean − RMSE_floor)`. `f = 0` is no better than predicting the mean, `f = 1` is the noise floor, `f > 1` suggests noise fitting or leakage.
- **Comparison.** Paired against an ECFP + gradient boosting baseline under random, scaffold and temporal splits, with confidence intervals and a minimum detectable effect, so differences inside the noise are labelled as such.

## The benchmark

The benchmark is a set of **17 potency-prediction tasks** built from ChEMBL_37. A task is a pair (protein target, measurement type), and IC50 and Ki are never pooled. For each task:

- **Data.** Human protein targets, pChEMBL values (−log10 of the molar potency), one value per (molecule, document), standardized with RDKit. Molecules found in MoleculeNet or TDC sets are removed (exact matches and ECFP4 Tanimoto > 0.8), so the tasks do not overlap with the usual benchmarks. A target is kept only if it has enough molecules measured in at least two documents (threshold in `configs/endpoints.yaml`).
- **Splits.** Random, scaffold (Bemis–Murcko) and temporal (by document year).
- **Models.** ECFP4 counts + GBM and random forest, an in-house D-MPNN (small tasks only), and frozen ChemBERTa-77M and MoLFormer-XL embeddings with a GBM head. The full run uses a reduced protocol: 3 seeds, 2 tuning configurations per model.
- **Metrics.** RMSE, R², and the fraction of the noise ceiling, with molecule-level confidence intervals and a paired comparison against ECFP+GBM.

Targets, as listed in the Croissant metadata (`croissant/noise_ceiling_mol.json`):

| Group | Target (ChEMBL ID) | Measurement |
|---|---|---|
| Kinases | EGFR (CHEMBL203), VEGFR2 (CHEMBL279), JAK2 (CHEMBL2971), PI3K-alpha (CHEMBL4005), B-Raf (CHEMBL5145) | IC50 |
| GPCRs | 5-HT1A (CHEMBL214), D2 (CHEMBL217), D3 (CHEMBL234), adenosine A2a (CHEMBL251), mu-opioid (CHEMBL233) | Ki |
| GPCRs | kappa-opioid (CHEMBL237) | IC50 and Ki |
| Nuclear receptor | estrogen receptor (CHEMBL206) | IC50 |
| Enzymes | thrombin (CHEMBL204), carbonic anhydrase 2 (CHEMBL205) | Ki |
| Enzymes | HDAC1 (CHEMBL325), BACE1 (CHEMBL4822) | IC50 |

That is 16 targets and 17 tasks. The candidate list in `configs/endpoints.yaml` has 29 targets; the remaining ones are not part of the released set. Per-task molecule counts are not listed here because the data is not tracked in git; the pipeline prints them (`scripts/02_standardize.py`, `scripts/03_estimate_ceiling.py`).

## Quick start

Python 3.10–3.13 (3.11–3.12 if RDKit wheels are missing for your version).

```bash
git clone https://github.com/Ruskinkot1/noise-ceiling-mol.git && cd noise-ceiling-mol
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
export OMP_NUM_THREADS=1      # otherwise sklearn GBM can be hundreds of times slower with parallel processes
python -m pytest -q           # 42 tests on synthetic data, no network
```
