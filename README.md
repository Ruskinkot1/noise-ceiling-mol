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

Compute a ceiling on simulated replicates (true noise σ = 0.4):

```python
import numpy as np, pandas as pd
from ncmol.ceiling import estimate_ceiling

rng = np.random.default_rng(0)
rows = []
for i in range(200):
    truth = rng.normal(6, 1)
    for d in range(rng.integers(2, 4)):          # 2 or 3 documents per molecule
        rows.append((f"m{i}", f"doc{d}", truth + rng.normal(0, 0.4)))
df = pd.DataFrame(rows, columns=["mol_id", "doc_id", "y"])

print(estimate_ceiling(df, min_docs=2, max_abs_pair_diff=3.0))
# sigma ≈ 0.39, rmse_floor ≈ 0.25, r2_max ≈ 0.94
```

For your own data, `df` needs three columns: `mol_id`, `doc_id`, and `y` (a log-scale value such as pChEMBL).

## Repository map

```
src/ncmol/   library: ceiling, chem, splits, models, gnn, metrics, stats, decontam
scripts/     numbered pipeline steps (01 extract → 02 standardize → 02b decontaminate → 03 ceiling → 06 embed → 04 benchmark → 05 fraction of ceiling → 07 slices) and runners
configs/     target list and model pretraining cutoffs
tests/       42 tests
docs/        design, decontamination, models, audit, pilot results, literature
croissant/   machine-readable dataset metadata (draft)
notebooks/   run_on_colab.ipynb
data/        not tracked in git; see data/README.md
```

Run the whole pipeline on the 3-task pilot with `DEVICE=cpu bash scripts/run_all.sh` (Apple Silicon: `DEVICE=mps NJOBS=1 ...`). It takes hours on CPU and every stage is resumable. Needs about 3 GB of disk. The stages were verified separately; `run_all.sh` has not been run end-to-end in one pass. Colab: [notebooks/run_on_colab.ipynb](https://colab.research.google.com/github/Ruskinkot1/noise-ceiling-mol/blob/main/notebooks/run_on_colab.ipynb) (not verified on Colab).

What each step does and how to run it step by step is in [docs/experiment_design.md](docs/experiment_design.md).

## Results so far

The results committed here are a **3-task pilot** (CHEMBL237 IC50, CHEMBL206 IC50, CHEMBL204 Ki): [docs/PILOT_RESULTS.md](docs/PILOT_RESULTS.md).

- Noise ceilings: σ = 0.69 / 0.53 / 0.31, i.e. `R²_max` = 0.63 / 0.85 / 0.96.
- Random and scaffold splits: ECFP+RF and ECFP+GBM scored about 0.09–0.11 of the ceiling higher than frozen ChemBERTa-77M and MoLFormer-XL heads.
- Temporal split: the order reverses, but all models are far from the ceiling (0.2–0.3) and the result is unstable across tasks.
- At task level no difference is distinguishable from noise: 3 tasks give no power for a Wilcoxon test.

These numbers do not generalize to other tasks, fine-tuned models or other embeddings.

## Further reading

| File | Content |
|---|---|
| `docs/experiment_design.md` | Metric, tasks, splits, models, hypotheses, statistics, slices |
| `docs/decontamination.md` | Removing molecules already present in public benchmarks |
| `docs/models.md` | Embedding models, revisions, licenses |
| `docs/AUDIT.md` | Data validity audit |
| `docs/NOVELTY.md`, `docs/literature.md` | Related work |

## Limitations

The ceiling is *total* cross-source variability, including systematic assay shifts, not pure instrument error. It assumes one homoscedastic `s²` per task, replicated molecules may be atypical, and pretraining cutoffs of some foundation models are unknown. Foundation models are frozen (embedding + head); nothing is fine-tuned. Scope is potency endpoints (Ki, IC50) from ChEMBL.

This is research code. Not verified: a full `run_all.sh` run, MPS and Colab execution, equivalence of the in-house D-MPNN to Chemprop.
