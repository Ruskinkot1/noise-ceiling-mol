# noise-ceiling-mol

**The noise ceiling: what molecular property benchmarks actually measure, and whether pretrained embeddings beat ECFP under fair evaluation relative to the measurement limit.**
Target track: NeurIPS Datasets & Benchmarks. Status: **early prototype** — the code for steps 3–5 has been verified on synthetic data, step 1 has been rewritten to use the REST API (parsing test on a mock), and steps 1–2 have **not yet been run** on the full set of targets.

## Hypothesis
1. Inter-laboratory error in ChEMBL is comparable to model error → part of the differences between models lies within label noise.
2. We estimate the ceiling from repeated measurements of the same molecules reported in **different documents**.
3. We recompute model results as a **fraction of the ceiling** and check whether the advantage of pretrained embeddings over ECFP+GBM holds under random / scaffold / temporal splits.

Novelty: partly taken — see [docs/NOVELTY.md](docs/NOVELTY.md) (the closest work is "Are we fitting data or noise?").

## Ceiling method (src/ncmol/ceiling.py)
Model `y_ij = t_i + e_ij`, where `e_ij` is between-document noise. The noise variance is estimated as the pooled within-molecule variance over molecules with ≥2 documents (values within a document are averaged so that the replicates are strictly cross-source):

- `RMSE_floor = sqrt(s² · mean(1/n_docs))` (label = mean over documents),
- `R²_max = 1 − s²·mean(1/n_docs) / Var(y)`,
- `fraction of ceiling`: `frac_rmse = RMSE_floor / RMSE`, `frac_r2 = R² / R²_max`. A value >1 indicates suspected noise fitting or leakage.
- Bootstrap over molecules; sensitivity to `min_docs` and to trimming of pairs by `|Δ|` (unit errors).

**Limitations (to be stated in the paper):** this is the *total* cross-source variability, including systematic assay shifts, rather than "pure" instrument error; homoscedasticity is assumed; replicated molecules may be atypical of the dataset.

## Structure
```
configs/endpoints.yaml   candidate targets (chembl_id from memory — NOT VERIFIED), thresholds
src/ncmol/               chem (standardization, ECFP), ceiling, splits, models, metrics
scripts/01_extract_chembl.py   export from the ChEMBL REST API (the SQLite dump is not used)
scripts/02_standardize.py      RDKit standardization, one y per (molecule, document), tasks.csv
scripts/03_estimate_ceiling.py ceiling + bootstrap + sensitivity
scripts/04_run_benchmark.py    ECFP+GBM/RF/MLP × 3 splits × 5 seeds
scripts/05_fraction_of_ceiling.py  fraction of ceiling, leaderboard, paired comparison with ECFP+GBM
tests/                   synthetic tests (σ recovery, splits)
docs/NOVELTY.md          stage 0
```

## Usage
```bash
pip install -e .            # + pytest
pytest                      # synthetic tests
python scripts/01_extract_chembl.py --targets CHEMBL240     # via the API; without --targets — candidates from configs/endpoints.yaml
python scripts/02_standardize.py        # prints how many tasks passed the threshold (need >= 15)
python scripts/03_estimate_ceiling.py
python scripts/04_run_benchmark.py
python scripts/05_fraction_of_ceiling.py
```
A task = (target, measurement type); IC50 and Ki are treated separately.

## Embeddings and D-MPNN 
- Embeddings: the interface is ready — `data/embeddings/<name>.npz` (`mol_ids`, `X`), models `emb-<name>_gbm` / `_mlp` / `_ridge`. There is no inference script; the choice of 3–4 open models has **not been made**.
- D-MPNN: via chemprop.

## Plan and stopping criteria
Wk 1 novelty (done, partly taken) · 2–3 data (stop: <15 tasks → narrow the scope) · 4–5 ceiling (stop: ceiling ≈ ideal → reformulate) · 6–7 models · 8–9 analysis · 10 recommendations (MDE, protocol) · 11–12 paper, code, leaderboard.

## Verified / not verified
- Verified: `pytest` (3 tests); steps 3→5 on synthetic data (σ recovered as 0.42 for a true value of 0.4).
- Not verified: steps 1–2 on real ChEMBL; the target list; all references in NOVELTY.md were checked only against search excerpts.

## Project skills

Claude skills for this project live in `.claude/skills/` (`ncm-researcher`, `ncm-novelty-check`, `ncm-collect-replicates`, `ncm-decontaminate`, `ncm-benchmark-models`, `ncm-ceiling-metric`). Sprint notes: `docs/sprint-40min.md`. 


## Running on Colab
The notebook [notebooks/run_on_colab.ipynb](notebooks/run_on_colab.ipynb) runs the whole pipeline (data → cleaning → ceiling → models → fraction of ceiling).
Open: https://colab.research.google.com/github/Ruskinkot1/noise-ceiling-mol/blob/cloud-data/notebooks/run_on_colab.ipynb
**Not verified on Colab.** Everything runs on CPU (no GPU is used); the full pilot takes hours. If the repository is private, the clone cell needs a GitHub token.

## Running on Mac (locally)
Not verified on macOS (developed on Linux). Requires Python 3.10–3.13 and about 3 GB of free disk space (data ~0.5 GB, HF model cache ~0.6 GB).
```bash
git clone -b cloud-data https://github.com/Ruskinkot1/noise-ceiling-mol.git && cd noise-ceiling-mol
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
export OMP_NUM_THREADS=1            # otherwise sklearn GBM can be hundreds of times slower with parallel processes
python -m pytest -q
python scripts/01_extract_chembl.py && python scripts/02_standardize.py
python scripts/build_registry.py && python scripts/02b_decontaminate.py --no-year --n-jobs 4
python scripts/make_pilot.py && python scripts/06_embed_foundation.py --processed data/pilot_raw --out-dir data/pilot_emb --models chemberta-77m-mlm molformer-xl-10pct
scripts/run_pilot.sh                # can be interrupted and restarted: completed jobs are skipped
```
The torch device is selected with the `--device {cpu,mps,cuda,auto}` argument (default `cpu`) of `scripts/06_embed_foundation.py` and `scripts/04_run_benchmark.py`,
and for the pilot with the `DEVICE` variable: `DEVICE=mps NJOBS=1 scripts/run_pilot.sh`. On an Apple Silicon Mac: `--device mps` (if an operation is unsupported, set `PYTORCH_ENABLE_MPS_FALLBACK=1`).
**MPS has not been tested** (this machine does not have it; only cpu and auto→cpu were tested). Embeddings and GNNs are accelerated; sklearn models (GBM/RF/MLP) always run on CPU.
If `pip install rdkit` cannot find wheels for your Python version, use Python 3.11–3.12.

## One command
```bash
git clone -b cloud-data https://github.com/Ruskinkot1/noise-ceiling-mol.git && cd noise-ceiling-mol
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .
DEVICE=cpu bash scripts/run_all.sh          # Mac Apple Silicon: DEVICE=mps NJOBS=1 bash scripts/run_all.sh
```
`scripts/run_all.sh` runs tests → ChEMBL export → cleaning → pilot → ceiling → models → fraction of ceiling → breakdowns. Completed stages are skipped (`FORCE=1` recomputes them).
I have not run the whole of `run_all.sh` in a single pass: the stages were verified separately.
