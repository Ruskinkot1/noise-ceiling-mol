---
name: ncm-researcher
description: Act as a careful ML-for-chemistry researcher on noise-ceiling-mol: own ChEMBL datasets not seen in public benchmarks, foundation chemical models vs small models (ECFP QSAR, simple GNN), scored against the label-noise ceiling. Use for any research, analysis, data or writing task in this project.
---

# Researcher skill for noise-ceiling-mol

## Task (current formulation)
Build our own datasets from ChEMBL, including per-document experimental examples absent from public benchmarks, and test how foundation chemical models reproduce them versus small models: QSAR on fingerprints (ECFP + GBM/RF/MLP) and the simplest graph neural networks (GCN / D-MPNN). The label-noise ceiling (replicate measurements of the same molecule from different documents) is the yardstick, not the only contribution. Target: NeurIPS Datasets & Benchmarks.

## Hypotheses (state them before computing; thresholds are fixed after the pilot, before the main run)
- H1 headroom: for most tasks the best small model already reaches a large fraction of the ceiling, so the room left for foundation models is small.
- H2 resolution: most foundation-vs-ECFP differences have paired-bootstrap CIs that include 0 once label noise is accounted for.
- H3 transfer: the foundation-model advantage over ECFP shrinks from random to scaffold to temporal split, and further after decontamination (post-cutoff, not in benchmarks or pretraining corpora).
- H4 sanity: fraction of ceiling > 1 signals fitting noise or leakage.

## Formal setup
Task k = (target, measurement type). Molecule i, documents d in D_i, y_id = t_i + e_id, e_id ~ (0, s_k^2), mean label over n_i documents.
- s_k^2 = pooled within-molecule variance over molecules with n_i >= 2 (values averaged inside a document).
- RMSE_floor = sqrt(s_k^2 * mean(1/n_i)); R2_max = 1 - s_k^2 * mean(1/n_i) / Var(y).
- Fraction of ceiling: RMSE_floor / RMSE, or R2 / R2_max. Bootstrap over molecules. Code: `src/ncmol/ceiling.py`.
- Leakage L = frac_before_decontamination - frac_after, per model.

## Verified literature (from search excerpts, full texts not read; re-check before citing numbers)
- Landrum & Riniker 2024, JCIM: combining IC50/Ki from different sources is noisy; minimal curation: ~65% of points differ > 0.3 log units, 27% > 1 log unit, Kendall tau 0.51; stricter metadata matching: 48% / 13%, tau 0.71.
- Kramer et al. 2012, J Med Chem: reproducibility across labs bounds attainable accuracy; many ChEMBL duplicates are not independent measurements (secondary reporting says > 90%) -> use document/assay as the independence unit.
- Crusius, Cipcigan, Biggin, "Are we fitting data or noise?" (Faraday Discuss. 2025): performance bounds from assumed/published error; 4 of 9 datasets at or beyond the bound. Closest competitor; cite and contrast (empirical replicate noise vs assumed noise).
- Praski et al. 2025 (arXiv 2508.06199): 25 pretrained embedding models x 25 datasets; nearly all not better than ECFP.
- Schuh et al. audit (via Chemistry World): 51 benchmark configurations, leakage and contradictory labels; rankings can flip.
- MoleculeACE (van Tilborg 2022): 30 ChEMBL targets, activity cliffs; descriptor methods beat deep learning.
- Polaris (Ash et al. 2024): guidelines for curation, splits and statistical tests; no noise-ceiling metric seen (not verified).
- TDC ADMET group: 22 datasets, scaffold split, MAE/Spearman/AUROC; metrics not tied to a measurement ceiling (not verified in primary source).
- MoleculeNet critiques (WelQrate, 2026 survey): invalid/undefined stereo structures, heterogeneous IC50 sources (BACE "71% undefined stereocenter" is secondary, check).

## Demonstrated by RDKit in this project
- Three spellings of aspirin (acid, reordered SMILES, sodium salt) collapse to one InChIKey after `ncmol.chem.standardize` (BSYNRYMUTXBXSQ).
- Gefitinib, erlotinib, lapatinib: ECFP4 Tanimoto 0.41 / 0.32 / 0.28 and three different Murcko scaffolds, so a scaffold split can put close analogs on both sides; scaffold split does not remove near-analog leakage.
- Implied Gaussian sigma from Landrum & Riniker fractions: 0.47 (P>0.3 = 0.65) vs 0.64 (P>1 = 0.27) -> heavy tails; justify robust estimates and the |delta| trimming sensitivity.
- Synthetic check: true sigma 0.5 recovered as 0.51 (400 molecules); this is a code check, not a result.

## Working rules
- Never invent citations, numbers, IDs or datasets; open the source or mark "не проверено".
- Every result carries ChEMBL release, download date, code commit, seeds, split, decontamination setting.
- Tune ECFP+GBM as carefully as any foundation model; foundation models by frozen embeddings + linear/GBM head first.
- Independence: unit is the document; collapse repeats of the same assay_id; filter potential_duplicate and data_validity_comment (not yet implemented for assay-level collapse).
- Decontamination: registry of InChIKeys (MoleculeNet, TDC, Polaris, pretraining corpora), Tanimoto ECFP4 > 0.8 (pilot-tuned), documents after each model's cutoff (undocumented cutoff = "не проверено"). Keep an uncleaned copy to measure leakage.
- Report CIs (5 seeds, paired bootstrap); say when a difference is inside the noise.
- Local and open only; ask before any install or download over 500 MB (laptop disk is nearly full); clean up afterwards.

## Environment facts
- From the local sandbox the ChEMBL REST API answered HTTP 500 and the ChEMBL FTP was unreachable; Hugging Face worked. Do not simulate data: report and stop, or run in the cloud session.
- Cached locally: ChemBERTa-5M/10M/77M (MLM, MTR), MoLFormer-XL-both-10pct. torch 2.4, transformers 4.51, torch_geometric 2.6 are installed.
- Repo pipeline: `scripts/01_extract_chembl.py` ... `05_fraction_of_ceiling.py`; `scripts/collect_replicates.py` overlaps with 01 and should be merged.

## Workflow (sibling skills)
1. `ncm-novelty-check` 2. `ncm-collect-replicates` 3. `ncm-decontaminate` 4. `ncm-benchmark-models` 5. `ncm-ceiling-metric`

## Stop criteria
- Direct duplicate in novelty check: narrow or change topic.
- Fewer than 15 tasks with >= 100 replicated molecules after decontamination: add PubChem BioAssay / BindingDB or narrow the claim; do not loosen filters silently.
- Ceiling near perfect: reframe as a model comparison.
- If an outcome contradicts a hypothesis, report it as a result.

## Status (update when it changes)
Verified: pytest (3 tests), steps 3-5 on synthetic data, RDKit demonstrations above. Not verified: steps 1-2 on real ChEMBL, target ID list, GNN and foundation-model pipelines, all literature numbers beyond search excerpts.
