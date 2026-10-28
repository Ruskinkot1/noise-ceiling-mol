# Experimental design of noise-ceiling-mol

Status: full design (W6). There are no results here: none of the numbers below were obtained on data; numbers in examples
are marked "example". Thresholds called "parameter" were chosen before looking at the data and were not tuned; they can be changed,
but then they must be changed in all tables as well. Anything not verified is marked "not verified".

Readers: chemists and ML specialists. Therefore the key concepts are explained twice.
- For the chemist: pActivity = −log10(concentration, M) (pChEMBL). Noise of 0.5 log units = roughly a 3-fold difference in
  concentration (10^0.5 ≈ 3.2; example, not a result). The noise ceiling shows what prediction accuracy can be
  expected if the experimental quantity itself is reproducible across independent publications only to that precision.
- For ML: the test label is noisy. The error of any model cannot be smaller than the label noise (RMSE_floor), and differences between models
  smaller than this noise are not interpretable (an analogue of test–retest reliability and noise ceiling in other fields; references not verified).

## 0. What we claim and what we do not
Question: what share of the differences between models on molecular property benchmarks lies within the noise of the experimental label,
and do pretrained embeddings outperform ECFP+GBM when measured by fraction of ceiling, on data that are not in existing
benchmarks. Ceiling = reproducibility of the value for a single molecule across independent ChEMBL documents. This is an
estimate of ALL cross-source variability (laboratory, assay conditions, recording errors), not of the "pure" instrument error
(definition — src/ncmol/ceiling.py). Real differences in assay conditions also fall into the "noise": for the task
"predict the value under the conditions of an arbitrary laboratory" this is the correct ceiling, for "predict a property of the molecule" it is conservative.

## 1. Metric: fraction of ceiling (decision of step 0.3)
Primary: `frac_ceiling = (RMSE_mean − RMSE_model) / (RMSE_mean − RMSE_floor)`.
- `RMSE_mean` — the RMSE of the "train mean" prediction on the same test set (scripts/04, column `rmse_mean`);
- `RMSE_floor = σ·sqrt(mean(1/n_docs))` over the test molecules (src/ncmol/ceiling.py); the label of a molecule = the mean over its documents,
  the label noise variance is σ²/n_docs, so the ceiling is computed for a SPECIFIC test set;
- 0 = no better than the mean, 1 = at the noise ceiling, >1 = suspected noise fitting or leakage, <0 = worse than the mean.
Why: a clear zero point for both audiences; independent of the pActivity scale.
Auxiliary (in tables, not in headlines): `frac_rmse = RMSE_floor/RMSE`, `frac_r2 = R²/R²_max`.
Quality metrics: RMSE, MAE, R², Spearman (src/ncmol/metrics.py); RMSE is primary because the ceiling relies on it.
We show Spearman and MAE, but there is no "ceiling" for them in the main table (for MAE there is the approximation σ·sqrt(2/π) under normal noise).

### 1.1 Attenuation correction (noisy test label)
Noise model: y_obs = t + e, Var(e_i) = σ²/n_i, e is uncorrelated with the model error. Then MSE_obs = MSE_true + RMSE_floor².
`metrics.attenuation_corrected` gives:
- `rmse_true = sqrt(max(RMSE² − floor², 0))`, `rmse_mean_true = sqrt(max(RMSE_mean² − floor², 0))`;
- `frac_ceiling_true = (rmse_mean_true − rmse_true) / rmse_mean_true` (in "true" units the ceiling = 0 error);
- `R²_true = 1 − (MSE − floor²)/(Var(y_obs) − floor²)`; label reliability `rel = 1 − floor²/Var(y_obs)`;
  `r_true ≈ r_obs/sqrt(rel)` (for Spearman it is an approximation, derived for Pearson).
Assumptions (not proven, to be checked): homoscedasticity of σ; independence of test noise and model error (violated if the
test documents also contain training molecules of the same series — "document correlation", see §3); σ is estimated on replicated
molecules and transferred to all. If `RMSE < floor`, the correction clips to 0 and sets the flag `below_floor` (noise fitting or leakage).
Headroom `headroom = (RMSE_mean − RMSE_floor)/RMSE_mean`: if it is < 0.2 (parameter `LOW_HEADROOM` in scripts/05), the denominator of the fraction of ceiling is close to 0 and the fraction is unstable —
the task is flagged `low_headroom`, and its result is read with caution. The correction is auxiliary: the headline is `frac_ceiling`; both versions are reported, and their discrepancy is part of the sensitivity analysis.
Uncertainty of σ: the bootstrap bounds from scripts/03 (`sigma_lo`, `sigma_hi`) are recomputed into `frac_ceiling_at_sigma_lo/hi`.

## 2. Tasks
A task = (ChEMBL target × measurement type Ki or IC50), pActivity = pChEMBL, only `standard_relation "="`, assay type B,
confidence_score ≥ 8 (scripts/01); one value per (molecule, document) — the mean within the document (scripts/02).
Ki and IC50 types are separate tasks (see §8.2). The model's dataset includes ALL molecules of the task (including those with a single document);
replicated molecules (≥2 documents) are those used to estimate σ. A task is admitted with ≥100 replicated molecules
(configs/endpoints.yaml); the project stopping criterion is fewer than 15 admitted tasks (ncm-researcher). The target list
in configs/endpoints.yaml is marked "not verified" (IDs written from memory, pref_name to be checked via the API).
External sets (TDC etc.) are handled separately, with a ceiling from other sources (not verified).

## 3. Splits and seeds
- random: random 80/20 by molecules;
- scaffold: Murcko scaffold groups go entirely into train or test (group order shuffled by seed; large scaffolds always go to train);
- temporal: test = the 20% of molecules with the latest year of first publication (`first_year`); deterministic, the seed changes only the model.
- 5 seeds (0–4) for each model; identical (task, split, seed) for all models.
Variance over seeds reflects instability of training and partitioning, NOT uncertainty due to the finite sample of molecules; the latter is given by
the cluster bootstrap over molecules (§6). Both kinds of uncertainty are shown separately.
Risks (checked by the slices of §9 and the artifacts of §10, not eliminated): (a) document correlation — molecules from one paper
(one series) in train and test; a split by documents is not done (not verified, a possible extension); (b) the scaffold split
does not guarantee low similarity (see max Tanimoto test→train); (c) temporal uses the molecule's first year, not the year of the specific measurement.

## 4. Models
All models see the same data, splits, and seeds.
- Baseline: ECFP4 (counts, 2048) + gradient boosting (`ecfp_gbm`, BASELINE in src/ncmol/models.py).
- Other classical: `ecfp_rf`, `ecfp_mlp`, `ecfp_ridge`.
- Pretrained embeddings: frozen + head (GBM or MLP on the embedding, `emb-<name>_gbm`), without fine-tuning — an explicit limitation:
  the conclusion applies to frozen features, not to fine-tuning. The list of embeddings is not fixed.
- GNN (D-MPNN via chemprop): planned in the docstring of models.py, NOT implemented.
Hyperparameter budget (a fairness requirement): the same for ALL models, including ECFP+GBM and the "heads" on top of embeddings.
State of the code: the hyperparameters in models.py are currently fixed and were not tuned, i.e. the requirement is not met — ECFP+GBM
is not tuned "as carefully as the pretrained model". Proposal (for W4, not implemented): random search over N=20 configurations
on an inner validation of train (of the same type as the outer split), the same N for all, best by RMSE; the number of configurations is recorded.
Until this exists, the results of model comparison cannot be considered "fair" in the sense of item 2 of the fairness list.

## 5. Hypotheses (before data) and refutation conditions
Thresholds are parameters, chosen before looking at the data.
- H1 (noise is substantial). On admitted tasks the ceiling is noticeably below ideal: the median R²_max over tasks < 0.9.
  Refuted if the median R²_max ≥ 0.9 (noise is small; then the noise argument does not work, reformulate as a model comparison —
  the ncm-researcher stop criterion) or if the CI of the median lies entirely above 0.9.
- H2 (differences between models are mostly within noise). For most "model vs ECFP+GBM" pairs per (split), the difference in fraction of ceiling
  is not distinguishable from 0: the adjusted Wilcoxon p over tasks ≥ 0.05 AND the CI of the paired bootstrap over molecules contains 0.
  Refuted if for ≥ half of the comparisons p_adj < 0.05 AND |mean Δ fraction of ceiling| ≥ MDE (§6.4).
- H3 (embeddings do not outperform ECFP+GBM in fraction of ceiling). For frozen embeddings, Δ(fraction of ceiling) vs ECFP+GBM is not positive
  under scaffold and temporal splits. Refuted if for at least one embedding under scaffold or temporal p_adj < 0.05 and
  mean Δ > 0 and ≥ MDE. Addendum H3b (exploratory): the advantage of embeddings (if any) decreases from random to scaffold and temporal.
- Trap checks (not hypotheses): fraction of ceiling > 1 or `below_floor` — leakage/noise fitting; discrepancy between the clean and raw
  sets (decontamination, ncm-decontaminate) = an estimate of the impact of leakage.
Note: with a small number of tasks T the Wilcoxon test is unattainable: at T=5 the minimum two-sided p = 0.0625; at T=6 it is 0.031, and after
Holm correction for m comparisons p ≤ 0.05/m is required. Therefore at T<6 the test is not reported (NaN), and the decision rests on the paired bootstrap.

## 6. Statistics (src/ncmol/stats.py)
1. Cluster bootstrap over molecules (the resampling unit is the molecule; all its rows across seeds go together): CIs for RMSE, MAE, R²,
   Spearman and fraction of ceiling; percentile 95% CIs, by default 500–1000 resamples. For the summary over tasks — the mean over tasks with
   equal weights, resamples are independent across tasks; a bootstrap over tasks is shown separately (the set of tasks is also a sample).
2. Paired bootstrap "model − ECFP+GBM": both models on the same resamples of molecules; result — Δ, CI, bootstrap p, an "inside noise" flag (the CI contains 0).
3. Wilcoxon (signed-rank, two-sided) over tasks on Δ fraction of ceiling (and RMSE); the multiplicity correction — PRIMARY is Holm
   (FWER control, no assumptions about dependence; family = all "model×split" comparisons in the table), sensitivity — Benjamini–Hochberg (FDR).
   Both are in paired_vs_baseline.csv (`p_holm`, `p_bh`).
4. MDE (minimum detectable difference) at α=0.05, power 0.8: `MDE = (z_{0.975}+z_{0.8})·SE ≈ 2.8·SE`.
   (a) empirical — from the SE of the paired difference from the bootstrap (in slices.csv: `se`, `mde_80`);
   (b) model-based, a function of noise and sample size (`stats.mde_rmse_gaussian`): for errors e_k = u_k + ε, Var(u)=v, corr(u_a,u_b)=ρ,
       shared label noise Var(ε)=floor²: Var(e_a²−e_b²) = 2v(1−ρ)·(2v(1+ρ)+4·floor²), SE(ΔMSE)=sqrt(Var/n), ΔRMSE ≈ ΔMSE/(2·RMSE_obs).
       Assumptions: Gaussianity, noise independent of errors; ρ is a parameter (estimated from predictions.csv, not set "out of thin air").
       Consequence: MDE grows with label noise and falls as 1/sqrt(n_test). Verified in a simulation in tests/test_stats.py (not on real data).
   In units of fraction of ceiling: division by (RMSE_mean − RMSE_floor) (`stats.mde_frac_units`).
Wording of the conclusion: a difference smaller than the MDE with a CI containing 0 → "inside noise" (we cannot distinguish it from zero), not "the models are equal".

## 7. Metrics on subsets: activity cliffs
Cliff pair: Tanimoto(ECFP4, 2048 bits) ≥ `sim` and |ΔpActivity| ≥ `delta`; default parameters 0.8 and 1.5, NOT tuned, changed in scripts/07.
Two views: (a) cliff molecules — test molecules that have a cliff partner in train (`cliff_mask_vs_train`): ΔRMSE "model − ECFP+GBM" on them;
(b) pairs within the test set: the share of correctly predicted sign of Δ and the RMSE of the predicted difference (`cliff_pair_metrics`; cliff_pairs.csv).
Noise caveat: some "cliffs" are label noise. The expected probability that noise alone gives |Δ| ≥ delta for a pair with n_i, n_j documents, under normal
noise, is `stats.p_noise_exceeds` (approximation); cliffs among replicated molecules are more reliable. There are no results on cliffs yet.

## 8. Sensitivity
### 8.1 Filter thresholds
The ceiling is recomputed on a grid: min_docs ∈ {2, 3} × max |Δ| between documents ∈ {none, 3, 2, 1} (scripts/03, ceiling_sensitivity.csv; in 02c — σ without a filter, with 3 and 2).
The conclusion about the ceiling/fraction of ceiling is considered robust if the model order and the "inside noise" flags do not change when the thresholds change (check via scripts/05 for different ceilings.csv; there is no automation).
Trimming by |Δ| removes unit/recording errors, but also lowers σ, i.e. overstates attainability; therefore we show both versions.
### 8.2 Ki vs IC50
The tasks are treated separately and cannot be merged. Theory: for a competitive inhibitor at equilibrium, Ki and IC50 are related by the Cheng–Prusoff equation
(IC50 = Ki·(1 + [S]/Km) for a substrate; for radioligand binding — with [L]/Kd; the reference to the original was not verified). This implies only that under a
competitive mechanism and known conditions IC50 ≥ Ki, and the shift depends on the conditions of the specific assay; for non-competitive mechanisms and
incomparable conditions there is no universal conversion, and we do NOT convert Ki↔IC50. Empirically (02c, type_compare.csv): the number of molecules in both types, the mean
difference pIC50 − pKi, its SD, correlations, σ by type. If σ(Ki) and σ(IC50) differ noticeably, the ceiling and conclusions are given per type separately.
### 8.3 Years
The relation of |Δ| between documents to the difference in years and to the activity level (02c: spearman_absdelta_vs_dyear, sigma_pairs_dyear_*); a ceiling by stratum (only documents after a threshold year) is an extension of scripts/03, not implemented.
Temporal split: drift of mean activity and range between train and test — in comparison_artifacts.csv (`y_mean_*`, `y_sd_*`, `year_median_*`).

## 9. Sample size
- MDE as a function of n_test and noise (§6.4): a table/curve for the grid n ∈ {50, 100, 200, 500, 1000, 5000}, floor from ceilings.csv, v from the measured RMSE, ρ from predictions.csv.
- A breakdown by task size (terciles of the number of molecules) in scripts/07 (`task_size`).
- Learning curve (fraction of ceiling vs train fraction: 25/50/100%) — requires a `--train-frac` parameter in scripts/04; not implemented.

## 10. Data audit and comparison artifacts
scripts/02c_audit_data.py → results/audit/*.csv: the share of molecules with ≥2 documents; the distribution of |Δ| between documents and between assays of the SAME document
(if the noise within a document is small and between documents is large — the noise is dominated by the inter-laboratory component); pairs with |Δ| ≥ 2–3 with a list of documents
(big_pairs.csv, threshold — parameter `--big-delta`, not a data filter); the shift of a document and an assay relative to the consensus of the others (leave-one-out, flagged at |z|>3 and ≥5 shared molecules);
the share of pairs with the same assay_id (in ChEMBL assays are usually tied to a document; a nonzero share is a check); Ki vs IC50; stereo variants (one InChIKey14 = `mol_id`, but different
full InChIKeys: stereoisomers merged by standardization, and the difference in |Δ| for pairs with different/identical variants).
NOT checked (results/audit/not_checked.csv): shared authors of documents; assay conditions (cell line, organism, substrate, ATP, description — not in the raw columns of scripts/01);
republication of the same data in different documents; tautomers; manual audit of large pairs.
scripts/07 → results/slices/comparison_artifacts.csv: per task×split×seed — train/test size, the mean n_docs of the test, the test RMSE_floor and its ratio to the label SD,
range imbalance (means/SD train vs test, KS statistic, share of test outside the 5–95% of train), the distribution of max Tanimoto test→train (mean, median, q10, q90, share ≥0.8), the share of cliff molecules,
median year train vs test. These numbers must be published next to the model tables: the difference between splits is partly explained by them.
Slices (scripts/07 → slices.csv): the difference "model − ECFP+GBM" (ΔRMSE, <0 — the model is better; and ΔRMSE/σ) with a cluster CI and the label "inside noise"/"model better"/"model worse"
by: target, split, molecule noise (n_docs 1/2/3+), Tanimoto bins to train, cliff/non-cliff, activity terciles, task size. A slice cell includes only tasks that have ≥30 molecules in it (`--min-mols`), so the task composition changes from cell to cell; it is given in `n_tasks`. Many slices = many comparisons; the slices are exploratory, no corrections are applied to them —
this is stated in the paper, and findings from slices are not formulated as confirmed.

## 11. Fairness of the comparison (user requirement; mandatory in the Methods/Limitations of the paper)
1. The same data, splits, and seeds for all models. (met in scripts/04 by construction)
2. The same hyperparameter tuning budget for ECFP+GBM/RF/MLP, the GNN, and the "heads" on top of embeddings. (NOT met: see §4)
3. Foundation models: only frozen + head, without fine-tuning; this is an explicit limitation.
4. The test label is noisy: fraction of ceiling + attenuation correction (§1.1).
5. Risk against small models: the PubChem/ZINC corpora may contain our molecules, cutoff dates may be "not verified"; decontamination is mandatory,
   the leakage estimate = clean vs raw.
6. Paired comparison across tasks with CIs and a multiple-comparison correction (Holm; BH as sensitivity); differences within noise are to be called that.

## 12. Data contract and run order
Order: 01 → 02 → decontamination → 02c (audit) → 03 (ceiling) → 04 (benchmark) → 05 (fraction of ceiling) → 07 (slices, artifacts).
Input of 05: results/benchmark.csv (task, split, seed, model, n_train, n_test, test_mean_inv_ndocs, rmse, mae, r2, spearman, rmse_mean) and results/ceilings.csv
(task, sigma, var_y, sigma_lo, sigma_hi). For CIs over molecules and slices, results/predictions.csv (or .parquet) is needed: mandatory
`task, split, seed, model, mol_id, pred`; `y`, `n_docs`, `y_train_mean`, when absent, are reconstructed from data/processed/<task>.molecules.csv and by recomputing the split
(ncmol.splits; requires the same row order of molecules.csv as in scripts/04). One row = test molecule × model × seed.
Each result carries: the data version (ChEMBL release, date), the code commit, seeds, the split type (recorded by whoever runs it).

## 13. Limitations (brief)
Homoscedasticity and transfer of σ from replicated molecules to all; noise includes real differences in conditions; train–test document correlation; fixed
non-tuned models at the present time; cliff and filter thresholds are parameters; slices without multiplicity correction; the number of admitted tasks may be small
(then Wilcoxon is unavailable); results on real data are absent at the time of writing.
