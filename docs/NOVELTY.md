# Stage 0. Novelty check (2022–2026)

Method: 5 search queries; only abstracts/excerpts of the search results were read, full texts were **not** read.
All numbers below come from these excerpts; anything unconfirmed is marked "not verified". The search is incomplete:
for example, work on noise in TDC/Polaris, label noise in ADMET, and pharma-internal reports are not covered.

| # | Work | Year | Gist | Difference from the project | Link |
|---|---|---|---|---|---|
| 1 | Landrum, Riniker — *Combining IC50 or Ki Values from Different Sources Is a Source of Significant Noise* (JCIM 64(5):1560–1567, DOI 10.1021/acs.jcim.4c00049) | 2024 | Inter-assay noise in ChEMBL for a single molecule–target pair: with minimal curation ~65% of points differ by >0.3 log units, 27% by >1 log unit, Kendall τ=0.51; with matching metadata τ=0.71; noise is similar for Ki | They measure the noise but do **not** translate it into an accuracy limit for models and do not recompute model metrics | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10934815/) |
| 2 | *Are we fitting data or noise? Analysing the predictive power of commonly used datasets in drug, materials and molecular discovery* (Faraday Discuss. 256, 2025; authors — not verified) | 2024–25 | Performance bounds (aleatoric/realistic bound) obtained by adding noise of the size of the experimental error; 9 datasets; in 4 of 9 the models reached/exceeded the bound; NoiseEstimator package | **Closest competitor on the "ceiling" idea.** The noise is set by an assumed/published error rather than estimated from real replicates across different ChEMBL documents; no embeddings vs ECFP comparison and no splits | [RSC](https://pubs.rsc.org/doi/d4fd00091a) |
| 3 | *Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning* (arXiv 2508.06199) | 2025 | 25 pretrained models × 25 datasets, hierarchical Bayesian test: almost all neural models are no better than ECFP; the exception is CLAMP | No accounting for label noise/ceiling; not in terms of "fraction of ceiling"; our key differences are the ceiling and 3 split types | [arXiv](https://arxiv.org/abs/2508.06199) |
| 4 | *Exposing the Limitations of Molecular Machine Learning with Activity Cliffs* (MoleculeACE, JCIM; authors/year — not verified, ~2022) | 2022 | 24 methods, 30 targets: all are worse on activity cliffs, classical descriptors beat deep models | The data were curated against "false" cliffs, but they do not estimate the noise-induced limit | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9749029/) |
| 5 | J. Cheminformatics 2025, DOI 10.1186/s13321-025-00956-y (title/authors — not verified) | 2025 | Noise in matched molecular pairs: agreement within 0.3 — 44–46% (Ki/IC50) with minimal curation, 66–79% after | Noise of pair differences, not a ceiling for models | [J Cheminf](https://biomedcentral-jcheminf.publicaciones.saludcastillayleon.es/articles/10.1186/s13321-025-00956-y) |
| 6 | *When Molecular Similarity Works: Property Cliffs Reveal Hidden Errors* (arXiv 2605.17265, **preprint**, not verified) | 2026 | An evaluation protocol that identifies cliff regions between train/test | About cliffs, not label noise | [arXiv](https://arxiv.org/pdf/2605.17265) |
| 7 | *On Improving Graph Neural Networks for QSAR* (arXiv 2605.10722, **preprint**, not verified) | 2026 | GNN pretraining on ECFP prediction; gain on 5 of 6 Biogen sets, worse OOD on affinity | A method, not an evaluation methodology; useful as a baseline idea | [arXiv](https://www.alphaxiv.org/abs/2605.10722.md) |
| — | Kalliokoski et al.? *Comparability of Mixed IC50 Data – A Statistical Analysis* (PLoS ONE, 2013; authors not verified) | 2013 | Mixed IC50 from ChEMBL are moderately noisy (SD ≈ 25% higher than Ki) | Outside the 2022–26 window; context | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3628986/) |

## Conclusion: the idea is **partly taken**

- Taken: (a) the fact that ChEMBL inter-assay noise is large (#1); (b) the concept of a "noise-induced performance limit" (#2); (c) "embeddings do not beat ECFP" (#3).
- Free (based on the results of this search): the **combination** of three things — (1) a ceiling estimated from real repeated measurements from **different documents** on 15+ ChEMBL targets, with bootstrap and checks of thresholds/measurement type; (2) the "fraction of ceiling" metric instead of absolute RMSE/R²; (3) a check of whether the embedding–ECFP gap persists in these terms under random/scaffold/temporal splits.
- Recommendation: **continue, with a narrowed scope**. Frame the contribution as an *evaluation protocol + minimum detectable effect*, not "yet another embedding benchmark". Work #2 must be cited and explicitly contrasted (empirical noise from replicates vs. assumed noise). Embeddings serve as a case study.
- Risk: if #2 or its follow-up already estimates from real replicates, the contribution reduces to item (3). Before week 2, read #1, #2, #3 in full and repeat the search (Polaris, TDC, "label noise" ADMET).

## Update (W8, 2026-10-09)

Details and the full table across 6 topics: `docs/literature.md`. Full texts were still not read.

Corrections to the table above (via Crossref/arXiv, metadata cross-checked):

| # | What was corrected |
|---|---|
| 2 | Authors: Daniel Crusius, Flaviu Cipcigan, Philip C. Biggin. Title: *Are we fitting data or noise? Analysing the predictive power of commonly used datasets in drug-, materials-, and molecular-discovery*. Faraday Discuss. 256, 304–321, 2025, DOI 10.1039/d4fd00091a. The RSC page did not open (403); cross-checked via Crossref |
| 3 | Authors: Praski, Adamczyk, Czech (arXiv 2508.06199, 2025) |
| 4 | MoleculeACE: van Tilborg, Alenicheva, Grisoni, JCIM 62(23):5938–5951, 2022, DOI 10.1021/acs.jcim.2c01073 |
| 5 | Title: *Matched pairs demonstrate robustness against inter-assay variability*; Nelen, Pérez-Sánchez, De Winter, Van Rompaey; J. Cheminformatics 17(1), 2025 |
| 6 | Authors: Hu et al. (arXiv 2605.17265, preprint); abstract read: CliffSplit/CliffLoss, QM9 and MoleculeNet |

New works:

| # | Work | Year | Gist | Difference from the project | Link |
|---|---|---|---|---|---|
| 8 | Klamt, Nejdl, Tang — *Decomposing the Generalization Gap in PROTAC Activity Prediction: Variance Attribution and the Inter-Laboratory Ceiling* (arXiv 2605.11764, preprint) | 2026 | Inter-laboratory variance as the main component of the random vs LOTO gap; the estimate bounds the contribution at 0.124 AUROC | **Closest on the "inter-laboratory ceiling" idea**, but a single domain (PROTAC), a binary metric, no ECFP vs embeddings and no "fraction of ceiling" | [arXiv](https://arxiv.org/abs/2605.11764) |
| 9 | Schuh, Daniluk, Sieber — *Auditing widely used biomolecular benchmarks reveals systematic data inconsistencies* (Chem. Sci. 17(36), 2026, DOI 10.1039/d6sc01799a) | 2026 | Audit of Polaris/TDC/MoleculeNet/DTI: leakage, conflicting labels, redundancy; fragility of leaderboards | Data and leaderboards, not a ceiling from replicates and not embeddings vs ECFP | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13425739/) |
| 10 | Ash et al. — *Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug Discovery* (JCIM 65(18):9398–9411, 2025, DOI 10.1021/acs.jcim.5c01609) | 2025 | Method comparison protocols and statistical tests (Polaris); content not read | Overlaps with the "reporting protocol"; in our work the significance threshold is tied to the noise ceiling | [DOI](https://doi.org/10.1021/acs.jcim.5c01609) |
| 11 | Kamuntavičius et al. (J. Cheminformatics 17:108, 2025, DOI 10.1186/s13321-025-01041-0) | 2025 | Friedman+Nemenyi over 25 sets; training on TDC, testing on Biogen | No noise/ceiling | [DOI](https://doi.org/10.1186/s13321-025-01041-0) |

Check for a direct competitor (ceiling estimation from replicates across different ChEMBL documents + comparison of foundation models with ECFP): not found (many queries), but this is an absence in search, not proof. The verdict remains **partly taken**; new: K1 (#8) shows that the idea of an inter-laboratory ceiling already exists, so no priority claim should be made on the term itself. Narrowing: "fraction of ceiling from independent ChEMBL replicates across a set of endpoints + ECFP+GBM vs pretrained embeddings under 3 splits + MDE".
