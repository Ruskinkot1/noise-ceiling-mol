# paper/ — NeurIPS Datasets & Benchmarks paper (draft W7)

## Structure
- `main.tex` — skeleton; loads `neurips_2026.sty` if it is present alongside, otherwise `article` (with a "replace with the official style" note).
- `sections/`: abstract, introduction (explanations for chemists and ML specialists), related (contrast with "Are we fitting data or noise?"), data, ceiling (formulas from `src/ncmol/ceiling.py` and `docs/experiment_design.md`), benchmark, results (placeholders only), recommendations (MDE, protocol, checklist), limitations, datasheet.

## Verified at the time of writing (2026-10-09)
- The style archive `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip` can be downloaded (HTTP 200) and contains `neurips_2026.sty`, `neurips_2026.tex`, `checklist.tex`. The link was taken from the NeurIPS 2026 CFP page. The files were NOT copied into the repository; the D&B options (including the double-blind mode) have not been verified.
- The D&B 2025 page was retrieved: Croissant metadata, hosting (Dataverse/Kaggle/HF/OpenML), code and data availability for reviewers. The D&B 2026 page returned 404 -> the 2026 requirements, page limit, and checklist: **not verified**.
- ChEMBL license: the FTP README states CC BY-SA 3.0 Unported (verified); the latest release in the README is chembl_37 dated 01/05/2026; the release to be used has not been chosen.

## TBD
All result numbers, the ChEMBL release/date, filters, the number of tasks, the list of benchmarks for decontamination, the choice of foundation models, the hyperparameter-tuning budget, the attenuation correction, the multiple-comparison correction method, validation of the MDE formula, authors, code license, DOI/hosting/Croissant.

## References needed for references.bib (W8). Currently the text contains `\cite{TODO-...}`
- TODO-chembl — ChEMBL (the database paper; choose the current one)
- TODO-landrum-riniker — Landrum, Riniker, JCIM 64(5):1560, DOI 10.1021/acs.jcim.4c00049
- TODO-fitting-data-or-noise — "Are we fitting data or noise?", Faraday Discuss. 256 (2025), DOI fragment d4fd00091a; authors not verified
- TODO-mixed-ic50-2013 — "Comparability of Mixed IC50 Data", PLoS ONE 2013; authors not verified
- TODO-mmp-noise-2025 — J Cheminf 2025, DOI 10.1186/s13321-025-00956-y; title not verified
- TODO-pretrained-embeddings-benchmark — arXiv 2508.06199
- TODO-moleculeace — MoleculeACE (JCIM ~2022); authors not verified
- TODO-arxiv-2605-17265, TODO-arxiv-2605-10722 — 2026 preprints, not verified
- TODO-chemprop, TODO-chemberta, TODO-molformer — the model papers (find the originals)
Once `references.bib` exists, replace the `TODO-*` keys with real ones; main.tex will pick up the bib automatically.

## Build
The environment has no pdflatex/latexmk/tectonic (`which` returns nothing) — **not built**. Locally: put `neurips_2026.sty` in `paper/`, then `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
