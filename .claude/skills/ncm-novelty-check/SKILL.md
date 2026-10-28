---
name: ncm-novelty-check
description: Check whether the noise-ceiling idea (inter-assay noise ceiling for molecular property benchmarks, pretrained embeddings vs ECFP) is already published. Use at the start of the project or before writing claims in the paper.
---

# Novelty check

1. Search arXiv, OpenReview, Google Scholar, ChemRxiv for 2022–2026 work on: noise ceiling, inter-assay / inter-lab variability, label noise in ChEMBL and TDC, activity cliffs as a performance limit, pretrained molecular embeddings vs ECFP.
2. For each work record: title, year, venue, what it does, how it differs from noise-ceiling-mol, link. Mark anything you did not open as "не проверено".
3. Never invent references. If a paper cannot be found, say so.
4. Finish with a verdict: free / partly taken / taken, and a suggested narrowing if not free.

Output: a table in `docs/novelty.md`, in Russian.
