# Data validity audit (scripts/02c_audit_data.py, cleaned data, 58 target × type pairs)

Full tables: results/audit/*.csv (not in git). Below is only what was computed.

## Stereoisomers: does merging by InChIKey14 inflate the noise?
Pairs of measurements of the same molecule from different documents, weighted by number of pairs:

| Pair type | Pairs | Mean \\|Δ pActivity\\| | Fraction with \\|Δ\\| ≥ 1 |
|---|---|---|---|
| Same full InChIKey | 578 | 0.857 | 0.341 |
| Different full keys, same InChIKey14 | 1,667 | 0.528 | 0.195 |

Pairs with different stereo variants are **not noisier** than pairs with the same variant, so merging by InChIKey14 does not by itself inflate the noise.
Caveats: "different full keys" includes the case where stereochemistry is not specified in one of the documents; the numbers are raw, with no correction for tasks with few pairs;
the fraction of molecules with several full keys is about 4%. The question is closed for this sample, not in general.

## Other
- The fraction of molecules replicated in two or more documents is small (on the order of 1–6% per task).
- Without a filter on large discrepancies, σ is huge for some tasks (hERG IC50: 1.80 pChEMBL), i.e. outliers and unit errors are real; in the ceiling they are limited by the threshold \\|Δ\\| ≤ 3.
- Not verified (results/audit/not_checked.csv): shared authors of documents, assay conditions (description fields were not included in the export), republication of data, tautomers, manual audit of large pairs.
- ChEMBL version: ChEMBL_37, release 2026-05-01 (per API /status, 2026-10-11).
