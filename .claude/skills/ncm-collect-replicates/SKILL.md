---
name: ncm-collect-replicates
description: Collect molecules with replicate measurements from independent ChEMBL documents using scripts/collect_replicates.py. Use when building or refreshing the replicate dataset.
---

# Collect replicates

1. Run a pilot on 3–5 targets first:
   `python scripts/collect_replicates.py --targets CHEMBL203 CHEMBL240 CHEMBL220 --types Ki IC50 --min-year 2023 --out data/replicates.csv`
2. Verify every target ID by its `pref_name` in ChEMBL; do not trust IDs from memory.
3. Filters: relation `=`, pchembl_value present, empty data_validity_comment, assay type B, reliable target confidence.
4. Standardize with RDKit (largest fragment, neutralize, InChIKey). Average repeats inside one document.
5. Keep (molecule, target, type) groups with at least 2 distinct document ids.
6. Report a loss table per filtering step and molecules with >=2 sources per endpoint. Show the pilot to the user before scaling to 15–30 endpoints.

Thresholds (100 molecules per endpoint, 3 log-unit outlier flag) are starting values, tune on the pilot.
