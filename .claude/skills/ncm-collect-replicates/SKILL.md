---
name: ncm-collect-replicates
description: Collect molecules with replicate measurements from independent ChEMBL documents using scripts/01_extract_chembl.py (REST API). Use when building or refreshing the replicate dataset.
---

# Collect replicates

1. Run a pilot on 3–5 targets first:
   `python scripts/01_extract_chembl.py --targets CHEMBL203 CHEMBL240 CHEMBL220 --types Ki IC50` (сырьё в data/raw/<ID>.csv; затем `scripts/02_standardize.py`). API недоступен -> остановиться, данные не имитировать. SQLite-дамп (5.7 ГБ) не качать.
2. Verify every target ID by its `pref_name` in ChEMBL; do not trust IDs from memory.
3. Filters: relation `=`, pchembl_value present, empty data_validity_comment, assay type B, assay confidence_score >= 8 (отдельный запрос к assay).
4. Standardize with RDKit (largest fragment, neutralize, InChIKey). Average repeats inside one document.
5. Keep (molecule, target, type) groups with at least 2 distinct document ids.
6. Report a loss table per filtering step and molecules with >=2 sources per endpoint. Show the pilot to the user before scaling to 15–30 endpoints.

Thresholds (100 molecules per endpoint, 3 log-unit outlier flag) are starting values, tune on the pilot.
