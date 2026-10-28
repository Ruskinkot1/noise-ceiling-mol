Data directory (not committed to git).
- The ChEMBL SQLite dump (5.7 GB) is NOT used: data comes from the REST API (scripts/01).
- `data/raw/`        — output of scripts/01
- `data/processed/`  — output of scripts/02
- `data/embeddings/` — embeddings (see scripts/06_embed_foundation.py and docs/models.md)

## Decontamination
- `data/registry/<source>.csv.gz` — registry of "already seen" molecules (scripts/build_registry.py): 51 files (11 MoleculeNet + 40 TDC), ~10 MB gz, 42 MB with `_raw/`; not tracked in git. `_status.csv` — download status, `_polaris_catalog.csv` — Polaris metadata (the tables were not downloaded: a login is required). Details: docs/decontamination.md.
- `data/processed_raw/` — copy of `data/processed/` without cleaning; `data/processed_clean/` — after scripts/02b_decontaminate.py (+ `decontam_loss.csv`, `decontam_models.csv`).
- Rebuild: `python scripts/build_registry.py [--only moleculenet tdc polaris] [--skip-done]`; then `python scripts/02b_decontaminate.py [--pilot]`.
