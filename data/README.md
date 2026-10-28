Каталог для данных (в git не коммитится).
- SQLite-дамп ChEMBL (5.7 ГБ) НЕ используется: данные берутся из REST API (scripts/01).
- `data/raw/`        — вывод scripts/01
- `data/processed/`  — вывод scripts/02
- `data/embeddings/` — эмбеддинги (см. README, раздел «Эмбеддинги»)

## Деконтаминация (W5)
- `data/registry/<источник>.csv.gz` — реестр «уже виденных» молекул (scripts/build_registry.py): 51 файл (11 MoleculeNet + 40 TDC), ~10 МБ gz, 42 МБ с `_raw/`; в git не идёт. `_status.csv` — статус скачивания, `_polaris_catalog.csv` — метаданные Polaris (таблицы не скачаны: нужен логин). Подробности: docs/decontamination.md.
- `data/processed_raw/` — копия `data/processed/` без очистки; `data/processed_clean/` — после scripts/02b_decontaminate.py (+ `decontam_loss.csv`, `decontam_models.csv`).
- Пересборка: `python scripts/build_registry.py [--only moleculenet tdc polaris] [--skip-done]`; затем `python scripts/02b_decontaminate.py [--pilot]`.
