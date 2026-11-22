Каталог для данных. В git лежат обе когорты задач, из которых собран набор на 41 задачу
(выгрузка ChEMBL 37 от 2026-10-10; лицензия ChEMBL — CC BY-SA 3.0, производные данные под той же лицензией):

| Когорта | Выгрузка по мишеням | До деконтаминации | После деконтаминации |
| --- | --- | --- | --- |
| первая, 17 задач | `raw/` | `full_raw/` | `full_clean/` |
| вторая, 24 задачи | `raw_ext/` | `ext_raw/` | `ext_clean/` |

Плюс `processed/`, `processed_clean/`, `pilot_raw/`, `pilot_clean/`. Из `ext_*` коммитятся только
24 задачи, прошедшие отбор; остальные кандидаты (всего в каталогах 57 задач) остаются локальными —
список и признак отбора лежат в `ext_clean/tasks.csv`.

`data/all_clean/` в git не хранится: это каталог симлинков на обе когорты, который собирает
`scripts/15_merge_cohorts.py` вместе с `results/all_clean/`. Запустить его надо один раз после клонирования.

Не лежат: эмбеддинги (`full_emb/`, `ext_emb/`, `pilot_emb/` — пересчитываются scripts/06), реестр бенчмарков (`registry/` — scripts/build_registry.py), `processed_raw/` (копия `processed/`), помолекулярные предсказания моделей (`results/**/*.pred.csv` — пересчитываются scripts/04).

- SQLite-дамп ChEMBL (5.7 ГБ) НЕ используется: данные берутся из REST API (scripts/01).
- `data/raw/`        — вывод scripts/01
- `data/processed/`  — вывод scripts/02
- `data/embeddings/` — эмбеддинги (см. README, раздел «Эмбеддинги»)

## Что где искать для доработок статьи
- Разложение шума на документный сдвиг и остаток, проверка покрытия на молекулах с 3+ документами,
  документы без пересечений: `*.measurements.csv` в `all_clean/` (строка на молекулу и документ).
  Посчитано: `scripts/17_document_offset.py` → `results/all_clean/document_offset.csv`
  (σ_d, σ_e, доля документной составляющей, верхняя граница выигрыша от знания документа),
  `scripts/18_ceiling_validation.py` → `ceiling_validation.csv` (покрытие, однородность дисперсии,
  изолированные документы), `scripts/19_confounding_gradient.py` → `confounding_gradient.csv`
  (связь доли потолка с шумом по четырём разбиениям).
- Сравнение молекул, удалённых деконтаминацией, с оставшимися: диф `full_raw/` против `full_clean/`
  и `ext_raw/` против `ext_clean/`; реестр для этого не нужен.
- Метаданные документов (авторы, журнал, тип) в репозиторий не входят: `scripts/11_provenance_audit.py`
  тянет их из API ChEMBL в локальный кеш.

## Деконтаминация (W5)
- `data/registry/<источник>.csv.gz` — реестр «уже виденных» молекул (scripts/build_registry.py): 51 файл (11 MoleculeNet + 40 TDC), ~10 МБ gz, 42 МБ с `_raw/`; в git не идёт. `_status.csv` — статус скачивания, `_polaris_catalog.csv` — метаданные Polaris (таблицы не скачаны: нужен логин). Подробности: docs/decontamination.md.
- `data/processed_raw/` — копия `data/processed/` без очистки; `data/processed_clean/` — после scripts/02b_decontaminate.py (+ `decontam_loss.csv`, `decontam_models.csv`).
- Пересборка: `python scripts/build_registry.py [--only moleculenet tdc polaris] [--skip-done]`; затем `python scripts/02b_decontaminate.py [--pilot]`.
