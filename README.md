# noise-ceiling-mol

**Потолок шума: что на самом деле измеряют бенчмарки молекулярных свойств, и бьют ли предобученные эмбеддинги ECFP при честной оценке относительно предела измерения.**
Целевой трек: NeurIPS Datasets & Benchmarks. Статус: **ранний прототип** — код шагов 3–5 проверен на синтетических данных, шаги 1–2 на реальной ChEMBL **ещё не запускались**.

## Гипотеза
1. Межлабораторная ошибка в ChEMBL сравнима с ошибкой моделей → часть различий между моделями лежит внутри шума меток.
2. Потолок оцениваем по повторным измерениям одних молекул из **разных документов**.
3. Результаты моделей пересчитываем в **долю потолка**; проверяем, держится ли преимущество предобученных эмбеддингов над ECFP+GBM при random / scaffold / temporal сплитах.

Новизна: частично занята — см. [docs/NOVELTY.md](docs/NOVELTY.md) (ближайшая работа — «Are we fitting data or noise?»).

## Метод потолка (src/ncmol/ceiling.py)
Модель `y_ij = t_i + e_ij`, `e_ij` — шум между документами. Оценка дисперсии шума — объединённая внутримолекулярная дисперсия по молекулам с ≥2 документами (значения внутри документа усредняются, чтобы повторы были именно межисточниковые):

- `RMSE_floor = sqrt(s² · mean(1/n_docs))` (метка = среднее по документам),
- `R²_max = 1 − s²·mean(1/n_docs) / Var(y)`,
- `доля потолка`: `frac_rmse = RMSE_floor / RMSE`, `frac_r2 = R² / R²_max`. Значение >1 = подозрение на подгонку шума или утечку.
- Бутстрэп по молекулам; чувствительность к `min_docs` и обрезке пар `|Δ|` (ошибки единиц).

**Ограничения (писать в статье):** это *вся* межисточниковая изменчивость, включая систематические сдвиги ассеев, а не «чистая» ошибка прибора; допущена гомоскедастичность; реплицированные молекулы могут быть нетипичны для набора.

## Структура
```
configs/endpoints.yaml   кандидаты мишеней (chembl_id по памяти — НЕ ПРОВЕРЕНО), пороги
sql/replicates.sql       выборка измерений одной мишени (IC50/Ki, '=', nM, pChEMBL, confidence>=8)
sql/discover_targets.sql ранжирование мишеней по числу молекул с >=2 документами
src/ncmol/               chem (стандартизация, ECFP), ceiling, splits, models, metrics
scripts/01_extract_chembl.py   выгрузка из SQLite ChEMBL (+ --discover)
scripts/02_standardize.py      RDKit-стандартизация, один y на (молекула, документ), tasks.csv
scripts/03_estimate_ceiling.py потолок + бутстрэп + чувствительность
scripts/04_run_benchmark.py    ECFP+GBM/RF/MLP × 3 сплита × 5 сидов
scripts/05_fraction_of_ceiling.py  доля потолка, лидерборд, парное сравнение с ECFP+GBM
tests/                   синтетические тесты (восстановление σ, сплиты)
docs/NOVELTY.md          этап 0
```

## Запуск
```bash
pip install -e .            # + pytest
pytest                      # синтетические тесты
# ChEMBL SQLite скачать в data/chembl/ (имя файла зависит от релиза)
python scripts/01_extract_chembl.py --db data/chembl/chembl_XX.db --discover --top 60 --min-mols 100
python scripts/01_extract_chembl.py --db data/chembl/chembl_XX.db     # или --targets CHEMBL...
python scripts/02_standardize.py        # печатает, сколько задач прошло порог (нужно >= 15)
python scripts/03_estimate_ceiling.py
python scripts/04_run_benchmark.py
python scripts/05_fraction_of_ceiling.py
```
Задача = (мишень, тип измерения); IC50 и Ki считаются отдельно.

## Эмбеддинги и D-MPNN (ещё не реализовано)
- Эмбеддинги: интерфейс готов — `data/embeddings/<name>.npz` (`mol_ids`, `X`), модель `emb-<name>_gbm` / `_mlp` / `_ridge`. Скрипта инференса нет; выбор 3–4 открытых моделей — **не сделан**.
- D-MPNN: через chemprop, **не подключён**.

## План и критерии остановки
Нед.1 новизна (сделано, частично занято) · 2–3 данные (стоп: <15 задач → сузить) · 4–5 потолок (стоп: потолок ≈ идеал → переформулировать) · 6–7 модели · 8–9 анализ · 10 рекомендации (MDE, протокол) · 11–12 статья, код, лидерборд.

## Проверено / не проверено
- Проверено: `pytest` (3 теста); шаги 3→5 на синтетике (σ восстановлен 0.42 при истинном 0.4).
- Не проверено: шаги 1–2 на реальной ChEMBL; список мишеней; все ссылки из NOVELTY.md сверены лишь по выдержкам поиска.

## Project skills

Claude skills for this project live in `.claude/skills/` (`ncm-researcher`, `ncm-novelty-check`, `ncm-collect-replicates`, `ncm-decontaminate`, `ncm-benchmark-models`, `ncm-ceiling-metric`). Sprint notes: `docs/sprint-40min.md`. Early pilot collector: `scripts/collect_replicates.py`.
