# noise-ceiling-mol

**Потолок шума: что на самом деле измеряют бенчмарки молекулярных свойств, и бьют ли предобученные эмбеддинги ECFP при честной оценке относительно предела измерения.**
Целевой трек: NeurIPS Datasets & Benchmarks. Статус: **ранний прототип** — код шагов 3–5 проверен на синтетических данных, шаг 1 переписан под REST API (тест парсинга на моке), шаги 1–2 на полном наборе мишеней **ещё не запускались**.

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
src/ncmol/               chem (стандартизация, ECFP), ceiling, splits, models, metrics
scripts/01_extract_chembl.py   выгрузка из ChEMBL REST API (SQLite-дамп не используется)
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
python scripts/01_extract_chembl.py --targets CHEMBL240     # по API; без --targets — кандидаты из configs/endpoints.yaml
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

Claude skills for this project live in `.claude/skills/` (`ncm-researcher`, `ncm-novelty-check`, `ncm-collect-replicates`, `ncm-decontaminate`, `ncm-benchmark-models`, `ncm-ceiling-metric`). Sprint notes: `docs/sprint-40min.md`. 


## Запуск на Colab
Ноутбук [notebooks/run_on_colab.ipynb](notebooks/run_on_colab.ipynb) проходит весь конвейер (данные → очистка → потолок → модели → доля потолка).
Открыть: https://colab.research.google.com/github/Ruskinkot1/noise-ceiling-mol/blob/cloud-data/notebooks/run_on_colab.ipynb
**Не проверено на Colab.** Всё считается на CPU (GPU не используется), полный пилот занимает часы. Если репозиторий приватный, в ячейке клонирования нужен токен GitHub.

## Запуск на Mac (локально)
Не проверено на macOS (разрабатывалось на Linux). Нужны Python 3.10–3.13 и около 3 ГБ свободного места (данные ~0,5 ГБ, кэш моделей HF ~0,6 ГБ).
```bash
git clone -b cloud-data https://github.com/Ruskinkot1/noise-ceiling-mol.git && cd noise-ceiling-mol
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
export OMP_NUM_THREADS=1            # иначе sklearn GBM бывает в сотни раз медленнее при параллельных процессах
python -m pytest -q
python scripts/01_extract_chembl.py && python scripts/02_standardize.py
python scripts/build_registry.py && python scripts/02b_decontaminate.py --no-year --n-jobs 4
python scripts/make_pilot.py && python scripts/06_embed_foundation.py --processed data/pilot_raw --out-dir data/pilot_emb --models chemberta-77m-mlm molformer-xl-10pct
scripts/run_pilot.sh                # можно прерывать и запускать снова: готовые задания пропускаются
```
Всё считается на CPU (GPU/MPS не используется). Если `pip install rdkit` не находит колёса для вашей версии Python, возьмите Python 3.11–3.12.
