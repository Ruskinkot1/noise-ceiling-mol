# Модели (W4)

Проверено 2026-10-09. Всё, что не подтверждено карточкой/кодом, помечено «не проверено».

## Таблица

| Модель (ключ в scripts/06) | Репозиторий @ revision | Лицензия | Размер весов | Дата отсечения обучения | Как запускается | Результат проверки |
|---|---|---|---|---|---|---|
| ECFP4 (счётчики, 2048) + GBM/RF/MLP/Ridge | rdkit + sklearn | BSD (rdkit, sklearn) | — | — | `--models ecfp_gbm ecfp_rf ecfp_mlp` | работает (тесты pipeline) |
| D-MPNN (`gnn_dmpnn`) | собственный код `src/ncmol/gnn.py`, чистый torch | код проекта | обучается с нуля | — | `--models gnn_dmpnn` | pytest зелёный; 200 синтетических молекул прогнаны через scripts/04 (3 сплита); см. ниже |
| ChemBERTa-77M-MLM (`chemberta-77m-mlm`) | DeepChem/ChemBERTa-77M-MLM @ ed8a5374 | в карточке/метаданных HF не указана: не проверено | 13,7 МБ (pytorch_model.bin) | не проверено | `06_embed_foundation.py --models chemberta-77m-mlm` -> `emb-chemberta-77m-mlm_gbm` | инференс на 20 мол. OK, dim 384, NaN нет, 1 невалидный SMILES обработан (нулевой вектор, `valid=False`) |
| ChemBERTa-77M-MTR (`chemberta-77m-mtr`) | DeepChem/ChemBERTa-77M-MTR @ 66b895cb | не проверено (как выше) | 14,0 МБ | не проверено | аналогично | инференс на 20 мол. OK, dim 384 |
| ChemBERTa-5M-MLM (`chemberta-5m-mlm`) | DeepChem/ChemBERTa-5M-MLM @ 695672a6 | не проверено | 13,7 МБ | не проверено | аналогично | инференс на 20 и на 200 мол. OK, dim 384 |
| MoLFormer-XL-both-10pct (`molformer-xl-10pct`) | ibm-research/MoLFormer-XL-both-10pct @ 361063d0 | Apache-2.0 (метаданные и карточка HF) | 187 МБ (safetensors) | карточка: 10% ZINC + 10% PubChem, дата отсечения не указана: не проверено | аналогично; `trust_remote_code=True` | код `modeling_molformer.py` (873 строки) и `configuration_molformer.py` прочитан/просмотрен grep'ом: только torch/transformers, нет сети/файлов/exec/pickle; токенайзер стандартный (tokenizer.json). Инференс на 20 мол. OK, dim 768 |
| roberta_zinc_480m (`roberta-zinc-480m`) | entropy/roberta_zinc_480m @ 05b00ec3 | MIT (строка «License: MIT» в карточке; в метаданных HF поля license нет) | 409 МБ (качается только model.safetensors, без дублирующего .bin) | ZINC, дата не указана: не проверено | аналогично | инференс на 20 мол. OK, dim 768 |

Не брались: Uni-Mol, GROVER (тяжёлые/3D, по заданию), ChemGPT-1.2B (4,9 ГБ, >500 МБ), seyonec/ChemBERTa-zinc-base-v1 (179 МБ, лицензия не указана; дублирует семейство ChemBERTa).
Максимум одной загрузки: 409 МБ (<500 МБ). Суммарный кэш HF после всех проверок ~610 МБ в `~/.cache/huggingface` (в репозиторий не попадает; при нехватке места удалить).

## Как запускать
```
pip install -e .            # torch, transformers, rdkit, sklearn уже стоят
python scripts/06_embed_foundation.py                      # все модели -> data/embeddings/<ключ>.npz
python scripts/06_embed_foundation.py --models molformer-xl-10pct --smoke 20   # проверка на 20 молекулах
python scripts/04_run_benchmark.py --models ecfp_gbm gnn_dmpnn emb-chemberta-77m-mlm_gbm emb-molformer-xl-10pct_gbm
```
`npz`: `mol_ids`, `X` (n x dim), `valid` (маска разобранных SMILES), `revision`, `repo`. Кэш: если файл есть, покрывает все
mol_id и revision совпал, модель пропускается (`--force` пересчитывает). Эмбеддинги = mean-pooling `last_hidden_state`
с маской внимания по канонизированным (RDKit) SMILES, батчи по длине, усечение 256 токенов, CPU, 4 потока, модели заморожены.
Суффикс головы в имени (`_gbm`, `_rf`, `_mlp`, `_ridge`) выбирает sklearn-голову над эмбеддингом.

## Интерфейс GNN
`featurize("gnn_dmpnn", smiles, ids)` возвращает object-массив SMILES (а не матрицу), `make_model("gnn_dmpnn", seed)` даёт
`DMPNNRegressor` с `fit(X_smiles, y)` / `predict`. scripts/04_run_benchmark.py менять не пришлось: `X[tr]` работает одинаково.
D-MPNN: направленные рёбра, depth 3, hidden 200, dropout 0.1, mean-pooling по атомам, FFN; Adam lr 1e-3, батч 64, до 60 эпох;
10% train — внутренняя валидация, ранняя остановка (patience 10) с возвратом лучших весов; цель стандартизуется по train.
Детерминизм: `torch.manual_seed(seed)` + `np.random.default_rng(seed)` (тест: два обучения с одним сидом дают идентичные
предсказания; CPU, 1 поток по умолчанию). Гиперпараметры не подбирались: по `experiment_design.md` (п. 2) бюджет подбора у GNN и
ECFP-моделей должен быть одинаковым; это остаётся задачей до финального запуска.

## Что проверено, что нет
- Проверено: тесты `tests/test_models_gnn.py` без сети (граф, обратные рёбра, невалидный SMILES, детерминизм, обучаемость на 200
  синтетических молекулах: корреляция > 0,5, ранняя остановка, маскированный mean-pooling на крошечной случайной RoBERTa,
  кэш/featurize эмбеддингов); `pytest -q` — 36 passed.
- Проверено: реальный инференс всех 5 моделей на 20 молекулах (с 1 заведомо невалидным SMILES); 200 синтетических молекул прогнаны
  через scripts/06 (ChemBERTa-5M) и scripts/04 (ecfp_gbm, gnn_dmpnn, эмбеддинг+ridge, 3 сплита, 1 сид). Синтетические метки
  ничего не говорят о качестве моделей.
- Время D-MPNN: 2000 молекул, 10 эпох ≈ 14 с при загрузке машины ~15 (другие процессы на 4 ядрах); 60 эпох на 8 тыс. молекул оценочно
  несколько минут на модель/сид/сплит. 5 сидов x 3 сплита x N задач x GNN — главная статья времени бенчмарка. На чистой машине не измерено.
- Не проверено: лицензии DeepChem/ChemBERTa-*; даты отсечения всех предобученных корпусов; пересечение корпусов предобучения
  (ZINC/PubChem/ChEMBL) с нашими молекулами (нужна деконтаминация, W-деконтаминация); качество на реальных данных проекта
  (данных `data/processed` в момент проверки не было); влияние усечения 256 токенов (на длинных SMILES); эквивалентность
  нашего D-MPNN реализации chemprop (код написан по описанию Yang et al. 2019, прямого сравнения нет).
- Предупреждение: при высокой загрузке CPU torch с несколькими потоками замедляется на порядок; в GNN поэтому по умолчанию 1 поток
  (`n_threads`).
- ChemBERTa: слой pooler отсутствует в чекпойнтах и инициализируется случайно, но не используется (берётся last_hidden_state).
