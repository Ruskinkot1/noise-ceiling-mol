# paper/ — статья для NeurIPS Datasets & Benchmarks (черновик W7)

## Структура
- `main.tex` — каркас; подключает `neurips_2026.sty`, если он лежит рядом, иначе `article` (пометка «заменить на официальный стиль»).
- `sections/`: abstract, introduction (объяснения для химиков и ML-специалистов), related (противопоставление «Are we fitting data or noise?»), data, ceiling (формулы из `src/ncmol/ceiling.py` и `docs/experiment_design.md`), benchmark, results (только заглушки), recommendations (MDE, протокол, чек-лист), limitations, datasheet.

## Проверено при написании (2026-10-09)
- Архив стиля `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip` скачивается (HTTP 200), внутри `neurips_2026.sty`, `neurips_2026.tex`, `checklist.tex`. Ссылка взята со страницы NeurIPS 2026 CFP. В репозиторий файлы НЕ копировались; опции D&B (в т.ч. режим двойной слепоты) не проверены.
- Страница D&B 2025 получена: Croissant-метаданные, хостинг (Dataverse/Kaggle/HF/OpenML), доступность кода и данных для рецензентов. Страница D&B 2026 вернула 404 -> требования 2026, лимит страниц и чек-лист: **не проверено**.
- Лицензия ChEMBL: FTP README говорит CC BY-SA 3.0 Unported (проверено); последний релиз на README — chembl_37 от 01/05/2026, используемый релиз не выбран.

## TBD
Все числа результатов, релиз/дата ChEMBL, фильтры, число задач, список бенчмарков для деконтаминации, выбор фундаментальных моделей, бюджет подбора гиперпараметров, поправка на аттенюацию, метод поправки на множественные сравнения, валидация формулы MDE, авторы, лицензия кода, DOI/хостинг/Croissant.

## Нужные ссылки для references.bib (W8). Сейчас в тексте `\cite{TODO-...}`
- TODO-chembl — ChEMBL (статья базы; выбрать актуальную)
- TODO-landrum-riniker — Landrum, Riniker, JCIM 64(5):1560, DOI 10.1021/acs.jcim.4c00049
- TODO-fitting-data-or-noise — «Are we fitting data or noise?», Faraday Discuss. 256 (2025), DOI-фрагмент d4fd00091a; авторы не проверены
- TODO-mixed-ic50-2013 — «Comparability of Mixed IC50 Data», PLoS ONE 2013; авторы не проверены
- TODO-mmp-noise-2025 — J Cheminf 2025, DOI 10.1186/s13321-025-00956-y; название не проверено
- TODO-pretrained-embeddings-benchmark — arXiv 2508.06199
- TODO-moleculeace — MoleculeACE (JCIM ~2022); авторы не проверены
- TODO-arxiv-2605-17265, TODO-arxiv-2605-10722 — препринты 2026, не проверены
- TODO-chemprop, TODO-chemberta, TODO-molformer — статьи моделей (искать оригиналы)
Когда появится `references.bib`, заменить ключи `TODO-*` на реальные; main.tex подключит bib автоматически.

## Сборка
В среде нет pdflatex/latexmk/tectonic (`which` пусто) — **не собиралось**. Локально: положить `neurips_2026.sty` в `paper/`, затем `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
