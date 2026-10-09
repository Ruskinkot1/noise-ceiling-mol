# Этап 0. Проверка новизны (2022–2026)

Метод: 5 поисковых запросов, прочитаны только аннотации/выдержки результатов поиска, полные тексты **не** читались.
Все цифры ниже — из этих выдержек; всё, что не подтверждено, помечено «не проверено». Поиск неполный:
не охвачены, например, работы по шуму в TDC/Polaris, label noise в ADMET, pharma-внутренние отчёты.

| # | Работа | Год | Суть | Отличие от проекта | Ссылка |
|---|---|---|---|---|---|
| 1 | Landrum, Riniker — *Combining IC50 or Ki Values from Different Sources Is a Source of Significant Noise* (JCIM 64(5):1560–1567, DOI 10.1021/acs.jcim.4c00049) | 2024 | Шум между ассеями ChEMBL для одной пары молекула–мишень: при минимальной курации ~65% точек расходятся >0.3 лог.ед., 27% >1 лог.ед., Kendall τ=0.51; при совпадении метаданных τ=0.71; для Ki шум схож | Измеряют шум, но **не** переводят его в предел точности моделей и не пересчитывают метрики моделей | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10934815/) |
| 2 | *Are we fitting data or noise? Analysing the predictive power of commonly used datasets in drug, materials and molecular discovery* (Faraday Discuss. 256, 2025; авторы — не проверено) | 2024–25 | Границы производительности (aleatoric/realistic bound) через добавление шума размером с экспериментальную ошибку; 9 датасетов; в 4 из 9 модели достигли/превысили предел; пакет NoiseEstimator | **Ближайший конкурент по идее «потолка».** Шум задаётся допущенной/опубликованной ошибкой, а не оценивается по реальным повторам из разных документов ChEMBL; нет сравнения эмбеддингов vs ECFP и сплитов | [RSC](https://pubs.rsc.org/doi/d4fd00091a) |
| 3 | *Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning* (arXiv 2508.06199) | 2025 | 25 предобученных моделей × 25 датасетов, иерархический байесовский тест: почти все нейронные модели не лучше ECFP; исключение — CLAMP | Нет учёта шума меток/потолка; не по «доле потолка»; наши ключевые отличия — потолок и 3 типа сплитов | [arXiv](https://arxiv.org/abs/2508.06199) |
| 4 | *Exposing the Limitations of Molecular Machine Learning with Activity Cliffs* (MoleculeACE, JCIM; авторы/год — не проверено, ~2022) | 2022 | 24 метода, 30 мишеней: все хуже на activity cliffs, классические дескрипторы лучше глубоких | Данные курировались против «ложных» cliffs, но предела из-за шума не оценивают | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9749029/) |
| 5 | J. Cheminformatics 2025, DOI 10.1186/s13321-025-00956-y (название/авторы — не проверено) | 2025 | Шум в matched molecular pairs: согласие в пределах 0.3 — 44–46% (Ki/IC50) при минимальной курации, 66–79% после | Шум разностей пар, а не потолок для моделей | [J Cheminf](https://biomedcentral-jcheminf.publicaciones.saludcastillayleon.es/articles/10.1186/s13321-025-00956-y) |
| 6 | *When Molecular Similarity Works: Property Cliffs Reveal Hidden Errors* (arXiv 2605.17265, **препринт**, не проверено) | 2026 | Протокол оценки, выявляющий области cliffs между train/test | Про cliffs, не про шум меток | [arXiv](https://arxiv.org/pdf/2605.17265) |
| 7 | *On Improving Graph Neural Networks for QSAR* (arXiv 2605.10722, **препринт**, не проверено) | 2026 | Предобучение GNN на предсказании ECFP; выигрыш на 5 из 6 Biogen-наборов, хуже OOD на аффинности | Метод, а не методология оценки; полезен как baseline-идея | [arXiv](https://www.alphaxiv.org/abs/2605.10722.md) |
| — | Kalliokoski и др.? *Comparability of Mixed IC50 Data – A Statistical Analysis* (PLoS ONE, 2013; авторы не проверено) | 2013 | Mixed IC50 из ChEMBL шумят умеренно (SD ≈ на 25% выше Ki) | Вне окна 2022–26; контекст | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3628986/) |

## Вывод: идея **частично занята**

- Занято: (а) факт, что межассейный шум ChEMBL велик (#1); (б) концепция «предела производительности из-за шума» (#2); (в) «эмбеддинги не бьют ECFP» (#3).
- Свободно (по результатам этого поиска): **соединение** трёх вещей — (1) потолок, оцененный по реальным повторным измерениям из **разных документов** на 15+ мишенях ChEMBL с бутстрэпом и проверкой порогов/типа измерения; (2) метрика «доля потолка» вместо абсолютного RMSE/R²; (3) проверка, сохраняется ли разрыв эмбеддинги–ECFP в этих терминах при random/scaffold/temporal сплитах.
- Рекомендация: **продолжать с сужением**. Вклад формулировать как *протокол оценки + минимально различимый эффект*, а не «ещё один бенчмарк эмбеддингов». Работу #2 обязательно цитировать и прямо противопоставить (эмпирический шум по повторам vs. допущенный шум). Эмбеддинги — кейс-стади.
- Риск: если #2 или её продолжение уже делают оценку по реальным повторам — вклад сводится к п.(3). Перед нед. 2 полностью прочитать #1, #2, #3 и повторить поиск (Polaris, TDC, "label noise" ADMET).

## Обновление (W8, 2026-10-09)

Подробности и полная таблица по 6 темам: `docs/literature.md`; ключи: `paper/references.bib`. Полные тексты по-прежнему не читались.

Поправки к таблице выше (по Crossref/arXiv, метаданные сверены):

| # | Что исправлено |
|---|---|
| 2 | Авторы: Daniel Crusius, Flaviu Cipcigan, Philip C. Biggin. Название: *Are we fitting data or noise? Analysing the predictive power of commonly used datasets in drug-, materials-, and molecular-discovery*. Faraday Discuss. 256, 304–321, 2025, DOI 10.1039/d4fd00091a. Страница RSC не открылась (403), сверка по Crossref |
| 3 | Авторы: Praski, Adamczyk, Czech (arXiv 2508.06199, 2025) |
| 4 | MoleculeACE: van Tilborg, Alenicheva, Grisoni, JCIM 62(23):5938–5951, 2022, DOI 10.1021/acs.jcim.2c01073 |
| 5 | Название: *Matched pairs demonstrate robustness against inter-assay variability*; Nelen, Pérez-Sánchez, De Winter, Van Rompaey; J. Cheminformatics 17(1), 2025 |
| 6 | Авторы: Hu и др. (arXiv 2605.17265, препринт); аннотация прочитана: CliffSplit/CliffLoss, QM9 и MoleculeNet |

Новые работы:

| # | Работа | Год | Суть | Отличие от проекта | Ссылка |
|---|---|---|---|---|---|
| 8 | Klamt, Nejdl, Tang — *Decomposing the Generalization Gap in PROTAC Activity Prediction: Variance Attribution and the Inter-Laboratory Ceiling* (arXiv 2605.11764, препринт) | 2026 | Межлабораторная дисперсия как главный компонент разрыва random vs LOTO; оценка ограничивает вклад величиной 0.124 AUROC | **Ближайший по идее «межлабораторного потолка»**, но один домен (PROTAC), бинарная метрика, нет ECFP vs эмбеддинги и нет «доли потолка» | [arXiv](https://arxiv.org/abs/2605.11764) |
| 9 | Schuh, Daniluk, Sieber — *Auditing widely used biomolecular benchmarks reveals systematic data inconsistencies* (Chem. Sci. 17(36), 2026, DOI 10.1039/d6sc01799a) | 2026 | Аудит Polaris/TDC/MoleculeNet/DTI: утечки, конфликтующие метки, избыточность; хрупкость лидербордов | Данные и лидерборды, не потолок по повторам и не эмбеддинги vs ECFP | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13425739/) |
| 10 | Ash и др. — *Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug Discovery* (JCIM 65(18):9398–9411, 2025, DOI 10.1021/acs.jcim.5c01609) | 2025 | Протоколы сравнения методов и статтесты (Polaris); содержание не читалось | Пересекается с «протоколом отчётности»; у нас порог значимости привязан к шумовому потолку | [DOI](https://doi.org/10.1021/acs.jcim.5c01609) |
| 11 | Kamuntavičius и др. (J. Cheminformatics 17:108, 2025, DOI 10.1186/s13321-025-01041-0) | 2025 | Friedman+Nemenyi по 25 наборам; обучение на TDC, тест на Biogen | Нет шума/потолка | [DOI](https://doi.org/10.1186/s13321-025-01041-0) |

Проверка прямого конкурента (оценка потолка по повторам из разных документов ChEMBL + сравнение фундаментальных моделей с ECFP): не найден (множество запросов), но это отсутствие в поиске, не доказательство. Вердикт по-прежнему **частично занято**; новое: K1 (#8) показывает, что идея межлабораторного потолка уже есть, приоритет на сам термин не заявлять. Сужение: «доля потолка по независимым повторам ChEMBL на множестве эндпоинтов + ECFP+GBM vs предобученные эмбеддинги при 3 сплитах + MDE».
