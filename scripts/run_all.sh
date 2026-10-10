#!/usr/bin/env bash
# Полный конвейер: данные ChEMBL -> очистка -> пилот (3 задачи) -> потолок -> модели -> доля потолка.
# Запуск:  bash scripts/run_all.sh            (cpu)
#          DEVICE=mps NJOBS=1 bash scripts/run_all.sh   (Mac Apple Silicon: эмбеддинги и GNN на GPU)
# Можно прерывать и запускать снова: готовые этапы пропускаются. FORCE=1 пересчитывает всё.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export DEVICE="${DEVICE:-cpu}"
export NJOBS="${NJOBS:-4}"
PILOT_TASKS="${PILOT_TASKS:-CHEMBL237_IC50 CHEMBL206_IC50 CHEMBL204_Ki}"
step() { echo; echo "=== $* ==="; }
need() { [ "${FORCE:-0}" = 1 ] || ! [ -e "$1" ]; }   # true, если надо считать

step "тесты"; python -m pytest -q
need data/raw/summary_done.flag && { step "01 выгрузка из ChEMBL (десятки минут)"; python scripts/01_extract_chembl.py; touch data/raw/summary_done.flag; } || echo "01: готово, пропуск"
need data/processed/tasks.csv && { step "02 стандартизация"; python scripts/02_standardize.py | tail -5; } || echo "02: готово, пропуск"
need data/registry/_status.csv && { step "реестр MoleculeNet/TDC"; python scripts/build_registry.py; } || echo "реестр: готов, пропуск"
need data/processed_clean/tasks.csv && { step "02b очистка (без фильтра по году)"; python scripts/02b_decontaminate.py --no-year --n-jobs 4 | tail -15; } || echo "02b: готово, пропуск"
step "02c аудит данных"; python scripts/02c_audit_data.py | tail -5 || true
need data/pilot_clean/tasks.csv && { step "подготовка пилота"; python scripts/make_pilot.py --tasks $PILOT_TASKS; } || echo "пилот подготовлен, пропуск"
need data/pilot_emb/molformer-xl-10pct.npz && { step "06 эмбеддинги (device=$DEVICE)"; python scripts/06_embed_foundation.py --processed data/pilot_raw --out-dir data/pilot_emb --models chemberta-77m-mlm molformer-xl-10pct --device "$DEVICE"; } || echo "эмбеддинги готовы, пропуск"
step "03 потолок шума"
for v in clean raw; do python scripts/03_estimate_ceiling.py --processed data/pilot_$v --out results/pilot/ceil_$v --n-boot 1000 | tail -3; done
step "04 модели (часы; прогресс в results/pilot/*.log)"; scripts/run_pilot.sh
step "склейка результатов"; python scripts/merge_pilot.py
step "05 доля потолка"
python scripts/05_fraction_of_ceiling.py --results results/pilot_clean --processed data/pilot_clean --n-boot 300 | tail -30
python scripts/05_fraction_of_ceiling.py --results results/pilot_raw   --processed data/pilot_raw   --n-boot 100 > /dev/null
step "07 разрезы"; python scripts/07_slices.py --results results/pilot_clean --processed data/pilot_clean | tail -5
echo; echo "Готово. Таблицы: results/pilot_clean/leaderboard.csv, paired_vs_baseline.csv; разрезы: results/slices/"
