#!/usr/bin/env bash
# Неочищенные данные (оценка утечки) для тех же 17 задач, что в run_full.sh. Готовые задания пропускаются.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 PREFIX=full DEVICE="${DEVICE:-cpu}"
NJOBS="${NJOBS:-5}"
TASKS=$(python -c "import pandas as pd; print(' '.join(pd.read_csv('data/full_raw/tasks.csv').task))")
[ -s results/full/ceil_raw/ceilings.csv ] || python scripts/03_estimate_ceiling.py --processed data/full_raw --out results/full/ceil_raw --n-boot 1000 | tail -3 || exit 1
CHEAP="ecfp_gbm ecfp_rf ecfp_mlp emb-chemberta-77m-mlm_gbm emb-molformer-xl-10pct_gbm"
for t in $TASKS; do echo "raw $t cheap 5 $CHEAP"; done | xargs -P "$NJOBS" -L1 bash scripts/_pilot_job.sh
echo FULL_RAW_DONE
