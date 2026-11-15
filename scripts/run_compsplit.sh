#!/usr/bin/env bash
# Строгий сплит по связным компонентам графа «молекулы — документы», проверка устойчивости
# document_split на подмножестве задач. Задачи передаются как prefix:task.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 DEVICE="${DEVICE:-cpu}" PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
NJOBS="${NJOBS:-6}"; SEEDS="${SEEDS:-5}"
CHEAP="ecfp_gbm ecfp_rf emb-chemberta-77m-mlm_gbm emb-molformer-xl-10pct_gbm"

run_one() {
  local P=${1%%:*} t=${1##*:}
  mkdir -p "results/$P"
  [ -s "results/$P/comp_clean_${t}_cheap.csv" ] && { echo "$P $t уже готово"; return 0; }
  python scripts/04_run_benchmark.py --processed "data/${P}_clean" --emb-dir "data/${P}_emb" \
    --tasks "$t" --seeds "$SEEDS" --n-trials 3 --device "$DEVICE" --splits component --models $CHEAP \
    --out "results/$P/comp_clean_${t}_cheap.csv" --pred-out "results/$P/comp_clean_${t}_cheap.pred.csv" \
    > "results/$P/comp_clean_${t}_cheap.log" 2>&1
  echo "$P $t exit=$?"
}
export -f run_one; export CHEAP SEEDS DEVICE
printf '%s\n' "$@" | xargs -P "$NJOBS" -L1 bash -c 'run_one "$0"'
echo COMPSPLIT_DONE
