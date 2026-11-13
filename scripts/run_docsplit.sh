#!/usr/bin/env bash
# Сплит по документам на обеих когортах. Документ целиком в train или test, поэтому у тестовой
# молекулы нет в обучении ни одной молекулы из её документа. Проверяет, что превышение потолка
# шума (f > 1) на random/scaffold создаётся поддержкой документа, а не обобщением.
# Результаты: results/<prefix>/doc_clean_<task>_cheap.csv, рядом с остальными.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 DEVICE="${DEVICE:-cpu}" PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"   # worktree, не основной чекаут
NJOBS="${NJOBS:-4}"
SEEDS="${SEEDS:-5}"
CHEAP="ecfp_gbm ecfp_rf ecfp_mlp emb-chemberta-77m-mlm_gbm emb-molformer-xl-10pct_gbm"

run_one() {   # $1 prefix  $2 task
  local P=$1 t=$2
  mkdir -p "results/$P"
  [ -s "results/$P/doc_clean_${t}_cheap.csv" ] && { echo "$P $t уже готово, пропуск"; return 0; }
  python scripts/04_run_benchmark.py --processed "data/${P}_clean" --emb-dir "data/${P}_emb" \
    --tasks "$t" --seeds "$SEEDS" --n-trials 3 --device "$DEVICE" --splits document --models $CHEAP \
    --out "results/$P/doc_clean_${t}_cheap.csv" --pred-out "results/$P/doc_clean_${t}_cheap.pred.csv" \
    > "results/$P/doc_clean_${t}_cheap.log" 2>&1
  echo "$P $t exit=$?"
}
export -f run_one; export CHEAP SEEDS DEVICE

for P in full ext; do
  TASKS=$(python -c "import pandas as pd; t=pd.read_csv('data/${P}_clean/tasks.csv'); print(' '.join(t[t.passes].task))")
  for t in $TASKS; do echo "$P $t"; done
done | xargs -P "$NJOBS" -L1 bash -c 'run_one $0 $1'
echo DOCSPLIT_DONE
