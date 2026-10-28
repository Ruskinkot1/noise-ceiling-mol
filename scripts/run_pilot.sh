#!/usr/bin/env bash
# Pilot: 3 tasks x 3 splits. Simple models and heads over embeddings: 5 seeds; GNN: 3 seeds; clean only.
# Deviations from the plan (tuning budget of 3 configurations instead of 6, GNN with 3 seeds) are recorded in docs/REPORT.md.
set -u
export DEVICE="${DEVICE:-cpu}"   # cpu | mps | cuda | auto; for mps/cuda NJOBS=1 is preferable (one process per GPU)
NJOBS="${NJOBS:-4}"
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
CHEAP="ecfp_gbm ecfp_rf ecfp_mlp emb-chemberta-77m-mlm_gbm emb-molformer-xl-10pct_gbm"
TASKS="CHEMBL237_IC50 CHEMBL206_IC50 CHEMBL204_Ki"
jobs=()
for v in clean raw; do for t in $TASKS; do
  jobs+=("$v $t cheap 5 $CHEAP")
done; done
for t in $TASKS; do jobs+=("clean $t gnn 3 gnn_dmpnn"); done
printf '%s\n' "${jobs[@]}" | xargs -P "$NJOBS" -I{} bash -c '
  set -- {}; v=$1; t=$2; k=$3; n=$4; shift 4
  [ -s results/pilot/${v}_${t}_${k}.csv ] && { echo "$v $t $k already done, skipping"; exit 0; }
  python scripts/04_run_benchmark.py --processed data/pilot_$v --emb-dir data/pilot_emb --tasks $t --seeds $n \
    --n-trials 3 --device "${DEVICE:-cpu}" --models "$@" --out results/pilot/${v}_${t}_${k}.csv --pred-out results/pilot/${v}_${t}_${k}.pred.csv \
    > results/pilot/${v}_${t}_${k}.log 2>&1; echo "$v $t $k exit=$?"'
