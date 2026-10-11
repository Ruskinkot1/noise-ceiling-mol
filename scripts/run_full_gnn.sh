#!/usr/bin/env bash
# Полный бенчмарк на очищенных данных (задачи из data/fullgnn_clean). Возобновляемый: задание = (задача, сплит), готовые пропускаются.
# STAGE=ecfp  — ECFP+GBM/RF (эмбеддинги не нужны);  STAGE=emb — головы над эмбеддингами (нужен data/full_emb);  STAGE=gnn — D-MPNN.
# Сокращённый протокол (отклонение от плана, см. docs/REPORT.md): 3 сида, подбор 2 конфигурации на модель, без ECFP+MLP.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export DEVICE="${DEVICE:-cpu}" STAGE="${STAGE:-gnn}"
NJOBS="${NJOBS:-3}"; SEEDS="${SEEDS:-3}"; TRIALS="${TRIALS:-2}"
case "$STAGE" in
  ecfp) MODELS="ecfp_gbm ecfp_rf" ;;
  emb)  MODELS="emb-chemberta-77m-mlm_gbm emb-molformer-xl-10pct_gbm" ;;
  gnn)  MODELS="gnn_dmpnn" ;;
esac
export MODELS SEEDS TRIALS
mkdir -p results/fullgnn
for d in data/fullgnn_clean/*.molecules.csv; do t=$(basename "$d" .molecules.csv); for sp in random scaffold temporal; do echo "$t $sp"; done; done |
xargs -P "$NJOBS" -L1 bash -c '
  t=$0; sp=$1; out=results/fullgnn/${STAGE}_${t}_${sp}.csv
  [ -s "$out" ] && { echo "$t $sp $STAGE: готово, пропуск"; exit 0; }
  python scripts/04_run_benchmark.py --processed data/fullgnn_clean --emb-dir data/full_emb --tasks $t --splits $sp \
    --seeds $SEEDS --n-trials $TRIALS --device "$DEVICE" --models $MODELS --out $out --pred-out results/fullgnn/${STAGE}_${t}_${sp}.pred.csv \
    > results/fullgnn/${STAGE}_${t}_${sp}.log 2>&1; echo "$t $sp $STAGE exit=$?"'
