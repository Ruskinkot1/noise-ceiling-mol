#!/usr/bin/env bash
# Пилот: 3 задачи x 3 сплита. Простые модели и головы над эмбеддингами — 5 сидов, GNN — 3 сида, только clean.
# Отклонения от плана (бюджет подбора 3 конфигурации вместо 6, GNN 3 сида) записаны в docs/REPORT.md.
set -u
export DEVICE="${DEVICE:-cpu}"   # cpu | mps | cuda | auto; для mps/cuda лучше NJOBS=1 (один процесс на GPU)
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
printf '%s\n' "${jobs[@]}" | xargs -P "$NJOBS" -L1 bash scripts/_pilot_job.sh
