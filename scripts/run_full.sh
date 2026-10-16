#!/usr/bin/env bash
# Полный эксперимент: 17 задач, прошедших порог после 02b. Очищенные данные; ECFP-модели + замороженные эмбеддинги, 5 сидов.
# GNN и дообучение здесь не входят (часы на задачу). Готовые задания пропускаются.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 PREFIX=full DEVICE="${DEVICE:-mps}" NJOBS="${NJOBS:-6}" PYTORCH_ENABLE_MPS_FALLBACK=1
TASKS="CHEMBL203_IC50 CHEMBL204_Ki CHEMBL205_Ki CHEMBL206_IC50 CHEMBL214_Ki CHEMBL217_Ki CHEMBL233_Ki CHEMBL234_Ki CHEMBL237_IC50 CHEMBL237_Ki CHEMBL251_Ki CHEMBL279_IC50 CHEMBL2971_IC50 CHEMBL325_IC50 CHEMBL4005_IC50 CHEMBL4822_IC50 CHEMBL5145_IC50"
[ -s data/full_emb/molformer-xl-10pct.npz ] || python scripts/06_embed_foundation.py --processed data/full_raw --out-dir data/full_emb --models chemberta-77m-mlm molformer-xl-10pct --device "$DEVICE" || exit 1
mkdir -p results/full
python scripts/03_estimate_ceiling.py --processed data/full_clean --out results/full/ceil_clean --n-boot 1000 | tail -20 || exit 1
CHEAP="ecfp_gbm ecfp_rf ecfp_mlp emb-chemberta-77m-mlm_gbm emb-molformer-xl-10pct_gbm"
for t in $TASKS; do echo "clean $t cheap 5 $CHEAP"; done | xargs -P "$NJOBS" -L1 bash scripts/_pilot_job.sh
echo FULL_CHEAP_DONE
