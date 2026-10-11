#!/usr/bin/env bash
# Хвост вычислений полного бенчмарка: ждёт эмбеддингов -> головы над эмбеддингами (STAGE=emb) -> GNN на 5 небольших задачах.
# Возобновляемый: можно запускать снова, готовое пропускается.
cd "$(dirname "$0")/.."
while [ ! -s data/full_emb/molformer-xl-10pct.npz ] || [ ! -s data/full_emb/chemberta-77m-mlm.npz ]; do sleep 60; done
echo "эмбеддинги готовы: $(date)"
STAGE=emb NJOBS=2 scripts/run_full.sh
echo "головы над эмбеддингами готовы: $(date)"
STAGE=gnn NJOBS=2 DEVICE="${DEVICE:-cpu}" scripts/run_full_gnn.sh
echo "GNN готова: $(date)"
