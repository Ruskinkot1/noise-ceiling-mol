#!/usr/bin/env bash
# Tail of the full-benchmark computation: wait for embeddings -> heads over embeddings (STAGE=emb) -> GNN on 5 small tasks.
# Resumable: can be rerun; completed work is skipped.
cd "$(dirname "$0")/.."
while [ ! -s data/full_emb/molformer-xl-10pct.npz ] || [ ! -s data/full_emb/chemberta-77m-mlm.npz ]; do sleep 60; done
echo "embeddings ready: $(date)"
STAGE=emb NJOBS=2 scripts/run_full.sh
echo "heads over embeddings ready: $(date)"
STAGE=gnn NJOBS=2 DEVICE="${DEVICE:-cpu}" scripts/run_full_gnn.sh
echo "GNN ready: $(date)"
