#!/usr/bin/env bash
# Extension cohort: targets of configs/endpoints_extension.yaml (step 13 selection) through the same
# pipeline as run_full.sh — decontamination, embeddings, noise ceiling, ECFP models and frozen embeddings,
# 3 splits, 5 seeds. Results land in results/ext/ next to results/full/ and are merged for the paper by
# scripts/make_paper_tables.py. Completed jobs are skipped, so the script can be interrupted and rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 PREFIX=ext DEVICE="${DEVICE:-cpu}" PYTORCH_ENABLE_MPS_FALLBACK=1
NJOBS="${NJOBS:-4}"
SEEDS="${SEEDS:-5}"

[ -s data/ext_clean/tasks.csv ] || python scripts/02b_decontaminate.py --processed data/processed_ext \
  --raw-out data/ext_raw --clean-out data/ext_clean --endpoints configs/endpoints_extension.yaml \
  --no-year --n-jobs "$NJOBS" | tail -15 || exit 1
[ -s data/ext_emb/molformer-xl-10pct.npz ] || python scripts/06_embed_foundation.py --processed data/ext_raw \
  --out-dir data/ext_emb --models chemberta-77m-mlm molformer-xl-10pct --device "$DEVICE" || exit 1
mkdir -p results/ext
[ -s results/ext/ceil_clean/ceilings.csv ] || python scripts/03_estimate_ceiling.py --processed data/ext_clean \
  --out results/ext/ceil_clean --n-boot 1000 | tail -20 || exit 1

TASKS=$(python -c "import pandas as pd; t=pd.read_csv('data/ext_clean/tasks.csv'); print(' '.join(t[t.passes].task if 'passes' in t else t.task))")
echo "tasks: $TASKS"
CHEAP="ecfp_gbm ecfp_rf ecfp_mlp emb-chemberta-77m-mlm_gbm emb-molformer-xl-10pct_gbm"
for t in $TASKS; do echo "clean $t cheap $SEEDS $CHEAP"; done | xargs -P "$NJOBS" -L1 bash scripts/_pilot_job.sh
echo EXT_CLEAN_DONE
