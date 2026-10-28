#!/usr/bin/env bash
# Full pipeline: ChEMBL data -> cleaning -> pilot (3 tasks) -> ceiling -> models -> fraction of ceiling.
# Usage:   bash scripts/run_all.sh            (cpu)
#          DEVICE=mps NJOBS=1 bash scripts/run_all.sh   (Mac Apple Silicon: embeddings and GNN on GPU)
# Can be interrupted and restarted: completed stages are skipped. FORCE=1 recomputes everything.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export DEVICE="${DEVICE:-cpu}"
export NJOBS="${NJOBS:-4}"
PILOT_TASKS="${PILOT_TASKS:-CHEMBL237_IC50 CHEMBL206_IC50 CHEMBL204_Ki}"
step() { echo; echo "=== $* ==="; }
need() { [ "${FORCE:-0}" = 1 ] || ! [ -e "$1" ]; }   # true if the stage needs to be computed

step "tests"; python -m pytest -q
need data/raw/summary_done.flag && { step "01 ChEMBL extraction (tens of minutes)"; python scripts/01_extract_chembl.py; touch data/raw/summary_done.flag; } || echo "01: done, skipping"
need data/processed/tasks.csv && { step "02 standardization"; python scripts/02_standardize.py | tail -5; } || echo "02: done, skipping"
need data/registry/_status.csv && { step "MoleculeNet/TDC registry"; python scripts/build_registry.py; } || echo "registry: done, skipping"
need data/processed_clean/tasks.csv && { step "02b cleaning (no year filter)"; python scripts/02b_decontaminate.py --no-year --n-jobs 4 | tail -15; } || echo "02b: done, skipping"
step "02c data audit"; python scripts/02c_audit_data.py | tail -5 || true
need data/pilot_clean/tasks.csv && { step "pilot preparation"; python scripts/make_pilot.py --tasks $PILOT_TASKS; } || echo "pilot prepared, skipping"
need data/pilot_emb/molformer-xl-10pct.npz && { step "06 embeddings (device=$DEVICE)"; python scripts/06_embed_foundation.py --processed data/pilot_raw --out-dir data/pilot_emb --models chemberta-77m-mlm molformer-xl-10pct --device "$DEVICE"; } || echo "embeddings done, skipping"
step "03 noise ceiling"
for v in clean raw; do python scripts/03_estimate_ceiling.py --processed data/pilot_$v --out results/pilot/ceil_$v --n-boot 1000 | tail -3; done
step "04 models (hours; progress in results/pilot/*.log)"; scripts/run_pilot.sh
step "merging results"; python scripts/merge_pilot.py
step "05 fraction of ceiling"
python scripts/05_fraction_of_ceiling.py --results results/pilot_clean --processed data/pilot_clean --n-boot 300 | tail -30
python scripts/05_fraction_of_ceiling.py --results results/pilot_raw   --processed data/pilot_raw   --n-boot 100 > /dev/null
step "07 slices"; python scripts/07_slices.py --results results/pilot_clean --processed data/pilot_clean | tail -5
echo; echo "Done. Tables: results/pilot_clean/leaderboard.csv, paired_vs_baseline.csv; slices: results/slices/"
