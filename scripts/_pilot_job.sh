#!/usr/bin/env bash
# Одно задание: _pilot_job.sh <clean|raw> <task> <cheap|gnn> <n_seeds> <models...>
# PREFIX (по умолчанию pilot): data/<PREFIX>_<v>, data/<PREFIX>_emb, results/<PREFIX>/.
# Вынесено из run_pilot.sh: BSD xargs (macOS) ограничивает длину подстановки -I 255 байтами.
P="${PREFIX:-pilot}"
v=$1; t=$2; k=$3; n=$4; shift 4
mkdir -p results/$P
[ -s results/$P/${v}_${t}_${k}.csv ] && { echo "$v $t $k уже готово, пропуск"; exit 0; }
python scripts/04_run_benchmark.py --processed data/${P}_$v --emb-dir data/${P}_emb --tasks $t --seeds $n \
  --n-trials 3 --device "${DEVICE:-cpu}" --models "$@" --out results/$P/${v}_${t}_${k}.csv --pred-out results/$P/${v}_${t}_${k}.pred.csv \
  > results/$P/${v}_${t}_${k}.log 2>&1; echo "$v $t $k exit=$?"
