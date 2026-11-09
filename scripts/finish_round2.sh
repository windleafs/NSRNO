#!/usr/bin/env bash
set -euo pipefail
cd /home_nvme/zhuangyang/nsrno
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/zhuangyang/miniconda3/envs/py310/bin/python
out=out/v5/forward_round2_analytic
for attempt in $(seq 1 180); do
  if test -f "$out/training_best_all_angles.json"; then break; fi
  sleep 5
done
test -f "$out/training_best_all_angles.json"
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/evaluate_round2_variants.py --out "$out" > "$out/variant_eval.log" 2>&1 &
variant_pid=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -u scripts/evaluate_input_pairing.py --out "$out" > "$out/pairing_eval.log" 2>&1 &
pairing_pid=$!
wait "$variant_pid"
wait "$pairing_pid"
"$PY" scripts/summarize_round2.py > out/v5/forward_round2_summary/summary.log
