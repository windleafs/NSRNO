#!/usr/bin/env bash
set -euo pipefail
cd /home_nvme/zhuangyang/nsrno
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
PY=/home/zhuangyang/miniconda3/envs/py310/bin/python
out=out/v5/forward_round2_analytic
mkdir -p "$out"
exec "$PY" -u -m nsrno.train_forward \
  --cache out/v5/forward_cache_v1 --out "$out" \
  --epochs 200 --batch 4 --eval-batch 4 --angles-per-record 2 --workers 4 \
  --loss-scaling record-window --output-scaling train-window --residual-frame analytic \
  --eval-every 5 --train-eval-records 64 --min-epochs 80 \
  --plateau-patience 40 --plateau-min-delta 0.0001 > "$out/train.log" 2>&1
