#!/usr/bin/env bash
set -euo pipefail
cd /home_nvme/zhuangyang/nsrno
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
out=out/v5/forward_round3_slowness
mkdir -p "$out"
exec /home/zhuangyang/miniconda3/envs/py310/bin/python -u -m nsrno.train_scatter \
  --aggregation global --query-coordinate slowness --out "$out" > "$out/train.log" 2>&1
