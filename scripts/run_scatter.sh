#!/usr/bin/env bash
set -euo pipefail
cd /home_nvme/zhuangyang/nsrno
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mode="$1"
export CUDA_VISIBLE_DEVICES="$2"
case "$mode" in local|global) ;; *) exit 2 ;; esac
out="out/v5/forward_round3_$mode"
mkdir -p "$out"
exec /home/zhuangyang/miniconda3/envs/py310/bin/python -u -m nsrno.train_scatter \
  --aggregation "$mode" --out "$out" > "$out/train.log" 2>&1
