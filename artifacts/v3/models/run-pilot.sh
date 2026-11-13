#!/usr/bin/env bash
set -u
cd /home/enxi/coding/v3-vision-training-20261109
export TORCHINDUCTOR_CACHE_DIR=/home/enxi/coding/v3-vision-training-20261109/compiler-cache
export TORCHINDUCTOR_COMPILE_THREADS=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
taskset -c 0,1 .venv/bin/python -u tools/train_v3.py \
  --data artifacts/v3/data/train-200k-v2 \
  --validation artifacts/v3/data/validation-14k-v2 \
  --train-boards artifacts/v3/data/train-boards-600-v2 \
  --boards artifacts/v3/data/known-layout-unseen-art-v2 artifacts/v3/data/new-layout-known-art-v2 artifacts/v3/data/new-layout-unseen-art-v2 \
  --output artifacts/v3/runs/pilot-compile-seed20261109 \
  --compile --compile-mode reduce-overhead --batch 448 --seed 20261109 \
  --pilot-seconds 600 --limit-train 20000 > pilot.log 2>&1
v3_pilot_status=$?
printf '%s\n' "$v3_pilot_status" > pilot.exit
exit "$v3_pilot_status"
