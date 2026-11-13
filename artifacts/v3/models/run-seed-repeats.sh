#!/usr/bin/env bash
set -u
cd /home/enxi/coding/v3-vision-training-20261109
export TORCHINDUCTOR_CACHE_DIR=/home/enxi/coding/v3-vision-training-20261109/compiler-cache
export TORCHINDUCTOR_COMPILE_THREADS=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
for v3_seed in 20261110 20261111; do
  taskset -c 0,1 .venv/bin/python -u tools/train_v3.py \
    --data artifacts/v3/data/train-200k-v2 \
    --validation artifacts/v3/data/validation-14k-v2 \
    --train-boards artifacts/v3/data/train-boards-600-v2 \
    --boards artifacts/v3/data/known-layout-unseen-art-v2 artifacts/v3/data/new-layout-known-art-v2 artifacts/v3/data/new-layout-unseen-art-v2 \
    --output "artifacts/v3/runs/full-compile-aug-seed${v3_seed}" \
    --compile --compile-mode reduce-overhead --batch 448 --seed "$v3_seed" --epochs 20 --augment \
    --wandb-project autistic-sv-67 --wandb-entity enxi-nanochat --wandb-mode online > "full-compile-aug-${v3_seed}.log" 2>&1
  v3_seed_status=$?
  printf '%s\n' "$v3_seed_status" > "full-compile-aug-${v3_seed}.exit"
  if [ "$v3_seed_status" -ne 0 ]; then
    exit "$v3_seed_status"
  fi
done
