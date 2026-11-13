#!/bin/bash
set -eu
cd /home/enxi/coding/v4-vision-recovery-20261109
export PYTHONPATH=. MPLCONFIGDIR=/tmp/v4-mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec 9>/tmp/grid-adventure-timing.lock
flock -n 9
taskset -c 0 /home/enxi/coding/p2-fast-planner-20261108/.venv/bin/python -m tools.v4_production export > artifacts/v4/models/export.log 2>&1
