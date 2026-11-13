#!/usr/bin/env bash
# Omarchy recipe. Run only from a NEW V4 copy; output directories are create-only.
set -euo pipefail
cd /home/enxi/coding/v4-vision-recovery-20261109
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/v4-mpl XDG_CACHE_HOME=/tmp/v4-cache
CPU_PY=/home/enxi/coding/p2-fast-planner-20261108/.venv/bin/python
exec 9>/tmp/grid-adventure-timing.lock
flock -n 9
if [[ "${1:-}" != "--resume-after-export" ]]; then
    taskset -c 0 "$CPU_PY" -m tools.v4_production export
fi
CANDIDATE=$("$CPU_PY" -c 'import json;print(json.load(open("artifacts/v4/models/candidate.json"))["source"])')
taskset -c 0 "$CPU_PY" -m tools.v4_crop_regressions "$CANDIDATE"
taskset -c 0 "$CPU_PY" -m tools.v4_development
