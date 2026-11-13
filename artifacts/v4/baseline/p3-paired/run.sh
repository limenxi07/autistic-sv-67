#!/bin/bash
set -eu
cd /tmp/v4-vision-recovery-20261109
export MPLCONFIGDIR=/tmp/v4-mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec 9>/tmp/grid-adventure-timing.lock
flock -n 9
[ "$(powerprofilesctl get)" = performance ]
[ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)" = performance ]
taskset -c 0 /home/enxi/coding/p2-fast-planner-20261108/.venv/bin/python -m tools.run_v4_p3_gates > artifacts/v4/baseline/p3-paired/run.log 2>&1
