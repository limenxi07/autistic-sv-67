#!/usr/bin/env bash
set -euo pipefail
# Run from the isolated Omarchy checkout after the V3 reservation ends.
# V3 and P3 share this lock; it covers each entire sequence including workers.
exec 9>/tmp/grid-adventure-timing.lock
flock 9
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=${GRID_PYTHON:-.venv/bin/python}
mkdir -p results/final
sha256sum agents/hierarchical_p3.py > results/final/remote-source-hash.txt
for suite in boss public heldout-v2 stress p3-heldout; do
  case "$suite" in
    boss) entry=Task3Agent; repeat=3; count=2; name=boss ;;
    public) entry=Agent; repeat=3; count=22; name=public ;;
    heldout-v2) entry=Agent; repeat=1; count=200; name=generated ;;
    stress) entry=Task3Agent; repeat=3; count=12; name=stress ;;
    p3-heldout) entry=Task3Agent; repeat=1; count=18; name=holdout ;;
  esac
  taskset -c 0 "$PY" -m tools.benchmark_hierarchical_p3 --source agents/hierarchical_p3.py --entrypoint "$entry" --suite "$suite" --repeat "$repeat" --count "$count" --output "results/final/$name.json"
done
