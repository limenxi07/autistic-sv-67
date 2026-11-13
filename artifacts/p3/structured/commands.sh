#!/usr/bin/env bash
# See HANDOFF.md for environment and scope. Output paths must be fresh.
set -euo pipefail
# Local correctness/package receipt, already completed:
MPLCONFIGDIR=/tmp/p3-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests
.venv/bin/python -m tools.differential_p3 --source agents/p3.py --output /tmp/p3-differential-recheck.json
.venv/bin/python -m tools.validate_p3_witnesses --output /tmp/p3-witness-recheck.json
.venv/bin/python -m tools.qualify_p3 --output /tmp/p3-qualification-recheck.json
# Omarchy from /tmp/p3-structured-20261109, after copying these P3/support files:
# bash tools/run_p3_acceptance.sh
# Diagnostic only (do not use these timings as acceptance):
# taskset -c 0 "$PY" -m tools.profile_p3 --source agents/p3.py --output /tmp/p3-profile-recheck.json
# Reproduce unsuccessful primitive branches (new output paths):
# taskset -c 0 "$PY" -m tools.benchmark_p3 --source artifacts/p3/structured/experiments/cap16384.py --entrypoint Task3Agent --suite stress --count 12 --cases 5 6 7 9 10 11 --repeat 1 --output /tmp/p3-cap-recheck.json
# taskset -c 0 "$PY" -m tools.benchmark_p3 --source artifacts/p3/structured/experiments/gem-ordering.py --entrypoint Task3Agent --suite stress --count 12 --cases 5 6 7 9 10 11 --repeat 1 --output /tmp/p3-gem-ordering-recheck.json
# The combined cap+ordering outcome is a local diagnostic; reproduce as a separate
# root profile (--source experiments/gem-cap16384.py), not as a qualified timing.
