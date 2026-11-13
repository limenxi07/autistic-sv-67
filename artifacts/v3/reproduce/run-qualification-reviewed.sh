#!/usr/bin/env bash
set -euo pipefail
cd /tmp/v3-vision-20261109
export MPLCONFIGDIR=/tmp/v3-mpl
V3_PY=/home/enxi/coding/p2-fast-planner-20261108/.venv/bin/python
V3_SOURCE=artifacts/v3/frozen-reviewed/task2_agent.py
V3_OUT=artifacts/v3/qualification-reviewed
finish() { printf '%s\n' "$?" > "$V3_OUT/pipeline.exit"; }
trap finish EXIT
powerprofilesctl get > "$V3_OUT/power-profile-at-start.txt"
[ "$(cat "$V3_OUT/power-profile-at-start.txt")" = performance ]
printf 'public\n' > "$V3_OUT/stage"
taskset -c 0 "$V3_PY" -m tools.benchmark_v3 --observation image --source "$V3_SOURCE" --jobs-json artifacts/v3/qualification/public-jobs.json --output "$V3_OUT/public-66.json"
"$V3_PY" -c 'import json; r=json.load(open("artifacts/v3/qualification-reviewed/public-66.json"))["results"]; assert len(r)==66 and all(x["status"]=="win" and x["effective_performance"]>=.5 and x["evaluation_seconds"]<4.25 for x in r)'
printf 'structured\n' > "$V3_OUT/stage"
taskset -c 0 "$V3_PY" -m tools.benchmark_v3 --observation gridstate --source "$V3_SOURCE" --jobs-json artifacts/v3/qualification/regression-jobs.json --output "$V3_OUT/structured-266.json"
"$V3_PY" - <<'PYGUARD'
import json
from pathlib import Path
candidate=json.loads(Path('artifacts/v3/qualification-reviewed/structured-266.json').read_text())['results']
baseline=json.loads(Path('artifacts/v3/qualification/structured-baseline.json').read_text())['results']
bykey={(r['level'],r['seed']):r for r in baseline}
assert len(candidate)==266 and all(r['status']=='win' and r['reward']>=bykey[(r['label'],r['seed'])]['reward'] for r in candidate)
PYGUARD
printf 'oracle\n' > "$V3_OUT/stage"
taskset -c 0 "$V3_PY" -m tools.benchmark_v3 --bridge-only --source "$V3_SOURCE" --output "$V3_OUT/oracle-22.json"
printf 'final-holdout-unlocked\n' > "$V3_OUT/stage"
taskset -c 0 "$V3_PY" -m tools.benchmark_v3 --observation image --source "$V3_SOURCE" --jobs-json "$V3_OUT/final-jobs.json" --output "$V3_OUT/final-300.json"
printf 'finished\n' > "$V3_OUT/stage"
