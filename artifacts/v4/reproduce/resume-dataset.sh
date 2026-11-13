#!/usr/bin/env bash
# Resume the immutable dataset transfer only. Never retrain or rerun final data.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/v4-uv-cache}"
test -f artifacts/v4/supplementary/parts.json
V4_DATA=artifacts/v4/supplementary/generated-datasets.tar.gz
if [[ ! -f "$V4_DATA" ]]; then
    rsync --partial --append --progress --timeout=30 \
        -e 'ssh -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1' \
        omarchy:/home/enxi/coding/v4-vision-recovery-20261109/artifacts/v4/supplementary/generated-datasets.tar.gz \
        "$V4_DATA.partial"
fi
uv run --frozen --no-sync python - <<'PY'
import json, runpy
from pathlib import Path
name = 'artifacts/v4/supplementary/generated-datasets.tar.gz'
meta = json.loads(Path('artifacts/v4/supplementary/parts.json').read_text())['v4_parts'][name]
complete = Path(name)
source = complete if complete.exists() else Path(name + '.partial')
api = runpy.run_path('artifacts/v4/reproduce/archive.py')
result = api['verify_part'](source, Path(meta['content_manifest']))
assert result['bytes'] == meta['bytes'] and result['sha256'] == meta['sha256']
assert result['content_manifest_sha256'] == meta['content_manifest_sha256']
if source != complete:
    source.rename(complete)
Path('artifacts/v4/qualification/local-dataset-archive-verified.json').write_text(json.dumps(result, indent=2) + '\n')
print('Complete dataset archive and every member verified. No training/evaluation was run.')
PY
