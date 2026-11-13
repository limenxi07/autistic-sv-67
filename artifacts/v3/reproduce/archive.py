"""Build only the declared V3 supplementary parts; never collect user config."""
import hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[3]
paths=set()
patterns=['agents/v3*.py','tools/*v3*.py','tests/test_v3*.py','artifacts/v3/runs/**/*',
          'artifacts/v3/models/selected-export/*','artifacts/v3/models/pilot-source/*',
          'artifacts/v3/models/*.txt','artifacts/v3/models/*.sh','artifacts/v3/models/*.json',
          'public_levels/*.py','docs/superpowers/*/2026-11-09-v3-vision*.md']
for pattern in patterns:paths.update(p for p in root.glob(pattern) if p.is_file())
for relative in ['agents/p2_next.py','agents/p1.py','agents/p2.py','tools/benchmark_p1.py',
                 'tools/benchmark_boss.py','tests/p1_fixtures.py','tests/p2_fixtures.py',
                 'tests/test_p1_state.py','tests/v3_fixtures.py','utils.py','pyproject.toml','uv.lock',
                 'artifacts/v3/data/freeze-final-v3.py','artifacts/v3/data/freeze-final-v4.py',
                 'artifacts/v3/data/manifest-v2.json','artifacts/v3/data/manifest-v4.json',
                 'artifacts/v3/models/feature_baseline.py','artifacts/v3/reproduce/archive.py',
                 'artifacts/v3/qualification/b10-initial.png',
                 'artifacts/v3/validation/recorded-jobs.json']:
 p=root/relative
 if not p.is_file():raise FileNotFoundError(relative)
 paths.add(p)
manifest={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)}
out=root/'artifacts/v3/supplementary';out.mkdir(exist_ok=True)
(out/'code-models-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(out/'code-models.tar.gz','w:gz',format=tarfile.PAX_FORMAT) as archive:
 for p in sorted(paths):archive.add(p,arcname=str(p.relative_to(root)))
 archive.add(out/'code-models-manifest.json',arcname='artifacts/v3/supplementary/code-models-manifest.json')
print('Archived',len(paths),'files')
