"""Pre-outcome audit correction: strict two-gem heldout families, no model calls."""
import json
from pathlib import Path
from tools.v3_data import canonical, digest, layout_record, render_boards, audit_boards, validate_manifest
root=Path('artifacts/v3/data')
manifest=json.loads((root/'manifest-v4.json').read_text())
parent=digest(canonical(manifest))
manifest['parent_manifest_sha256']=parent
manifest['final_revision']='v5: fresh corpus after review ambiguity-reconciliation fix; v2/v3/v4 retired before any model outcomes'
previous_final=[r for r in manifest['layouts'] if r['split']=='final']
manifest['layouts']=[r for r in manifest['layouts'] if r['split']!='final']
seen={r['sha256'] for r in manifest['layouts']} | {r['sha256'] for r in previous_final}
for old in ('manifest-v2.json','manifest-v3.json'):
    seen.update(r['sha256'] for r in json.loads((root/old).read_text())['layouts'] if r['split']=='final')
for previous in previous_final:
    index=previous['index'] + 20000
    record=layout_record(index,'final',multiple_gems=True)
    while record['sha256'] in seen:
        index+=1000
        record=layout_record(index,'final',multiple_gems=True)
    seen.add(record['sha256'])
    manifest['layouts'].append(record)
validate_manifest(manifest)
(root/'manifest-v5.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('manifest_sha256',digest(canonical(manifest)),flush=True)
if not (root/'final-300-v5').exists():
    print(render_boards(manifest,'final','final',root/'final-300-v5',count=100,appearances=3),flush=True)
audit=audit_boards(manifest,root/'final-300-v5')
print({k:v for k,v in audit.items() if k!='pixels_sha256'},flush=True)
