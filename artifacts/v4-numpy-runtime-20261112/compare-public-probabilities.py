"""Compare old Torch and new NumPy probabilities on public inputs only."""
import json
from pathlib import Path
import numpy as np
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
from tests.v3_fixtures import image_env
from tools.benchmark_p1 import load_agent

if __name__ == '__main__':
    old = load_agent('artifacts/v4-notebook-repair-20261110/frozen/task2_agent.py')
    old_ns = old._step_image.__globals__
    new = load_agent('artifacts/v4-numpy-runtime-20261112/candidate/task2_agent.py')
    new_ns = new._step_image.__globals__
    rows = []
    for label, builder in PUBLIC_GAMEPLAY_LEVEL_BUILDERS.items():
        for seed in (2, 42, 43, 20261109):
            obs, _ = image_env(builder(), seed=seed).reset()
            old(); left = old_ns['classify_evidence'](obs).scores.probabilities
            new(); right = new_ns['classify_evidence'](obs).scores.probabilities
            a = np.sort(left,axis=-1); b = np.sort(right,axis=-1)
            admissible_old = (a[:,:,-1] >= .9) & (a[:,:,-1]-a[:,:,-2] >= .1)
            admissible_new = (b[:,:,-1] >= .9) & (b[:,:,-1]-b[:,:,-2] >= .1)
            rows.append({'label':label,'seed':seed,'cells':left.shape[0]*left.shape[1], 'max_probability_error':float(np.abs(left-right).max()),'changed_argmax':int((left.argmax(-1)!=right.argmax(-1)).sum()),'changed_admissibility':int((admissible_old!=admissible_new).sum())})
    report = {'boards':len(rows),'cells':sum(r['cells'] for r in rows),'max_probability_error':max(r['max_probability_error'] for r in rows),'changed_argmax':sum(r['changed_argmax'] for r in rows),'changed_admissibility':sum(r['changed_admissibility'] for r in rows),'results':rows}
    Path('artifacts/v4-numpy-runtime-20261112/public-probability-parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}))
    assert report['changed_argmax']==report['changed_admissibility']==0
