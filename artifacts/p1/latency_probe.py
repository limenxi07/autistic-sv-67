"""Diagnostic only: black-box transition latency, not acceptance timing."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[name]='1'
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import hashlib,json,statistics
from time import perf_counter
from agents.p1 import state_key
from grid_adventure.grid import step
from grid_adventure.step import Action
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
from tests.p1_fixtures import fixture_cases
from tools.benchmark_p1 import provenance
samples=[];keys=[];case_counts={}

def observe(label,state):
    t=perf_counter();state_key(state);keys.append(perf_counter()-t)
    for action in Action:
        t=perf_counter();step(state,action);samples.append(perf_counter()-t)
    case_counts[label]=case_counts.get(label,0)+1

for label,builder in PUBLIC_GAMEPLAY_LEVEL_BUILDERS.items():
    state=builder();state.turn_limit=150;observe(label,state)
for split in ('mechanics','tuning','heldout'):
    for label,(builder,witness) in fixture_cases(split).items():
        state=builder();observe(label,state)
        for action in witness:
            state=step(state,action);observe(label,state)

def summary(values):
    ordered=sorted(values)
    return {'count':len(values),'mean_seconds':statistics.mean(values),
            'p99_seconds':ordered[int(.99*(len(ordered)-1))],'max_seconds':max(values)}
result={'diagnostic_only':True,'scope':'all22 public initial states; mechanic and generated witness trace states; every Action per state',
        'environment':provenance(),'simulation':summary(samples),'state_key':summary(keys),
        'case_state_counts':case_counts,'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'solver_sha256':hashlib.sha256(Path('agents/p1.py').read_bytes()).hexdigest()}
Path('artifacts/p1/latency.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('simulation','state_key')},indent=2))
