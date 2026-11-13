import json
from pathlib import Path
from tools.benchmark_p1 import load_agent
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
from tests.v3_fixtures import image_env

root = Path('artifacts/v4-episode-diagnostic-20261113')
source = root/'task2_agent_diagnostic.py'
rows=[]
for notebook in (False,True):
    if notebook:
        ns={}
        exec('def cell():\n'+''.join('    '+line+'\n' for line in source.read_text().splitlines())+'    return Agent\n',ns)
        cls=ns['cell']()
    else:
        cls=load_agent(str(source))
    for label in ('B3 So Close Yet So Far','B4 Measure Twice and Cut Once'):
        env=image_env(PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label](),seed=42)
        obs,_=env.reset();agent=cls();win=lose=False
        try:
            for turn in range(150):
                action=agent.step(obs)
                obs,_,win,lose,_=env.step(action)
                if win or lose:break
        except Exception as exc:
            marker='V4_EPISODE_4023 '
            assert str(exc).startswith(marker),str(exc)
            report=json.loads(str(exc)[len(marker):])
            assert len(report['steps'])==40,report
            assert report['steps'][0]['turn']==0
            assert report['transition_cache_hits']>0,report
            assert report['stages_inclusive']['winning_replay']['calls']>=1,report
            assert report['stages_inclusive']['cnn_total']['calls']>=1,report
            rows.append({'notebook':notebook,'label':label,'report':report})
        else:raise AssertionError('Missing deliberate diagnostic')
(root/'local-verification.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([{'notebook':r['notebook'],'label':r['label'],'trigger':r['report']['trigger'],'cache_hits':r['report']['transition_cache_hits'],'cache_misses':r['report']['transition_cache_misses'],'elapsed':r['report']['elapsed']} for r in rows],indent=2))
