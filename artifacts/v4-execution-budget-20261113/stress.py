import json,subprocess,sys
from pathlib import Path
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
worker=r'''
import json,sys,time
from tools.benchmark_p1 import load_agent
from tests.v3_fixtures import image_env
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS,PUBLIC_LEVEL_MAX_REWARDS
real_clock=time.perf_counter
clock_origin=real_clock()
time.perf_counter=lambda:clock_origin+4*(real_clock()-clock_origin)
cls=load_agent('artifacts/v4-execution-budget-20261113/candidate/task2_agent.py')
ns=sys.modules[cls.__module__].__dict__
original=ns['PlannerAdapter'].__init__
def init(self,budget,mode='p2b'):original(self,budget,mode=mode)
ns['PlannerAdapter'].__init__=init
label=sys.argv[1];env=image_env(PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label](),seed=42);obs,_=env.reset();a=cls();start=time.perf_counter();win=lose=False;reward=0;turns=0;first=None
while not(win or lose) and turns<150 and time.perf_counter()-start<6:
 action=a.step(obs)
 if first is None:first={'seconds':time.perf_counter()-start,'info':a._planner.info() if a._planner else a.info()}
 obs,r,win,lose,_=env.step(action);reward+=r;turns+=1
print(json.dumps({'label':label,'win':win,'reward':reward,'target':PUBLIC_LEVEL_MAX_REWARDS[label],'seconds':time.perf_counter()-start,'first':first}))
'''
rows=[]
for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS:
 p=subprocess.run([sys.executable,'-c',worker,label],capture_output=True,text=True,timeout=20)
 if p.returncode:raise RuntimeError(p.stderr)
 row=json.loads(p.stdout);rows.append(row);print(json.dumps(row),flush=True)
Path('artifacts/v4-execution-budget-20261113/stress4.json').write_text(json.dumps(rows,indent=2)+'\n')
