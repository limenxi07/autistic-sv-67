import json,subprocess,sys
from pathlib import Path
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
worker=r'''
import json,sys,time
from tools.benchmark_p1 import load_agent
from tests.v3_fixtures import image_env
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS,PUBLIC_LEVEL_MAX_REWARDS
real_clock=time.perf_counter
clock_origin=real_clock();extra=0.;scale=float(sys.argv[3]);env_floor=float(sys.argv[4])
def clock():return clock_origin+scale*(real_clock()-clock_origin)+extra
time.perf_counter=clock
cls=load_agent(sys.argv[1])
label=sys.argv[2];env=image_env(PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label](),seed=42);obs,_=env.reset();a=cls();start=clock();win=lose=False;reward=0;turns=0;first=None
while not(win or lose) and turns<150 and clock()-start<5:
 action=a.step(obs)
 if first is None:first={'seconds':clock()-start,'info':a._planner.info() if a._planner else a.info()}
 t=real_clock();obs,r,win,lose,_=env.step(action);extra+=max(0,env_floor-scale*(real_clock()-t));reward+=r;turns+=1
print(json.dumps({'label':label,'win':win,'reward':reward,'target':PUBLIC_LEVEL_MAX_REWARDS[label],'seconds':clock()-start,'turns':turns,'reason':a.info()['reason'],'first':first}))
'''
rows=[]
for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS:
 p=subprocess.run([sys.executable,'-c',worker,sys.argv[1],label,sys.argv[3],sys.argv[4]],capture_output=True,text=True,timeout=25)
 if p.returncode:raise RuntimeError(p.stderr)
 row=json.loads(p.stdout);rows.append(row);print(json.dumps(row),flush=True)
Path(sys.argv[2]).write_text(json.dumps(rows,indent=2)+'\n')
