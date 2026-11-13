import cProfile,json,pstats,sys,time
from pathlib import Path
from tools.benchmark_p1 import load_agent
from tests.v3_fixtures import image_env
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
source=sys.argv[1];label=sys.argv[2]
cls=load_agent(source)
env=image_env(PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label](),seed=42);obs,_=env.reset();agent=cls();win=lose=False;reward=0;rows=[]
prof=cProfile.Profile();start=time.perf_counter()
for turn in range(150):
 t=time.perf_counter();prof.enable();action=agent.step(obs);prof.disable();elapsed=time.perf_counter()-t
 t=time.perf_counter();obs,r,win,lose,_=env.step(action);gap=time.perf_counter()-t;reward+=r
 rows.append({'turn':turn,'action':str(action),'agent_s':elapsed,'env_s':gap,'reason':agent.info()['reason']})
 if win or lose:break
print(json.dumps({'label':label,'win':win,'reward':reward,'elapsed':time.perf_counter()-start,'agent_s':sum(r['agent_s'] for r in rows),'env_s':sum(r['env_s'] for r in rows),'steps':rows}))
pstats.Stats(prof).sort_stats('cumulative').print_stats(45)
