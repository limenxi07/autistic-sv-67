import runpy,json
from time import perf_counter
from grid_adventure.grid import step
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
for source in ('artifacts/p1/guidance-score/experiment_agent.py','artifacts/p1/guidance-score/reference-key-fixed.py','agents/p1.py'):
 namespace=runpy.run_path(source);function=namespace['search'];scope=function.__globals__
 calls=[0];wins=[];s=PUBLIC_GAMEPLAY_LEVEL_BUILDERS['A6 Pushable Box']();s.turn_limit=150;t=perf_counter()
 def sim(state,action):
  calls[0]+=1;n=step(state,action)
  if n.win and (not wins or n.score>wins[-1][1]):wins.append((calls[0],n.score,perf_counter()-t))
  return n
 scope['grid_step']=sim
 r=function(s,deadline=6000,mode='guided',clock=lambda:float(calls[0]))
 print(json.dumps({'source':source,'wins':wins,'calls':calls[0],'score':r.final_score,'reason':r.stop_reason,'states':r.peak_states,'records':r.peak_records,'seconds':perf_counter()-t}),flush=True)
