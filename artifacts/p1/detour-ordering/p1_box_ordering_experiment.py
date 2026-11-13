from time import perf_counter
from unittest.mock import patch
import json
import agents.p1 as p
from grid_adventure.grid import step
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
base=p.RouteOrdering
for penalty in (0,2,4,16,32):
 class Ordering(base):
  def distance(self,origin,target,phase=False,protection=0,obstacles=()):
   return super().distance(origin,target,phase,protection,tuple((point,penalty if cost==8 else cost) for point,cost in obstacles))
 s=PUBLIC_GAMEPLAY_LEVEL_BUILDERS['B6 Warehouse']();s.turn_limit=150
 calls=[0];first=[None];t=perf_counter()
 def sim(state,action):
  calls[0]+=1;n=step(state,action)
  if n.win and first[0] is None:first[0]=(calls[0],perf_counter()-t,n.score)
  return n
 with patch.object(p,'grid_step',sim),patch.object(p,'RouteOrdering',Ordering):
  r=p.search(s,deadline=900,mode='guided',clock=lambda:float(calls[0]))
 print(json.dumps({'penalty':penalty,'first':first[0],'score':r.final_score,'calls':calls[0]}),flush=True)
