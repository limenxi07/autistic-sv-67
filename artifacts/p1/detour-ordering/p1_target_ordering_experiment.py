from time import perf_counter
from unittest.mock import patch
import json
import agents.p1 as p
from grid_adventure.grid import step
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
base=p.RouteOrdering
for label in ('B5 Ow Oof Owie','B6 Warehouse'):
 for weight in (3.,10.,30.):
  for hazard in ((8.,16.,32.) if label.startswith('B5') else (8.,)):
   class Ordering(base):
    def __init__(self,*args,**kwargs):
     super().__init__(*args,**kwargs);self.hazard_weight=hazard
   s=PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label]();s.turn_limit=150
   calls=[0];first=[None];t=perf_counter()
   def sim(state,action):
    calls[0]+=1;n=step(state,action)
    if n.win and first[0] is None:first[0]=(calls[0],perf_counter()-t,n.score)
    return n
   with patch.object(p,'grid_step',sim),patch.object(p,'RouteOrdering',Ordering),patch.object(p,'SEARCH_WEIGHT',weight):
    r=p.search(s,deadline=900,mode='guided',clock=lambda:float(calls[0]))
   print(json.dumps({'level':label,'weight':weight,'hazard':hazard,'first':first[0],'score':r.final_score,'calls':calls[0]}),flush=True)
