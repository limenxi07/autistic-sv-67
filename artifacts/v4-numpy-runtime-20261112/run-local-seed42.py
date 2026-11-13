"""Local behavior diagnostic, fresh process per public level; not target timing."""
import json,os,subprocess,sys
from pathlib import Path
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

WORKER = r'''
import json,sys,time
from tools.benchmark_p1 import load_agent
from tests.v3_fixtures import image_env
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS,PUBLIC_LEVEL_MAX_REWARDS
label=sys.argv[1]
env=image_env(PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label](),seed=42)
obs,_=env.reset();cls=load_agent('artifacts/v4-numpy-runtime-20261112/candidate/task2_agent.py');a=cls()
start=time.perf_counter();win=lose=False;reward=0;turns=0
while not(win or lose) and turns<150 and time.perf_counter()-start<6:
 obs,r,win,lose,_=env.step(a.step(obs));reward+=r;turns+=1
print(json.dumps({'label':label,'win':win,'lose':lose,'reward':reward,'optimal_reward':PUBLIC_LEVEL_MAX_REWARDS[label],'turns':turns,'episode_seconds':time.perf_counter()-start,'torch_imported':'torch' in sys.modules,'agent_info':a.info()}))
'''
if __name__=='__main__':
    rows=[]
    for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS:
        result=subprocess.run([sys.executable,'-c',WORKER,label],env=os.environ,capture_output=True,text=True,timeout=20,check=False)
        row=json.loads(result.stdout.splitlines()[-1]) if result.returncode==0 else {'label':label,'win':False,'error':result.stderr}
        rows.append(row)
        Path('artifacts/v4-numpy-runtime-20261112/local-seed42.json').write_text(json.dumps({'diagnostic_only':True,'results':rows},indent=2)+'\n')
        print(label,row.get('win'),row.get('reward'),flush=True)
