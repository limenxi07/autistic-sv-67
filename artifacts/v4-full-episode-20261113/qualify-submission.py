import json
from pathlib import Path
from tools.benchmark_v4 import run_jobs

def main():
    root=Path('artifacts/v4-full-episode-20261113')
    old=json.loads(Path('artifacts/v4/validation/development/public-66.json').read_text())
    jobs=[row['job'] for row in old['results']]
    for name,source,batch in [('submission-public-66','submission/task2_agent.py',jobs),('submission-notebook-22','submission-notebook.py',[{'label':label,'seed':42} for label in dict.fromkeys(j['label'] for j in jobs)])]:
        report=run_jobs(batch,root/(name+'.json'),source=str(root/source))
        print(json.dumps({'batch':name,'report':report}),flush=True)
if __name__=='__main__':main()
