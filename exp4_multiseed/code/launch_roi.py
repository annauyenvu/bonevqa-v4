from pathlib import Path
import subprocess,sys,json,os
root=Path(__file__).resolve().parents[1]
with (root/'logs/roi_preflight.log').open('a') as log:
 p=subprocess.Popen([sys.executable,str(root/'code/roi_bank.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONHASHSEED':'42'})
(root/'roi_pid.json').write_text(json.dumps({'pid':p.pid}));print('ROI preflight PID',p.pid)
