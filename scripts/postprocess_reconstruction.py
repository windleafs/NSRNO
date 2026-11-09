from pathlib import Path
from argparse import ArgumentParser
import json,os,subprocess,sys,time
root=Path('/home_nvme/zhuangyang/nsrno'); os.chdir(root)
parser=ArgumentParser(); parser.add_argument('--out',type=Path,default=root/'out/v5/reconstruction_round1_train_200')
out=parser.parse_args().out
started=time.monotonic()
while not (out/'completion.json').exists():
    if time.monotonic()-started>1800: raise TimeoutError('Reconstruction incomplete after30 minutes')
    time.sleep(10)
for script in ('scripts/plot_reconstruction.py','scripts/evaluate_reconstruction_physics.py'):
    subprocess.run([sys.executable,script,'--out',str(out)],check=True)
print('Physical comparison and material figures complete.',flush=True)
