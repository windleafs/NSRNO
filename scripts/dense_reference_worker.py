from pathlib import Path
from argparse import ArgumentParser
import subprocess,sys,json
p=ArgumentParser(); p.add_argument('--part',type=int,required=True); part=p.parse_args().part
for a in range(11):
    if a%3!=part: continue
    subprocess.run([sys.executable,'scripts/capture_dense_reference.py','--angle',str(a)],check=True)
Path(f'out/v5/conditioned_dense_v1/reference_complete_{part}.json').write_text(json.dumps({'part':part,'complete':True}))
