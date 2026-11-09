"""Export authoritative sparse interpolation inputs in a process without torch."""
from pathlib import Path
import json,sys
import numpy as np
from nsrno.fullwave import load_generator

def main():
    out=Path('out/v5/time_triparam_v1');out.mkdir(parents=True,exist_ok=True)
    g=load_generator();g.bench.ultra_environment(True)
    case=g.geometry_case();fit=g.absorption_model(case)
    solver=g.bench.UltraSolver(case,case['maps'],g.ORDER,True,fit)
    src=next(p for p in solver.op.parameters if p.name=='src'); rec=solver.rec
    (out/'original_operator.c').write_text(str(solver.op))
    np.savez(out/'geometry.npz',x=case['x'],z=case['z'],xe=case['xe'],
        source_coordinates=np.array(src.coordinates.data),receiver_coordinates=np.array(rec.coordinates.data),
        origin=np.array(src.grid.origin),spacing=np.array(src.grid.spacing),weights=case['weights'])
    manifest=json.loads(Path('out/v5/manifest.json').read_text())
    meta={'shape':[len(case['z']),len(case['x'])],'face':case['face'],'pml':g.PML,'dx':case['dx'],'dt':g.DT,'nt':g.NT,'high_absorption':fit,'reference_file':manifest['reference_file']}
    (out/'geometry.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
