"""New solver entry point reusing immutable acquisition and RF line search.

Only this process selects the compact-QP direction implementation. The old
solver files and already-running process are never edited or altered.
"""
from pathlib import Path
import argparse
import hashlib
import json
from . import reconstruct_native_gn as framework
from .native_gn_secant import bounded_direction

def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--out',type=Path,required=True)
    args,_=p.parse_known_args()
    original_manifest=framework.source_manifest
    def manifest():
        out=original_manifest()
        for name in ['native_gn_secant.py','reconstruct_native_gn_secant.py']:
            out[name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        return out
    def direction(coordinates,gradient,hvp,blocks,**kwargs):
        run=json.loads((args.out/'run.json').read_text())
        folder=args.out/f'step{run["step"]:03d}'
        (args.out/'solver.json').write_text(json.dumps({
            'implementation':'native_gn_secant.bounded_direction',
            'framework':'immutable reconstruct_native_gn acquisition/gradient/line-search/map implementation',
            'nonlocal_curvature':'compact BFGS proposals and exact native HVP-certified steps',
            'curvature_vectors':'each step/hvpNN/direction.npy and data_image.npy',
            'truth_read_by_optimizer':False},indent=2)+'\n')
        return bounded_direction(coordinates,gradient,hvp,blocks,
            checkpoint_folder=folder,**kwargs)
    framework.source_manifest=manifest
    framework.bounded_direction=direction
    framework.main()

if __name__=='__main__':main()
