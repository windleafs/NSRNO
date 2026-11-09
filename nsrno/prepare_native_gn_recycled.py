"""CPU-only preparation of one genuine deflated residual direction."""
from pathlib import Path
import argparse
import json
import os
import resource
import time
import traceback
import numpy as np
from .native_gn_recycled_state import RecycledState,digest
from .native_gn_json import atomic


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('out/v5/reconstruction_native_gn_enriched_v1_train_200'))
    p.add_argument('--out',type=Path,default=Path('out/v5/native_recycled_krylov_preparation_v1'))
    a=p.parse_args()
    if a.out.exists():raise RuntimeError('Preserve CPU preparation')
    a.out.mkdir()
    names=['prepare_native_gn_recycled.py','native_gn_recycled_state.py','native_gn_galerkin.py','native_span_qp.py',
        'native_gn_coordinates.py','native_gn_secant.py','physical_box_direction.py','native_gn_json.py']
    run={'status':'preparing_on_CPU','pid':os.getpid(),'created_unix_s':time.time(),
        'source_root':str(a.source.resolve()),'new_wave_solves':0,'GPU_work':False,
        'truth_read':False,'joint_imaging_success_claimed':False,
        'source_sha256':{n:digest(Path(__file__).parent/n) for n in names}}
    for name in names:(a.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    atomic(a.out/'run.json',run);started=time.time()
    try:
        state=RecycledState(a.source,callback=lambda item:atomic(a.out/'progress.json',item))
        direction,details=state.span.next_direction()
        # The real worker consumes float32; persist exactly those input values.
        direction=direction.astype(np.float32);np.save(a.out/'next_direction.npy',direction)
        conjugacy=state.span.images@direction.reshape(-1).astype(np.float64)
        result={'restored_restricted_solution':state.initial_report,'next_direction':details,
            'input_dtype':'float32, exact native worker input values',
            'H_conjugacy_absolute_max_after_float32_cast':float(np.max(abs(conjugacy))),
            'direction_SHA256':digest(a.out/'next_direction.npy'),'new_true_HVP_computed':False,
            'full_projected_KKT_relative':state.span.projected_norm()/state.initial_norm,
            'fixed_reference_SHA256':digest(state.origin/'old_material.npy'),
            'peak_CPU_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'seconds':time.time()-started,'joint_imaging_success_claimed':False}
        atomic(a.out/'metrics.json',result)
        run.update(status='recycled_direction_CPU_prepared',finished_unix_s=time.time(),seconds=time.time()-started)
        atomic(a.out/'run.json',run);print(json.dumps(result),flush=True)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time());atomic(a.out/'run.json',run);raise


if __name__=='__main__':main()
