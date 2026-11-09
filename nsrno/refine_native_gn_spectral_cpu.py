"""Prepare an actual next GN proposal on CPU while RF acceptance finishes."""
from pathlib import Path
import argparse
import json
import os
import time
import traceback
import numpy as np
from .native_gn_spectral_state import SpectralSavedQuadratic
from .native_gn_json import atomic
from .refine_native_gn_cpu import solve_accurately,save_array
from .native_gn_coordinates import dot


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--symbol-snapshot',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--maximum',type=int,default=1200)
    p.add_argument('--tolerance',type=float,default=.01)
    args=p.parse_args()
    if args.out.exists():raise RuntimeError('Inspect existing CPU proposal; no implicit restart')
    args.out.mkdir(parents=True)
    run={'status':'restoring_spectral_multisecant_metric','pid':os.getpid(),
        'source_root':str(args.source_root.resolve()),'created_unix_s':time.time(),
        'GPU_used':False,'patient_wave_solves':0,'truth_read':False}
    atomic(args.out/'run.json',run)
    try:
        source=json.loads((args.source_root/'run.json').read_text())
        pids=[source['pid']]+json.loads((args.source_root/'worker_processes.json').read_text())['worker_pids']
        if any((Path('/proc')/str(p)).exists() for p in pids):
            raise RuntimeError('Source writer remains alive')
        state=SpectralSavedQuadratic(args.source_root,1,args.symbol_snapshot)
        difference=state.model.apply(state.u)-state.Hu
        defect=np.sqrt(dot(difference,difference)/max(dot(state.Hu,state.Hu),1e-300))
        if defect>2e-5:raise RuntimeError(f'Saved cumulative curvature not retained: {defect}')
        atomic(args.out/'restored.json',{'fixed_QP_value':state.value(),
            'true_QP_KKT_relative':state.norm()/state.initial_norm,
            'complete_HVP_count':len(state.paths),'known_state_model_Hu_relative_error':defect,
            'metric':state.spectral_report,'GPU_used':False,'new_wave_solves':0})
        run['status']='CPU_bound_constrained_spectral_proposal'
        atomic(args.out/'run.json',run)
        def progress(row,d,save):
            atomic(args.out/'progress.json',row)
            if save:save_array(args.out/'increment_checkpoint.npy',d)
        d,report=solve_accurately(state.model,state.gradient,state.low-state.u,
            state.high-state.u,np.zeros_like(state.u),maximum=args.maximum,
            tolerance=args.tolerance,callback=progress)
        save_array(args.out/'increment.npy',d)
        save_array(args.out/'total_candidate_log_direction.npy',state.u+d)
        report.update(fixed_QP_source_value=state.value(),reused_native_HVPs=len(state.paths),
            native_HVP_for_candidate_performed=False,accepted_material=False)
        atomic(args.out/'result.json',report)
        run.update(status='CPU_proposal_ready' if report['tolerance_met'] else 'CPU_budget_reached',
            tolerance_met=report['tolerance_met'],finished_unix_s=time.time())
        atomic(args.out/'run.json',run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),finished_unix_s=time.time())
        atomic(args.out/'run.json',run)
        raise


if __name__=='__main__':main()
