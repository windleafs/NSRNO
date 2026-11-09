"""Cheap cached-direction continuation after an internal 2.5 GiB-cap failure.

No learned inverse or sparse pressure factor is imported or built here.
This worker reuses saved fields and applies physical stencils without a new
wave solve. Original computation logs and the first cache remain immutable.
"""
from pathlib import Path
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
import gc
import json
import os
import sys
import time
import traceback
import numpy as np
import torch
from .conditioned_wave import ConditionedFrequencySystem
from .frequency_group_data import FrequencyGroupData, receiver_response
from .pressure_schur import PressureSchur
from .pressure_joint_recovery_kernels import scaled_joint_pressure_trial
from .wri_worker import atomic

COST_KEYS = ['data_halfcost_change', 'equation_halfcost_change',
    'joint_halfcost_change', 'joint_linear_slope', 'actual_joint_linear_curvature',
    'actual_linear_data_curvature', 'actual_linear_equation_curvature',
    'nonlinear_equation_remainder_halfnorm', 'linearized_joint_halfcost_change',
    'actual_conditional_material_curvature']

def original_rows(folder):
    rows = {}
    for path in folder.glob('joint_candidate000_rank*.log'):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if 'source' in row and 'position' in row:
                key = (row['source'], row['position'])
                if key in rows:
                    raise RuntimeError('Repeated original first-candidate frequency')
                rows[key] = row
    return rows

@torch.no_grad()
def run_recovery_worker(rank, device, assignments, folder, step, mode, cap,
                        maximum=64, tolerance=.003, proposal_directory=None,
                        label=None, line_scale=1.):
    if mode not in ['candidate_scaled', 'dual_joint']:
        raise ValueError('Recovery may not rebuild any pressure inverse')
    if proposal_directory is None:
        raise ValueError('Require a complete material proposal')
    old_stdout, old_stderr = sys.stdout, sys.stderr
    tag = label or mode
    log = open(folder/f'{tag}_rank{rank}.log', 'w', buffering=1)
    sys.stdout, sys.stderr = log, log
    started = time.time()
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(cap*2**30/
            torch.cuda.get_device_properties(device).total_memory, device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        name = f'cuda:{device}'
        root = folder.parent
        run = json.loads((root/'run.json').read_text())
        groups, geometry = {}, None
        for source in run['sources']:
            groups[source] = FrequencyGroupData(source, run['sample'], name, geometry)
            geometry = groups[source].geometry
        g = geometry
        old_material = torch.from_numpy(np.load(folder/'old_material.npy')).to(name)
        new_material = torch.from_numpy(np.load(proposal_directory/'material.npy')).to(name)
        q = torch.from_numpy((np.load(proposal_directory/'variables.npy')-
            np.load(folder/'old_variables.npy'))/.1).to(name, dtype=torch.float32)
        readmode = 'r+'
        banks = {s: {kind: np.load(root/'banks'/f'{s}_{kind}.npy', mmap_mode=readmode)
                     for kind in ['X','Y','Ydata','R','D','generation','penalty']}
                 for s in groups}
        first = ({s: np.load(folder/'trial_banks'/f'{s}_first_X.npy', mmap_mode='r')
                  for s in groups} if mode != 'dual_joint' else {})
        line = ({s: np.load(folder/'trial_banks'/f'{s}_line_X.npy', mmap_mode='r+')
                 for s in groups} if mode == 'candidate_scaled' else {})
        rows = []
        for number, (source, position) in enumerate(assignments):
            data, bank = groups[source], banks[source]
            frequency = data.frequency(position)
            response = receiver_response(frequency)
            penalty = float(bank['penalty'][position])
            if penalty <= 0 or not np.isfinite(penalty) or int(bank['generation'][position]) != step:
                raise RuntimeError('Wrong frozen penalty or immutable RHS generation')
            torch.cuda.reset_peak_memory_stats()
            atomic(folder/f'{tag}_progress_rank{rank}.json',
                {'phase':mode,'rank':rank,'device':device,'pid':os.getpid(),
                 'step':step,'source':source,'position':position,
                 'complete_frequencies':number,'assigned_frequencies':len(assignments),
                 'line_scale':line_scale,'unix_s':time.time()})
            system = ConditionedFrequencySystem(g, old_material[:2], old_material[2], frequency)
            newsystem = ConditionedFrequencySystem(g, new_material[:2], new_material[2], frequency)
            oldop, newop = PressureSchur(system), PressureSchur(newsystem)
            reference = torch.stack([data.reference(position, a) for a in range(11)])
            current = g.pad_field(torch.from_numpy(np.array(bank['X'][position])).to(name))
            R = g.pad_field(torch.from_numpy(np.array(bank['R'][position])).to(name))
            D = torch.from_numpy(np.array(bank['D'][position])).to(name)
            row = {'source':source,'position':position,'frequency_hz':frequency,
                   'penalty':penalty,'line_scale':line_scale}
            if mode == 'dual_joint':
                delta_reference = newop.material_change(oldop, reference)
                bank['Y'][position] = g.native(R-delta_reference-newop(current)).cpu().numpy()
                bank['Ydata'][position] = (D-newop.readout(current,response)).cpu().numpy()
                bank['Y'].flush()
                bank['Ydata'].flush()
                del delta_reference
            else:
                cached = g.pad_field(torch.from_numpy(np.array(first[source][position])).to(name))
                if not bool(torch.isfinite(cached).all()):
                    raise RuntimeError('Nonfinite cached pressure: '+str((source,position)))
                trial, cost, y, yd = scaled_joint_pressure_trial(oldop,newop,reference,
                    current,R,D,response,cached,penalty,real_increment=q,
                    line_scale=line_scale)
                scale = 1/(data.energy*len(groups))
                for key in COST_KEYS:
                    row[key] = cost[key]*scale
                row['joint_field_delta_norm'] = cost['field_delta_norm']
                line[source][position] = g.native(trial).cpu().numpy()
                line[source].flush()
                del cached,trial,cost,y,yd
            row.update(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                elapsed_seconds=time.time()-started)
            rows.append(row)
            print(json.dumps(row),flush=True)
            del system,newsystem,oldop,newop,reference,current,R,D
            gc.collect()
            if number % 8 == 7:
                torch.cuda.empty_cache()
        observed = [(r['source'],r['position']) for r in rows]
        if observed != assignments:
            raise RuntimeError('Recovery must include every assigned frequency')
        torch.cuda.synchronize()
        summary = {'status':'complete','phase':mode,'rank':rank,'pid':os.getpid(),
            'frequencies':len(rows),'assignments':observed,
            'seconds':time.time()-started,
            'peak_allocated_GiB':max(r['peak_allocated_GiB'] for r in rows),
            'bank_writes':True,'extra_pressure_solves':0,'truth_read':False}
        for key in COST_KEYS:
            summary[key] = sum(r.get(key,0.) for r in rows)
        del groups,old_material,new_material,q
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        atomic(folder/f'{tag}_rank{rank}.json',summary)
    except BaseException:
        error = traceback.format_exc()
        print(error,flush=True)
        atomic(folder/f'{tag}_rank{rank}.failed.json',{'pid':os.getpid(),'error':error})
        raise
    finally:
        sys.stdout,sys.stderr = old_stdout,old_stderr
        log.flush()
        log.close()
