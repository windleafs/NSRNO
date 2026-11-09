"""Resume only unfinished pressure frequencies with independent batched CG."""
from pathlib import Path
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
import gc
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np
import torch
from .conditioned_wave import ConditionedFrequencySystem, LearnedPreconditioner
from .dense_wave import learned_apply
from .frequency_group_data import FrequencyGroupData, receiver_response
from .native_compact_weighted_pressure_gpu import two_level_pressure
from .pressure_schur import PressureSchur
from .pressure_receiver_normal import PressureReceiverNormal
from .pressure_stable_cg import update_independent
from .wri_worker import atomic


@torch.no_grad()
def run_stable_field(rank, device, assignments, folder, step, cap, maximum=64, tolerance=.003):
    tag = 'field_stable'
    log = open(folder/f'{tag}_rank{rank}.log', 'w', buffering=1)
    sys.stdout, sys.stderr = log, log
    start = time.time()
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(cap*2**30/torch.cuda.get_device_properties(device).total_memory, device)
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
        material = torch.from_numpy(np.load(folder/'old_material.npy')).to(name)
        uniform = material.clone()
        uniform[0].fill_(1540)
        uniform[1].fill_(1000)
        uniform[2].fill_(.002)
        checkpoint = Path(run['stable_field_checkpoint'])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != run['stable_field_checkpoint_sha256']:
            raise RuntimeError('Stable field-inverse checkpoint changed')
        network = LearnedPreconditioner().to(name)
        network.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
        network.requires_grad_(False)
        network.eval()
        banks = {source: {kind: np.load(root/'banks'/f'{source}_{kind}.npy', mmap_mode='r+')
            for kind in ['X', 'Y', 'Ydata', 'R', 'D', 'generation', 'penalty']} for source in groups}
        rows = []
        for number, (source, position) in enumerate(assignments):
            data, bank = groups[source], banks[source]
            frequency = data.frequency(position)
            response = receiver_response(frequency)
            torch.cuda.reset_peak_memory_stats()
            atomic(folder/f'{tag}_progress_rank{rank}.json', {'phase': 'field', 'rank': rank,
                'pid': os.getpid(), 'device': device, 'step': step, 'source': source,
                'position': position, 'complete_frequencies': number,
                'assigned_frequencies': len(assignments), 'unix_s': time.time()})
            system = ConditionedFrequencySystem(g, material[:2], material[2], frequency)
            op = PressureSchur(system)
            refop = PressureSchur(ConditionedFrequencySystem(g, uniform[:2], uniform[2], frequency))
            reference = torch.stack([data.reference(position, a) for a in range(11)])
            current = g.pad_field(torch.from_numpy(np.array(bank['X'][position])).to(name))
            target = torch.stack([data.target(position, a) for a in range(11)])
            physical_rhs = op.scattering_rhs(refop, reference)
            pressure, pressure_adjoint, pressure_meta = two_level_pressure(system,
                learned_apply(network, system), learned_apply(network, system, adjoint=True))
            old_penalty = float(bank['penalty'][position])
            preconditioner = PressureReceiverNormal(op, pressure_adjoint, response,
                old_penalty if old_penalty > 0 else None)
            penalty = preconditioner.penalty
            bank['penalty'][position] = penalty
            bank['penalty'].flush()
            generation = int(bank['generation'][position])
            if generation == step-1:
                R = physical_rhs+g.pad_field(torch.from_numpy(np.array(bank['Y'][position])).to(name))
                D = target+torch.from_numpy(np.array(bank['Ydata'][position])).to(name)
                bank['R'][position] = g.native(R).cpu().numpy()
                bank['D'][position] = D.cpu().numpy()
                bank['R'].flush()
                bank['D'].flush()
                bank['generation'][position] = step
                bank['generation'].flush()
            elif generation == step:
                R = g.pad_field(torch.from_numpy(np.array(bank['R'][position])).to(name))
                D = torch.from_numpy(np.array(bank['D'][position])).to(name)
            else:
                raise RuntimeError('Wrong immutable RHS generation in stable continuation')
            values, record, base, data_base = update_independent(op, R, D, response,
                current, penalty, pressure, pressure_adjoint, preconditioner,
                maximum=maximum, tolerance=tolerance)
            row = {'source': source, 'position': position, 'frequency_hz': frequency,
                'block': record, 'penalty': penalty, 'pressure_preconditioner': pressure_meta,
                'receiver_preconditioner': preconditioner.metadata,
                'data_loss': .5*float((target-op.readout(values, response)).abs().square().sum(dtype=torch.float64))/(data.energy*len(groups)),
                'PDE_loss': .5*penalty*float((physical_rhs-op(values)).abs().square().sum(dtype=torch.float64))/(data.energy*len(groups))}
            bank['X'][position] = g.native(values).cpu().numpy()
            bank['Y'][position] = g.native(base).cpu().numpy()
            bank['Ydata'][position] = data_base.cpu().numpy()
            for kind in ['X', 'Y', 'Ydata']:
                bank[kind].flush()
            row.update(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                       elapsed_seconds=time.time()-start)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del system, op, refop, reference, current, target, physical_rhs
            del pressure, pressure_adjoint, preconditioner, R, D, values, record, base, data_base
            gc.collect()
            if number % 8 == 7:
                torch.cuda.empty_cache()
        atomic(folder/f'{tag}_rank{rank}.json', {'status': 'complete', 'rank': rank,
            'pid': os.getpid(), 'frequencies': len(rows), 'assignments': assignments,
            'seconds': time.time()-start, 'truth_read': False})
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        atomic(folder/f'{tag}_rank{rank}.failed.json', {'pid': os.getpid(), 'error': error})
        raise
    finally:
        log.close()
