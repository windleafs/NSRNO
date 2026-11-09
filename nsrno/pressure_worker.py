"""Separate GPU phases for scalar-pressure WRI on all acquisition channels."""
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
from .conditioned_wave import ConditionedFrequencySystem, LearnedPreconditioner
from .dense_wave import learned_apply
from .frequency_group_data import FrequencyGroupData, receiver_response
from .native_compact_weighted_pressure_gpu import two_level_pressure
from .block_wavefield_step import update_block
from .pressure_schur import PressureSchur
from .pressure_material_stencil import PressureMaterialStencil
from .pressure_receiver_normal import PressureReceiverNormal
from .wri_worker import atomic


@torch.no_grad()
def run_worker(rank, device, assignments, folder, step, mode, cap,
               maximum=64, tolerance=.003, proposal_file=None, label=None):
    tag = label or mode
    log = open(folder/f'{tag}_rank{rank}.log', 'w', buffering=1)
    sys.stdout = log
    sys.stderr = log
    started = time.time()
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(cap*2**30 /
            torch.cuda.get_device_properties(device).total_memory, device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        name = f'cuda:{device}'
        root = folder.parent
        run = json.loads((root/'run.json').read_text())
        groups = {}
        geometry = None
        for source in run['sources']:
            groups[source] = FrequencyGroupData(source, run['sample'], name, geometry)
            geometry = groups[source].geometry
        g = geometry
        old_material = torch.from_numpy(np.load(folder/'old_material.npy')).to(name)
        material = old_material if proposal_file is None else torch.from_numpy(np.load(proposal_file)).to(name)
        uniform = old_material.clone()
        uniform[0].fill_(1540)
        uniform[1].fill_(1000)
        uniform[2].fill_(.002)
        network = None
        if mode == 'field':
            network = LearnedPreconditioner().to(name)
            network.load_state_dict(torch.load('out/v5/forward_round8_dense_full/last.pt',
                map_location='cpu', weights_only=True)['model'])
            network.requires_grad_(False)
            network.eval()
        banks = {}
        for source in groups:
            banks[source] = {kind: np.load(root/'banks'/f'{source}_{kind}.npy', mmap_mode='r+')
                for kind in ['X', 'Y', 'Ydata', 'R', 'D', 'generation', 'penalty']}
        if mode == 'statistics':
            packed = torch.zeros((190, g.h, g.w), device=name, dtype=torch.float64)
            local_right = torch.zeros((19, g.h, g.w), device=name, dtype=torch.float64)
        rows = []
        for number, (source, position) in enumerate(assignments):
            data = groups[source]
            frequency = data.frequency(position)
            response = receiver_response(frequency)
            bank = banks[source]
            torch.cuda.reset_peak_memory_stats()
            atomic(folder/f'{tag}_progress_rank{rank}.json', {'phase': mode, 'rank': rank,
                'device': device, 'pid': os.getpid(), 'step': step, 'source': source,
                'position': position, 'complete_frequencies': number,
                'assigned_frequencies': len(assignments), 'unix_s': time.time()})
            system = ConditionedFrequencySystem(g, old_material[:2], old_material[2], frequency)
            op = PressureSchur(system)
            refop = PressureSchur(ConditionedFrequencySystem(g, uniform[:2], uniform[2], frequency))
            reference = torch.stack([data.reference(position, angle) for angle in range(11)])
            current = g.pad_field(torch.from_numpy(np.array(bank['X'][position])).to(name))
            target = torch.stack([data.target(position, angle) for angle in range(11)])
            if mode == 'field':
                pressure, pressure_adjoint, pressure_meta = two_level_pressure(system,
                    learned_apply(network, system), learned_apply(network, system, adjoint=True))
                saved_penalty = float(bank['penalty'][position])
                normal_pre = PressureReceiverNormal(op, pressure_adjoint, response,
                    penalty=saved_penalty if saved_penalty > 0 else None)
                penalty = normal_pre.penalty
                bank['penalty'][position] = penalty
                bank['penalty'].flush()
                generation = int(bank['generation'][position])
                if generation == step-1:
                    rhs = op.scattering_rhs(refop, reference)
                    shifted_rhs = rhs + g.pad_field(torch.from_numpy(np.array(bank['Y'][position])).to(name))
                    shifted_target = target + torch.from_numpy(np.array(bank['Ydata'][position])).to(name)
                    # Durable immutable per-frequency RHS precedes all field/base writes.
                    bank['R'][position] = g.native(shifted_rhs).cpu().numpy()
                    bank['D'][position] = shifted_target.cpu().numpy()
                    bank['R'].flush()
                    bank['D'].flush()
                    bank['generation'][position] = step
                    bank['generation'].flush()
                    del rhs
                elif generation == step:
                    shifted_rhs = g.pad_field(torch.from_numpy(np.array(bank['R'][position])).to(name))
                    shifted_target = torch.from_numpy(np.array(bank['D'][position])).to(name)
                else:
                    raise RuntimeError('Pressure RHS belongs to a different outer generation')
                values, record, base, data_base = update_block(op, shifted_rhs, shifted_target,
                    response, current, penalty, pressure, pressure_adjoint, normal_pre,
                    maximum=maximum, tolerance=tolerance)
                physical_eq = op.scattering_rhs(refop, reference) - op(values)
                physical_data = target-op.readout(values, response)
                row = {'source': source, 'position': position, 'frequency_hz': frequency,
                    'block': record, 'penalty': penalty, 'pressure_preconditioner': pressure_meta,
                    'receiver_preconditioner': normal_pre.metadata,
                    'data_loss': .5*float(physical_data.abs().square().sum(dtype=torch.float64))/(data.energy*len(groups)),
                    'PDE_loss': .5*penalty*float(physical_eq.abs().square().sum(dtype=torch.float64))/(data.energy*len(groups))}
                bank['X'][position] = g.native(values).cpu().numpy()
                bank['Y'][position] = g.native(base).cpu().numpy()
                bank['Ydata'][position] = data_base.cpu().numpy()
                for kind in ['X', 'Y', 'Ydata']:
                    bank[kind].flush()
                del values, base, data_base, record, physical_eq, physical_data
                del shifted_rhs, shifted_target, normal_pre, pressure, pressure_adjoint
            else:
                if int(bank['generation'][position]) != step:
                    raise RuntimeError('Missing immutable RHS for this full-frequency phase')
                R = g.pad_field(torch.from_numpy(np.array(bank['R'][position])).to(name))
                e = op(current)-R
                total = reference+current
                penalty = float(bank['penalty'][position])
                scale = penalty/(data.energy*len(groups))
                row = {'source': source, 'position': position, 'frequency_hz': frequency, 'penalty': penalty}
                if mode == 'statistics':
                    stencil = PressureMaterialStencil(op, total)
                    stencil.accumulate(g.native(e), scale, packed, local_right)
                    del stencil
                elif mode in ['candidate', 'dual']:
                    if proposal_file is None:
                        raise RuntimeError('A saved physical material proposal is required')
                    newop = PressureSchur(ConditionedFrequencySystem(g, material[:2], material[2], frequency))
                    change = newop.material_change(op, total)
                    if mode == 'candidate':
                        dc = change.to(torch.complex128)
                        ec = e.to(torch.complex128)
                        row['equation_cost_change'] = .5*scale*float(
                            (2*(dc.conj()*ec).real+dc.abs().square()).sum())
                        del dc, ec
                    else:
                        # Recompute from immutable R, rather than subtracting twice
                        # from partially rewritten Y after an interruption.
                        bank['Y'][position] = g.native(-e-change).cpu().numpy()
                        D = torch.from_numpy(np.array(bank['D'][position])).to(name)
                        bank['Ydata'][position] = (D-op.readout(current, response)).cpu().numpy()
                        bank['Y'].flush()
                        bank['Ydata'].flush()
                        del D
                    del newop, change
                else:
                    raise ValueError(mode)
                del R, e, total
            row.update(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                       elapsed_seconds=time.time()-started)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del system, op, refop, reference, current, target
            gc.collect()
            if number % 8 == 7:
                torch.cuda.empty_cache()
        if mode == 'statistics':
            for kind, value in [('gram', packed), ('right', local_right)]:
                # Chunked transfer avoids a duplicate host allocation of the full Gram.
                output = np.lib.format.open_memmap(folder/f'{kind}_rank{rank}.npy', mode='w+',
                    dtype=np.float64, shape=tuple(value.shape))
                for i in range(len(value)):
                    output[i] = value[i].cpu().numpy()
                output.flush()
                del output
        observed = [(r['source'], r['position']) for r in rows]
        if observed != assignments:
            raise RuntimeError('Missing or duplicated complete-frequency records')
        summary = {'status': 'complete', 'phase': mode, 'rank': rank, 'pid': os.getpid(),
            'frequencies': len(rows), 'assignments': observed, 'seconds': time.time()-started,
            'peak_allocated_GiB': max(r['peak_allocated_GiB'] for r in rows),
            'equation_cost_change': sum(r.get('equation_cost_change', 0.) for r in rows),
            'data_loss': sum(r.get('data_loss', 0.) for r in rows),
            'PDE_loss': sum(r.get('PDE_loss', 0.) for r in rows),
            'unconverged_shots': sum(sum(p > tolerance or q > tolerance for p, q in zip(
                r['block']['physical_global_normal_residual'], r['block']['preconditioned_global_normal_residual']))
                for r in rows if 'block' in r),
            'max_physical_global_normal_residual': max((max(r['block']['physical_global_normal_residual'])
                for r in rows if 'block' in r), default=0),
            'max_preconditioned_global_normal_residual': max((max(r['block']['preconditioned_global_normal_residual'])
                for r in rows if 'block' in r), default=0),
            'geometry': {'native_shape': [g.h, g.w], 'padded_shape': [g.H, g.W],
                         'offset_zx': [g.pz, g.px], 'pml': g.pml},
            'truth_read': False}
        atomic(folder/f'{tag}_rank{rank}.json', summary)
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        atomic(folder/f'{tag}_rank{rank}.failed.json', {'pid': os.getpid(), 'error': error})
        raise
    finally:
        log.close()
