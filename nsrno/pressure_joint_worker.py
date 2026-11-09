"""Complete-frequency workers for a future simultaneous pressure/material step.

No running conditional-pressure source imports this module. Full receiver
statistics reuse the required field preconditioner basis, then consume it.
Candidate backtracking uses cached fields and no pressure-factor setup.
"""
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
from .pressure_stable_cg import update_independent as update_block
from .pressure_schur import PressureSchur
from .pressure_receiver_normal import PressureReceiverNormal
from .pressure_joint_kernels import (ReceiverMetric, PressureBornColumns,
    projected_receiver_residual, joint_pressure_trial, scaled_joint_pressure_trial)
from .wri_worker import atomic


@torch.no_grad()
def run_joint_worker(rank, device, assignments, folder, step, mode, cap,
                     maximum=64, tolerance=.003, proposal_directory=None,
                     label=None, line_scale=1.):
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
        new_material = old_material
        q = None
        if proposal_directory is not None:
            new_material = torch.from_numpy(np.load(proposal_directory/'material.npy')).to(name)
            old_variables = np.load(folder/'old_variables.npy')
            proposed = np.load(proposal_directory/'variables.npy')
            q = torch.from_numpy((proposed-old_variables)/.1).to(name, dtype=torch.float32)
            del old_variables, proposed
        uniform = old_material.clone()
        uniform[0].fill_(1540)
        uniform[1].fill_(1000)
        uniform[2].fill_(.002)
        network = None
        if mode in ['field_joint', 'candidate_first']:
            network = LearnedPreconditioner().to(name)
            checkpoint = Path(run['field_inverse_checkpoint'])
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != run['field_inverse_checkpoint_sha256']:
                raise RuntimeError('Frozen field-inverse weights changed between joint phases')
            network.load_state_dict(torch.load(checkpoint,
                map_location='cpu', weights_only=True)['model'])
            network.requires_grad_(False)
            network.eval()
        banks, first, backtrack = {}, {}, {}
        for source in groups:
            banks[source] = {kind: np.load(root/'banks'/f'{source}_{kind}.npy', mmap_mode='r+')
                for kind in ['X', 'Y', 'Ydata', 'R', 'D', 'generation', 'penalty']}
            if mode in ['candidate_first', 'candidate_scaled']:
                first[source] = np.load(folder/'trial_banks'/f'{source}_first_X.npy',
                    mmap_mode='r+' if mode == 'candidate_first' else 'r')
            if mode == 'candidate_scaled':
                backtrack[source] = np.load(folder/'trial_banks'/f'{source}_line_X.npy', mmap_mode='r+')
        if mode == 'field_joint':
            normal = torch.zeros((6, g.h, g.w), device=name, dtype=torch.float64)
            gradient = torch.zeros((3, g.h, g.w), device=name, dtype=torch.float64)
        rows = []
        for number, (source, position) in enumerate(assignments):
            data = groups[source]
            frequency = data.frequency(position)
            response = receiver_response(frequency)
            bank = banks[source]
            penalty = float(bank['penalty'][position])
            if penalty <= 0 or not np.isfinite(penalty):
                raise RuntimeError('Joint continuation requires unchanged positive pressure penalties')
            torch.cuda.reset_peak_memory_stats()
            atomic(folder/f'{tag}_progress_rank{rank}.json', {'phase': mode, 'rank': rank,
                'device': device, 'pid': os.getpid(), 'step': step, 'source': source,
                'position': position, 'complete_frequencies': number,
                'assigned_frequencies': len(assignments), 'line_scale': line_scale,
                'unix_s': time.time()})
            old_system = ConditionedFrequencySystem(g, old_material[:2], old_material[2], frequency)
            oldop = PressureSchur(old_system)
            reference = torch.stack([data.reference(position, a) for a in range(11)])
            current = g.pad_field(torch.from_numpy(np.array(bank['X'][position])).to(name))
            row = {'source': source, 'position': position, 'frequency_hz': frequency,
                   'penalty': penalty, 'line_scale': line_scale}
            if mode == 'field_joint':
                refop = PressureSchur(ConditionedFrequencySystem(g, uniform[:2], uniform[2], frequency))
                target = torch.stack([data.target(position, a) for a in range(11)])
                physical_rhs = oldop.scattering_rhs(refop, reference)
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
                    raise RuntimeError('Wrong joint-field RHS generation')
                pressure, pressure_adjoint, pressure_meta = two_level_pressure(old_system,
                    learned_apply(network, old_system), learned_apply(network, old_system, adjoint=True))
                preconditioner = PressureReceiverNormal(oldop, pressure_adjoint, response, penalty)
                receiver_meta = dict(preconditioner.metadata)
                values, record, base, data_base = update_block(oldop, R, D, response, current,
                    penalty, pressure, pressure_adjoint, preconditioner,
                    maximum=maximum, tolerance=tolerance)
                row.update(block=record, pressure_preconditioner=pressure_meta,
                           receiver_preconditioner=receiver_meta)
                row['data_loss'] = .5*float((target-oldop.readout(values, response)).abs().square().sum(dtype=torch.float64))/(data.energy*len(groups))
                row['PDE_loss'] = .5*penalty*float((physical_rhs-oldop(values)).abs().square().sum(dtype=torch.float64))/(data.energy*len(groups))
                bank['X'][position] = g.native(values).cpu().numpy()
                bank['Y'][position] = g.native(base).cpu().numpy()
                bank['Ydata'][position] = data_base.cpu().numpy()
                for kind in ['X', 'Y', 'Ydata']:
                    bank[kind].flush()
                del current, base, data_base, target, physical_rhs, refop
                metric = ReceiverMetric.from_field_preconditioner(preconditioner)
                np.save(folder/'receiver_gram'/f'{source}_{position:03d}.npy', metric.gram.cpu().numpy())
                white_residual, raw_projection = projected_receiver_residual(oldop, values, R, D,
                    response, pressure, metric, return_raw=True)
                np.save(folder/'receiver_projection'/f'{source}_{position:03d}.npy', raw_projection.cpu().numpy())
                real, imag = metric.whiten_basis_in_place(preconditioner.real, preconditioner.imag)
                # The consumed basis must never be used for another CG call.
                del preconditioner
                born = PressureBornColumns(oldop, reference+values)
                blocks, gdata = born.receiver_statistics(real, imag, white_residual,
                    1/(data.energy*len(groups)))
                normal.add_(blocks)
                gradient.add_(gdata)
                row.update(curvature='same-node receiver-space blocks; off-node data blocks omitted',
                    old_field_gradient_correction_included=True,
                    curvature_GiB=(blocks.numel()+gdata.numel())*8/2**30)
                del values, metric, white_residual, raw_projection, real, imag, born, blocks, gdata
                del R, D, pressure, pressure_adjoint
            else:
                if int(bank['generation'][position]) != step:
                    raise RuntimeError('Missing joint-step immutable augmented RHS')
                R = g.pad_field(torch.from_numpy(np.array(bank['R'][position])).to(name))
                D = torch.from_numpy(np.array(bank['D'][position])).to(name)
                newop = PressureSchur(ConditionedFrequencySystem(g, new_material[:2], new_material[2], frequency))
                if mode == 'candidate_first':
                    if line_scale != 1.:
                        raise RuntimeError('The cached first GN candidate must use unit line scale')
                    pressure, pressure_adjoint, pressure_meta = two_level_pressure(old_system,
                        learned_apply(network, old_system), learned_apply(network, old_system, adjoint=True))
                    gamma = torch.from_numpy(np.load(folder/'receiver_gram'/f'{source}_{position:03d}.npy')).to(name)
                    raw_projection = torch.from_numpy(np.load(folder/'receiver_projection'/f'{source}_{position:03d}.npy')).to(name)
                    metric = ReceiverMetric(gamma, penalty)
                    trial, candidate, _, _ = joint_pressure_trial(oldop, newop, reference,
                        current, R, D, response, pressure, pressure_adjoint, metric,
                        real_increment=q, line_scale=1., projection_raw=raw_projection)
                    first[source][position] = g.native(trial).cpu().numpy()
                    first[source].flush()
                    row['pressure_preconditioner'] = pressure_meta
                    del pressure, pressure_adjoint, gamma, raw_projection, metric
                elif mode == 'candidate_scaled':
                    first_trial = g.pad_field(torch.from_numpy(np.array(first[source][position])).to(name))
                    trial, candidate, _, _ = scaled_joint_pressure_trial(oldop, newop,
                        reference, current, R, D, response, first_trial, penalty,
                        real_increment=q, line_scale=line_scale)
                    backtrack[source][position] = g.native(trial).cpu().numpy()
                    backtrack[source].flush()
                    del first_trial
                elif mode == 'dual_joint':
                    delta_reference = newop.material_change(oldop, reference)
                    bank['Y'][position] = g.native(R-delta_reference-newop(current)).cpu().numpy()
                    bank['Ydata'][position] = (D-newop.readout(current, response)).cpu().numpy()
                    bank['Y'].flush()
                    bank['Ydata'].flush()
                    del delta_reference
                else:
                    raise ValueError(mode)
                if mode in ['candidate_first', 'candidate_scaled']:
                    scale = 1/(data.energy*len(groups))
                    for key in ['data_halfcost_change', 'equation_halfcost_change',
                                'joint_halfcost_change', 'joint_linear_slope',
                                'conditional_curvature_along_material',
                                'receiver_reduced_curvature_along_material',
                                'conditional_material_linear_slope',
                                'receiver_projected_material_linear_slope']:
                        if key in candidate:
                            row[key] = candidate[key]*scale
                    row['joint_field_delta_norm'] = candidate['field_delta_norm']
                    del trial, candidate
                del current, newop, R, D
            row.update(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                       elapsed_seconds=time.time()-started)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del old_system, oldop, reference
            gc.collect()
            if number % 8 == 7:
                torch.cuda.empty_cache()
        if mode == 'field_joint':
            path = folder/f'joint_normal_rank{rank}.npz'
            np.savez(path, blocks=normal.cpu().numpy(), gradient=gradient.cpu().numpy())
        observed = [(r['source'], r['position']) for r in rows]
        if observed != assignments:
            raise RuntimeError('Incomplete full-frequency joint worker')
        summary = {'status': 'complete', 'phase': mode, 'rank': rank, 'pid': os.getpid(),
            'frequencies': len(rows), 'assignments': observed, 'seconds': time.time()-started,
            'peak_allocated_GiB': max(r['peak_allocated_GiB'] for r in rows),
            'normal_file': str(folder/f'joint_normal_rank{rank}.npz') if mode == 'field_joint' else None,
            'unconverged_shots': sum(sum(p > tolerance or q > tolerance for p, q in zip(
                r['block']['physical_global_normal_residual'], r['block']['preconditioned_global_normal_residual']))
                for r in rows if 'block' in r),
            'max_physical_normal': max((max(r['block']['physical_global_normal_residual'])
                for r in rows if 'block' in r), default=0),
            'max_preconditioned_normal': max((max(r['block']['preconditioned_global_normal_residual'])
                for r in rows if 'block' in r), default=0),
            'truth_read': False}
        for key in ['data_loss', 'PDE_loss', 'data_halfcost_change', 'equation_halfcost_change',
                    'joint_halfcost_change', 'joint_linear_slope',
                    'conditional_curvature_along_material', 'receiver_reduced_curvature_along_material',
                    'conditional_material_linear_slope', 'receiver_projected_material_linear_slope']:
            summary[key] = sum(r.get(key, 0.) for r in rows)
        atomic(folder/f'{tag}_rank{rank}.json', summary)
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        atomic(folder/f'{tag}_rank{rank}.failed.json', {'pid': os.getpid(), 'error': error})
        raise
    finally:
        log.close()
