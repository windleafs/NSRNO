"""Continue unchanged WRI from a COMPLETE material/dual/RF boundary.

This entry never stops another process, adopts partial fields, or reads
material truth. It defaults to one further complete outer step. The current
step3 GPU continuation remains a separate, immutable running program.
"""
from pathlib import Path
import ctypes, os
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
import argparse, fcntl, hashlib, json, shutil, time, traceback
import numpy as np
from .wri_material import MaterialVariables
from .wri_material_v2 import MaterialNormal
from .wri_worker_block_gpu import atomic
from .reconstruct_wri import Resources, score, evaluate
from .reconstruct_wri_gpu import workers


GROUPS = {'lf': 180, 'point3': 270}


def live(pid):
    path = Path('/proc') / str(pid) / 'cmdline'
    return path.exists() and bool(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root, backend, qualification):
    run = json.loads((root / 'run.json').read_text())
    if run['status'] != 'revised_step_finished':
        raise RuntimeError('Require a completed material/dual/RF outer boundary; current phase: ' + run['status'])
    step = run['completed_step']
    if step != run['step'] or step >= run['iterations']:
        raise RuntimeError('No valid next step in the existing outer-step plan')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if run['boot_id'] != boot or live(run['pid']):
        raise RuntimeError('Source coordinator must have naturally exited in the current boot')
    if list(run['sources']) != list(GROUPS) or len(run['devices']) != 3:
        raise RuntimeError('Require the unchanged two acquisition groups and three frequency workers')
    previous = root / f'step{step:03d}'
    complete = json.loads((previous / 'complete_sweep.json').read_text())
    if complete['status'] != 'complete' or complete['step'] != step or complete['solver_revision'] not in [3, 4]:
        raise RuntimeError('Require a complete GPU-era dual sweep, not Y_base or partial frequencies')
    for name in ['field_processes.json', 'field_continue_processes.json', 'dual_processes.json']:
        path = previous / name
        if path.exists() and any(live(pid) for pid in json.loads(path.read_text())['pids']):
            raise RuntimeError('A source field or dual worker is still live')
    rf = json.loads((previous / 'rf_score.json').read_text())
    if not np.isfinite(rf['primary_loss']) or rf['primary_loss'] < 0:
        raise RuntimeError('Completed native RF score missing or nonfinite')
    if (root / f'step{step+1:03d}').exists():
        raise RuntimeError('Next-step directory already exists; no automatic partial resumption')
    hashes = run['solver_changes'][-1]['source_sha256']
    for name, digest in hashes.items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise RuntimeError('Source changed since the complete boundary: ' + name)
    files = {}
    for name in ['variables.npy', 'material.npy', 'parameters.npy']:
        path = previous / name
        value = np.load(path, mmap_mode='r')
        if value.shape != (3, 961, 1121) or not np.isfinite(value).all():
            raise RuntimeError('Incomplete native material checkpoint: ' + name)
        files[name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    for source, count in GROUPS.items():
        for kind in ['X', 'Y', 'Ydata', 'penalty']:
            value = np.load(root / 'banks' / f'{source}_{kind}.npy', mmap_mode='r')
            shape = ((count, 11, 3, 961, 1121) if kind in ['X', 'Y'] else
                     (count, 11, 192) if kind == 'Ydata' else (count,))
            if value.shape != shape:
                raise RuntimeError('Wrong complete-bank shape: ' + source + '_' + kind)
            if kind != 'penalty' and value.dtype != np.complex64:
                raise RuntimeError('Unexpected field-bank dtype')
            if kind == 'penalty' and (not np.isfinite(value).all() or not (value > 0).all()):
                raise RuntimeError('Frozen original penalty must remain positive and finite')
    if backend == 'gpu':
        if qualification is None:
            raise RuntimeError('Require the completed GPU integration report')
        q = json.loads(qualification.read_text())
        if not q.get('passed') or q.get('all11_shots') is not True or q['sampled_maximum_whole_card_MiB'][str(q['device'])] > 16384:
            raise RuntimeError('GPU integration or whole-card memory report did not pass')
        for name, digest in q['code_sha256'].items():
            if sha(Path(__file__).with_name(name)) != digest:
                raise RuntimeError('Qualified GPU source changed: ' + name)
    return run, step, previous, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--gpu-qualification', type=Path)
    parser.add_argument('--cap-gib', type=float)
    parser.add_argument('--field-cg', type=int, default=64)
    parser.add_argument('--through-step', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.out.resolve()
    args.cap_gib = args.cap_gib if args.cap_gib is not None else (14.5 if args.backend == 'gpu' else 11.)
    if args.cap_gib <= 0 or args.cap_gib + .75 > 16 or args.field_cg < 1:
        raise RuntimeError('Invalid allocator or field-iteration budget')
    run, completed, previous, files = validate(root, args.backend, args.gpu_qualification)
    last = completed + 1 if args.through_step is None else args.through_step
    if not completed < last <= run['iterations']:
        raise RuntimeError('Requested steps exceed the existing complete outer-step plan')
    if args.check_only:
        print(json.dumps({'complete_checkpoint_valid': True, 'step': completed, 'through_step': last,
            'source_pid': run['pid'], 'source_exited': True, 'all450_all11': True,
            'signals_sent': False, 'bank_writes': False, 'material_files': files}), flush=True)
        return
    lock = open(root / 'gpu_coordinator.lock', 'a')
    lock_deadline = time.time()+10
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() >= lock_deadline:
                raise
            time.sleep(.1)
    monitor = None
    state_changed = False
    try:
        # Recheck under the same lock used by the step3 GPU coordinator.
        expected_step, expected_pid = completed, run['pid']
        run, completed, previous, files = validate(root, args.backend, args.gpu_qualification)
        if completed != expected_step or run['pid'] != expected_pid:
            raise RuntimeError('Complete boundary changed during preparation')
        archive = root / f'source_v4_{args.backend}_step{completed+1:03d}'
        archive.mkdir(exist_ok=False)
        names = sorted(set(run['solver_changes'][-1]['source_sha256']) | {Path(__file__).name})
        hashes = {}
        for name in names:
            path = Path(__file__).with_name(name)
            hashes[name] = sha(path)
            shutil.copy2(path, archive / name)
        shutil.copy2(root / 'run.json', archive / 'run_before_complete_resume.json')
        atomic(archive / 'complete_boundary.json', {'type': 'complete_material_dual_RF',
            'step': completed, 'source_pid': run['pid'], 'files': files,
            'field_banks': 'X and complete Y/Ydata duals, not partial Y_base',
            'all450_frequencies': True, 'all11_transmissions': True, 'truth_read': False})
        args.devices = run['devices']
        args.sample = run['sample']
        args.material_cg = run['material_CG']
        args.tv = run['TV_weight']
        args.damping = run['damping_relative']
        run.setdefault('solver_changes', []).append({'step': completed+1, 'pid': os.getpid(),
            'source_pid': run['pid'], 'unix_s': time.time(), 'source_sha256': hashes,
            'backend': args.backend, 'checkpoint_type': 'complete_material_dual_RF',
            'physical_equation_changed': False, 'observations_changed': False,
            'material_update_changed': False, 'frozen_penalties_changed': False})
        run.update(pid=os.getpid(), solver_revision=4, pressure_backend=args.backend,
            cap_GiB=args.cap_gib, fixed_rhs_directory=None, status='preparing_complete_outer_step',
            GPU_qualification=str(args.gpu_qualification) if args.backend == 'gpu' else None)
        atomic(root / 'run.json', run)
        state_changed = True
        monitor = Resources(root, resume=True)
        start = time.time()
        initial = root / 'initial'
        variables = MaterialVariables(np.load(initial / 'material.npy'), np.load(initial / 'parameters.npy'))
        current = np.load(previous / 'variables.npy')
        best = json.loads((root / 'best_RF.json').read_text())
        tasks = [(source, i) for source, count in GROUPS.items() for i in range(count)]
        assignments = [tasks[rank::3] for rank in range(3)]
        for step in range(completed+1, last+1):
            folder = root / f'step{step:03d}'
            folder.mkdir(exist_ok=False)
            (folder / 'moments').mkdir()
            shutil.copy2(previous / 'variables.npy', folder / 'old_variables.npy')
            shutil.copy2(previous / 'material.npy', folder / 'old_material.npy')
            run.update(status='field_sweep', step=step, phase_started_unix_s=time.time(),
                active_phase_cap_GiB=args.cap_gib)
            atomic(root / 'run.json', run)
            records = workers(folder, step, 'field', assignments, args, monitor)
            normal, moments = {}, []
            for record in records:
                with np.load(record['normal_path']) as values:
                    for key in values.files:
                        normal[key] = normal.get(key, 0) + values[key]
                moments.extend(record['moments'])
            observed = [(m['source'], m['position']) for m in moments]
            if len(observed) != 450 or set(observed) != set(tasks):
                raise RuntimeError('Missing or repeated full-frequency material statistics')
            atomic(folder / 'field_convergence_summary_v2.json', {'solver_revision': 4,
                'backend': args.backend, 'frequencies': 450, 'transmissions_per_frequency': 11,
                'maximum_physical_global_normal_residual': max(r['max_physical_global_normal_residual'] for r in records),
                'maximum_preconditioned_global_normal_residual': max(r['max_preconditioned_global_normal_residual'] for r in records),
                'unconverged_shots': sum(r['unconverged_shots'] for r in records),
                'requested_tolerance': .003, 'scope': 'field normal equations, not material RF accuracy'})
            run['status'] = 'material_update'
            atomic(root / 'run.json', run)
            system = MaterialNormal(variables, records[0]['geometry'], normal)
            def progress(row):
                atomic(folder / 'material_CG_progress_v4.json', row)
                if row['iteration'] % 8 == 0:
                    print(json.dumps({'step': step, 'material_CG': row}), flush=True)
            proposed, record = system.solve(current, moments, args.tv, args.damping,
                maximum=args.material_cg, callback=progress)
            np.save(folder / 'variables.npy', proposed)
            np.save(folder / 'material.npy', variables.material(proposed))
            np.save(folder / 'parameters.npy', variables.parameters(proposed))
            atomic(folder / 'material_update.json', record)
            del system, normal, records, moments
            run['status'] = 'dual_update'
            dual_args = argparse.Namespace(**vars(args))
            dual_args.cap_gib = min(args.cap_gib, 2.)
            run['active_phase_cap_GiB'] = dual_args.cap_gib
            atomic(root / 'run.json', run)
            workers(folder, step, 'dual', assignments, dual_args, monitor)
            atomic(folder / 'complete_sweep.json', {'status': 'complete', 'step': step,
                'solver_revision': 4, 'unix_s': time.time(), 'truth_read': False})
            run['status'] = 'full_native_RF_score'
            run['active_phase_cap_GiB'] = 1.1
            atomic(root / 'run.json', run)
            rf = score(folder / 'material.npy', args.sample, folder / 'rf_score.json', args.devices, 1.1, monitor)
            if rf['primary_loss'] < best['primary_loss']:
                best = {'step': step, 'material': str(folder / 'material.npy'), 'primary_loss': rf['primary_loss']}
                atomic(root / 'best_RF.json', best)
            evaluate(folder / 'material.npy', args.sample, initial / 'material.npy', monitor)
            row = {'step': step, 'material_accepted': record['accepted'], 'primary_full_RF_loss': rf['primary_loss'],
                'best_RF': best, 'solver_revision': 4, 'backend': args.backend,
                'elapsed_since_resume_seconds': time.time()-start, 'truth_read_by_optimizer': False}
            with (root / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            current, previous = proposed, folder
            run['completed_step'] = step
        run.update(status='revised_step_finished', completed_step=last,
            remaining_original_planned_steps=run['iterations']-last, best_RF=best,
            joint_reconstruction_success_claimed=False, active_phase_cap_GiB=None)
        atomic(root / 'run.json', run)
    except BaseException:
        if state_changed:
            run.update(status='failed', error=traceback.format_exc())
            atomic(root / 'run.json', run)
        raise
    finally:
        if monitor is not None:
            monitor.log.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
