"""One 0.5 MHz paired V5 case: 22 known-water + 22 patient shots, no torch.

Known-water RF is stored after the declared linear receiving operator, not
as a 580 MiB native-pressure archive. Patient RF subtracts that same reference.
"""
from pathlib import Path
import argparse
import ctypes
import gc
import os
import signal
import subprocess
import sys
import time
import traceback
if os.name == 'posix':
    if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
        raise OSError('Cannot disable process transparent huge pages')
import numpy as np
from .fullwave import initialize
from .generate_point_tx import source_layout
from .v5_materials import evaluation_material, material_paths
from .low_frequency_0p5 import (
    F0, NATIVE_NT, NATIVE_DT, AMPLITUDE, SAMPLE, CASE, atomic,
    digest, processed_coefficients, save_coefficients, acquisition_metadata)


def gpu_processes():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory',
         '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in result.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            rows.append({'pid': int(fields[0]), 'used_MiB': int(fields[1])})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--after-run', type=Path, required=True)
    p.add_argument('--max-seconds', type=float, default=1200)
    a = p.parse_args()
    if 'torch' in sys.modules or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Original generator must be torch-free and use physical GPU 0')
    preceding = __import__('json').loads((a.after_run / 'run.json').read_text())
    if preceding.get('status') != 'conditional_secant_training_finished':
        raise RuntimeError('Finish and save the preceding neural model first')
    if Path('/proc', str(preceding['pid'])).exists() or gpu_processes():
        raise RuntimeError('Another GPU process is still live; do not duplicate work')
    if a.out.exists():
        raise RuntimeError('Inspect existing paired shots; never replay or overwrite implicitly')
    import shutil
    # Compact double DFT reference and observations require under 16 MiB.
    reserve = 2 * 2**30 + 400 * 2**20
    if shutil.disk_usage('.').free < reserve + 20 * 2**20:
        raise RuntimeError('Insufficient reserve for the compact paired acquisition')
    if not 0 < a.max_seconds <= 1200:
        raise ValueError('This generation phase has a 1200 s maximum')
    a.out.mkdir()
    started = time.time()
    fit_path = Path('out/v5/low_frequency_3mhz_v1/index.json')
    old = __import__('json').loads(fit_path.read_text())
    point_path = Path('out/v5/point_tx_3mhz_v1/index.json')
    point = __import__('json').loads(point_path.read_text())
    if point['absorption'] != old['absorption']:
        raise ValueError('Existing 3 MHz media use different constitutive fits')
    angles = old['acquisition']['angles_deg']
    if len(angles) != 11:
        raise ValueError('All original 11 angles are required')
    plan_path = Path('out/v5/independent_material_3mhz_v1/material_plan.json')
    material_record = next(r for r in __import__('json').loads(plan_path.read_text())['records']
                           if r['id'] == 'coverage_dev_train_200')
    if material_record['case'] != CASE:
        raise ValueError('Only the authorized current development case is admitted')
    paths = material_paths(SAMPLE)
    state = {
        'status': 'initializing_paired_0p5MHz_generation',
        'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': started + a.max_seconds,
        'maximum_wave_solves': 44, 'completed_wave_solves': 0,
        'completed_reference_shots': 0, 'completed_patient_shots': 0,
        'new_material_derivatives': 0, 'only_new_centre_frequency_hz': F0,
        'sample': SAMPLE, 'case': CASE, 'new_training_cases': 0,
        'reserved_case_unread': 'Neg_47_Left',
        'acquisition': acquisition_metadata(), 'angles_deg': angles,
        'absorption': old['absorption'], 'absorption_index_SHA256': digest(fit_path),
        'material_input_SHA256': {str(v): digest(v) for v in paths},
        'material_truth_used_only_by_generator': True,
        'preceding_weights_SHA256': preceding['weights_SHA256'],
        'source_SHA256': {n: digest(Path(__file__).parent / n) for n in [
            Path(__file__).name, 'low_frequency_0p5.py', 'fullwave.py',
            'generate_point_tx.py', 'v5_materials.py']},
        'per_card_limit_GiB': 16, 'sampled_peak_owned_MiB': 0,
        'memory_sampled_after_allocation_and_each_shot': True,
        'native_pressure_archives_created': False,
        'RF_reference_subtraction': 'same linear operator applied before subtraction',
    }
    atomic(a.out / 'run.json', state)
    interrupted = [0]
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda value, frame: interrupted.__setitem__(0, value))

    def boundary():
        if interrupted[0]:
            raise InterruptedError('Stop requested; keep completed paired shots')
        if time.time() >= state['deadline_unix_s']:
            raise TimeoutError('Generation time budget exhausted; keep completed paired shots')

    def resources(phase):
        rows = gpu_processes()
        owned = sum(r['used_MiB'] for r in rows if r['pid'] == os.getpid())
        state['sampled_peak_owned_MiB'] = max(state['sampled_peak_owned_MiB'], owned)
        resource = {'unix_s': time.time(), 'phase': phase, 'owned_MiB': owned,
                    'GPU_processes': rows, 'disk_free_GiB': shutil.disk_usage('.').free / 2**30}
        with (a.out / 'resources.jsonl').open('a') as f:
            f.write(__import__('json').dumps(resource) + '\n')
        if owned > 16384 or 'torch' in sys.modules:
            raise RuntimeError('Original-generator resource boundary exceeded')
        if shutil.disk_usage('.').free < reserve:
            raise RuntimeError('Disk reserve exhausted')
        atomic(a.out / 'run.json', state)

    try:
        material = evaluation_material(SAMPLE)
        expected = np.load(material_record['material3'], mmap_mode='r')
        if digest(material_record['material3']) != material_record['material_file_SHA256']:
            raise ValueError('Original paired material changed')
        if not np.array_equal(material, expected):
            raise ValueError('0.5 MHz must use exactly the existing 3 MHz c/rho/alpha arrays')
        del expected
        g = initialize()
        if 'torch' in sys.modules:
            raise RuntimeError('Original runtime imported torch')
        state['original_generator_SHA256'] = digest(Path(g.__file__))
        base_case = g.geometry_case()
        cols, elements, positions, point_weights = source_layout(base_case)
        old_layout = np.load(point['source_layout'])
        if not np.array_equal(point_weights, old_layout['source_weights']):
            raise ValueError('Point sources must retain their original positions and aperture')
        layout = a.out / 'source_layout.npz'
        np.savez(layout, source_columns=cols, element_indices=elements,
                 point_transmit_x_m=positions, point_weights=point_weights,
                 receiver_x_m=base_case['xe'], plane_angles_deg=np.asarray(angles))
        state['source_layout_SHA256'] = digest(layout)
        state['point_transmit_x_m'] = positions.tolist()
        state['status'] = 'generating_paired_0p5MHz'
        rows = []

        def case_for(kind):
            maps = None if kind == 'reference' else {
                'sound_speed': np.ascontiguousarray(material[0]),
                'density': np.ascontiguousarray(material[1]),
                'alpha_coeff': np.ascontiguousarray(material[2]),
                'BonA': np.zeros_like(material[2])}
            c = g.geometry_case(maps)
            c.update(f0=F0, nt=NATIVE_NT, dt=NATIVE_DT, end=90e-6, tref=4/F0)
            return c

        for family in ['lf', 'point3']:
            folder = a.out / family
            folder.mkdir()
            reference_hashes = {}
            source_hashes = {}
            for kind in ['reference', SAMPLE]:
                boundary()
                c = case_for(kind)
                solver = g.solver_for(c, c['maps'], old['absorption'])
                resources(family + '_' + kind + '_allocated')
                for shot in range(11):
                    boundary()
                    if family == 'lf':
                        tref = g.set_angle(solver, c, angles[shot])
                    else:
                        c.update(weights=point_weights[shot], delays=np.zeros(len(cols)), tref=4/F0)
                        _, _, wave = g.bench.source_arrays(c)
                        solver.source.data[:] = wave.T
                        tref = 4/F0
                    solver.source.data[:] *= np.float32(AMPLITUDE / 1e5)
                    wave = np.asarray(solver.source.data)
                    source_hash = __import__('hashlib').sha256(np.ascontiguousarray(wave).tobytes()).hexdigest()
                    source_energy = float(np.square(wave.astype(np.float64)).sum() * NATIVE_DT)
                    if kind == 'reference':
                        source_hashes[shot] = source_hash
                    elif source_hash != source_hashes[shot]:
                        raise ValueError('Material-dependent source detected')
                    atomic(a.out / 'progress.json', {
                        'phase': kind, 'family': family, 'shot': shot,
                        'state': 'propagating', 'completed_wave_solves': state['completed_wave_solves'],
                        'unix_s': time.time()})
                    raw, timing = solver.run()
                    coefficients = processed_coefficients(raw)
                    if kind == 'reference':
                        output = folder / f'reference_{shot:02d}.npy'
                    else:
                        reference_path = folder / f'reference_{shot:02d}.npy'
                        if digest(reference_path) != reference_hashes[shot]:
                            raise ValueError('Protected reference changed')
                        coefficients -= np.load(reference_path)
                        output = folder / f'{SAMPLE}_{shot:02d}.npy'
                    output_hash = save_coefficients(output, coefficients)
                    if kind == 'reference':
                        reference_hashes[shot] = output_hash
                        state['completed_reference_shots'] += 1
                    else:
                        state['completed_patient_shots'] += 1
                    state['completed_wave_solves'] += 1
                    row = {'family': family, 'kind': kind, 'shot': shot,
                           'path': str(output.resolve()), 'SHA256': output_hash,
                           'source_SHA256': source_hash,
                           'source_tref_s': float(tref),
                           'discrete_source_time_energy_Pa2_s_sum_over_source_points': source_energy,
                           'seconds': float(timing['solve_readback_s'])}
                    rows.append(row)
                    atomic(folder / f'{kind}_{shot:02d}.json', row)
                    atomic(a.out / 'completed_shots.json', {'rows': rows})
                    resources(family + '_' + kind + f'_{shot:02d}_completed')
                    print(__import__('json').dumps({
                        'phase': kind, 'family': family, 'shot': shot,
                        'completed_wave_solves': state['completed_wave_solves'],
                        'seconds': row['seconds'],
                        'sampled_peak_owned_MiB': state['sampled_peak_owned_MiB']}), flush=True)
                    del raw, coefficients
                del solver
                gc.collect()
        if state['completed_wave_solves'] != 44:
            raise RuntimeError('Incomplete paired acquisition')
        state.update(status='paired_0p5MHz_generation_finished',
                     finished_unix_s=time.time(), seconds=time.time() - started)
        atomic(a.out / 'index.json', {**state, 'shots': rows,
               'optimizer_may_read_truth_arrays': False,
               'patient_target_RF_is_scattered_pressure': True})
        atomic(a.out / 'run.json', state)
        print(__import__('json').dumps({
            'status': state['status'], 'sample': SAMPLE, 'completed_wave_solves': 44,
            'seconds': state['seconds'], 'sampled_peak_owned_MiB': state['sampled_peak_owned_MiB']}), flush=True)
    except BaseException:
        state.update(status='paired_0p5MHz_generation_interrupted_or_failed',
                     error=traceback.format_exc(), stopped_unix_s=time.time())
        atomic(a.out / 'run.json', state)
        raise


if __name__ == '__main__':
    main()
