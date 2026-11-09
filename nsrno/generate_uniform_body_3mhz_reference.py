"""Known uniform-body 3 MHz background, matched to existing UltraWave targets.

This is one reference material (11 shots), not another patient dataset.
Preserves the exact old total-water subtraction and separate 60/90 us operators.
Never starts while the named neural experiment or any GPU process is live.
"""
from pathlib import Path
import argparse
import ctypes
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
if os.name == 'posix':
    if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
        raise OSError('Cannot disable process transparent huge pages')
import numpy as np
from .fullwave import initialize
from .generate_independent_material_3mhz import digest, atomic, chunk_hash
from .prepare_conditional_secants import roi, envelope_from_processed_RF


def gpu_processes():
    text = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
         '--format=csv,noheader,nounits'], text=True)
    result = []
    for line in text.splitlines():
        pid, uuid, used = [part.strip() for part in line.split(',')]
        if used.isdigit():
            result.append({'pid': int(pid), 'gpu_uuid': uuid, 'used_MiB': int(used)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('out/v5/independent_material_3mhz_v1'))
    parser.add_argument('--after-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if 'torch' in sys.modules:
        raise RuntimeError('Original UltraWave process must not import torch')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Only physical GPU 0 is admitted')
    preceding = json.loads((args.after_run / 'run.json').read_text())
    if preceding['status'] != 'conditional_secant_training_finished':
        raise RuntimeError('Finish the existing neural experiment first')
    if Path('/proc', str(preceding['pid'])).exists() or gpu_processes():
        raise RuntimeError('Another GPU process is still live; no concurrent experiment')
    if args.out.exists():
        raise RuntimeError('Inspect existing ledger; never overwrite or replay implicitly')
    generated = json.loads((args.root / 'generated_index.json').read_text())
    plan = json.loads((args.root / 'material_plan.json').read_text())
    if generated['status'] != 'complete' or plan['status'] != 'materials_prepared':
        raise RuntimeError('Require the original completed, fixed acquisition')
    reference_path = args.root / 'reference_native.npy'
    expected_reference = {r['reference_SHA256'] for r in generated['records']}
    if len(expected_reference) != 1 or digest(reference_path) not in expected_reference:
        raise RuntimeError('Protected old total-water RF reference changed')
    reference = np.load(reference_path, mmap_mode='r')
    acquisition = plan['acquisition']
    angles = acquisition['angles_deg']
    if reference.shape != (11, 36001, 192) or len(angles) != 11:
        raise ValueError('Original full-duration LF acquisition is required')
    if acquisition['native_dt_s'] != 2.5e-9 or acquisition['native_nt'] != 36001:
        raise ValueError('Original native acquisition clock changed')
    low_path = Path('out/v5/low_frequency_3mhz_v1/index.json')
    fit = json.loads(low_path.read_text())['absorption']
    edge_path = Path('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    edge = np.broadcast_to(np.load(edge_path), (961, 1121))
    if not np.isfinite(edge).all() or edge.min() < 0:
        raise ValueError('Invalid known boundary attenuation')
    # Reserve the same 2 GiB + 400 MiB floor, plus all new small arrays.
    if shutil.disk_usage('.').free < 2 * 2**30 + 400 * 2**20 + 40 * 2**20:
        raise RuntimeError('Insufficient disk reserve for this compact reference')
    args.out.mkdir()
    started = time.time()
    state = {
        'status': 'starting_known_uniform_reference', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'completed_shots': 0, 'maximum_new_wave_solves': 11,
        'maximum_walltime_seconds': 600, 'new_material_derivatives': 0,
        'only_new_centre_frequency_hz': 3e6, 'all_angles': 11, 'receivers': 192,
        'known_material': {'sound_speed_m_s': 1540., 'density_kg_m3': 1000.,
                           'body_alpha': .5, 'coupling_alpha': .002,
                           'body_start_depth_m': .001},
        'material_truth_read': False, 'patient_target_RF_read': False,
        'reserved_case_unread': 'Neg_47_Left',
        'water_reference_SHA256': next(iter(expected_reference)),
        'material_plan_SHA256': digest(args.root / 'material_plan.json'),
        'absorption_index_SHA256': digest(low_path),
        'known_edge_SHA256': digest(edge_path),
        'after_run': str(args.after_run.resolve()),
        'preceding_weights_SHA256': preceding['weights_SHA256'],
        'source_SHA256': {name: digest(Path(__file__).parent / name) for name in [
            Path(__file__).name, 'fullwave.py', 'generate_independent_material_3mhz.py',
            'prepare_conditional_secants.py', 'rf_representation.py']},
        'per_card_limit_GiB': 16, 'sampled_owned_peak_MiB': 0,
        'memory_sampling': 'after solver allocation, before and after every actual shot',
        'no_full_native_RF_output': True,
        'no_new_patient_material_or_perturbation': True,
        'same_total_water_reference_as_existing_training_targets': True,
        'short_record_processed_from_native_prefix_not_cropped_long_RF': True,
    }
    atomic(args.out / 'run.json', state)

    def resources(phase):
        rows = gpu_processes()
        owned = sum(row['used_MiB'] for row in rows if row['pid'] == os.getpid())
        state['sampled_owned_peak_MiB'] = max(state['sampled_owned_peak_MiB'], owned)
        row = {'unix_s': time.time(), 'phase': phase, 'processes': rows,
               'owned_MiB': owned, 'disk_free_GiB': shutil.disk_usage('.').free / 2**30}
        with (args.out / 'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        if owned > 16384:
            raise RuntimeError('Sampled original-solver GPU memory exceeds 16 GiB')
        if 'torch' in sys.modules:
            raise RuntimeError('Torch imported into the original solver process')

    long_temp = args.out / 'RF90_envelope.partial.npy'
    short_temp = args.out / 'RF60_envelope.partial.npy'
    try:
        g = initialize()
        state['original_generator_SHA256'] = digest(Path(g.__file__))
        geometry = g.geometry_case()
        if len(geometry['z']) != 961 or len(geometry['x']) != 1121:
            raise ValueError('Original native material grid changed')
        material = np.empty((3, 961, 1121), np.float32)
        material[0] = 1540.; material[1] = 1000.; material[2] = .002 + edge
        body = np.asarray(geometry['z']) >= .001 - 1e-12
        material[2, body] = .5 + edge[body]
        native_path = args.out / 'known_uniform_material.npy'
        np.save(native_path, material)
        context = material[:, ::4, ::4].copy(); context[2] -= edge[::4, ::4]
        baseline_roi = roi(material); baseline_roi[2] -= roi(
            np.broadcast_to(edge, (3, 961, 1121)))[2]
        baseline_roi[:, :5] = np.asarray([1540., 1000., .002], np.float32)[:, None, None]
        np.save(args.out / 'known_uniform_context.npy', context)
        np.save(args.out / 'known_uniform_ROI.npy', baseline_roi)
        state['known_material_SHA256'] = digest(native_path)
        maps = {'sound_speed': material[0].copy(), 'density': material[1].copy(),
                'alpha_coeff': material[2].copy(), 'BonA': np.zeros_like(material[2])}
        case = g.geometry_case(maps)
        case.update(f0=3e6, nt=36001, end=90e-6, tref=4/3e6)
        solver = g.solver_for(case, case['maps'], fit)
        resources('original_solver_allocated')
        long = np.lib.format.open_memmap(long_temp, mode='w+', dtype=np.float32,
                                        shape=(11, 2, 192, 541))
        short = np.lib.format.open_memmap(short_temp, mode='w+', dtype=np.float32,
                                         shape=(11, 2, 192, 361))
        shots = []; clock_metadata = {}
        state['status'] = 'generating_known_uniform_reference'
        atomic(args.out / 'run.json', state)
        for shot, angle in enumerate(angles):
            if time.time() - started > 600:
                raise RuntimeError('Known-reference generation reached its fixed time cap')
            if digest(__file__) != state['source_SHA256'][Path(__file__).name]:
                raise RuntimeError('Generation source changed during the experiment')
            resources('before_shot')
            tref = g.set_angle(solver, case, angle)
            raw, timing = solver.run()
            resources('after_shot')
            if raw.shape != (36001, 192) or not np.isfinite(raw).all():
                raise ValueError('Invalid native uniform-body RF')
            scattered = raw - reference[shot]
            full, _ = g.sim.analytic_channels(scattered, 2.5e-9, band=[1.5e6, 4.5e6])
            prefix, _ = g.sim.analytic_channels(scattered[:24001], 2.5e-9,
                                                band=[1.5e6, 4.5e6])
            if full.shape != (3601, 192) or prefix.shape != (2401, 192):
                raise ValueError('Processed RF clock changed')
            # Reuse the exact already-qualified 11-shot representation routine.
            # Broadcasting costs CPU arithmetic, but adds no simulation or disk RF.
            for tag, processed, destination in [('90', full, long), ('60', prefix, short)]:
                packed, meta = envelope_from_processed_RF(
                    np.broadcast_to(processed.T[None], (11, 192, processed.shape[0])))
                if not np.isfinite(packed[0]).all():
                    raise ValueError('Nonfinite compact uniform-body RF')
                destination[shot] = packed[0]; destination.flush()
                clock_metadata[tag] = meta
                del packed
            shots.append({'angle_index': shot, 'angle_deg': float(angle),
                          'source_tref_s': float(tref),
                          'solve_seconds': float(timing['solve_readback_s']),
                          'RF90_envelope_SHA256': chunk_hash(long[shot]),
                          'RF60_envelope_SHA256': chunk_hash(short[shot])})
            state.update(completed_shots=len(shots), seconds=time.time() - started)
            atomic(args.out / 'shots.json', {'completed_shots': shots,
                                             'clock_metadata': clock_metadata})
            atomic(args.out / 'run.json', state)
            print(json.dumps(shots[-1]), flush=True)
            del raw, scattered, full, prefix
        del solver, long, short
        gc.collect()
        long_path = args.out / 'RF90_envelope.npy'
        short_path = args.out / 'RF60_envelope.npy'
        long_temp.replace(long_path); short_temp.replace(short_path)
        result = {
            'status': 'known_uniform_body_reference_complete',
            'known_material': str(native_path.resolve()),
            'context': str((args.out / 'known_uniform_context.npy').resolve()),
            'baseline_ROI': str((args.out / 'known_uniform_ROI.npy').resolve()),
            'RF_views': [{'raw_rf': str(path.resolve()), 'SHA256': digest(path),
                          **clock_metadata[tag]} for tag, path in
                         [('60', short_path), ('90', long_path)]],
            'material_truth_used': False, 'new_wave_solves': len(shots),
            'new_patient_materials': 0, 'source_frequency_hz': 3e6,
            'water_reference_SHA256': state['water_reference_SHA256'],
            'known_material_SHA256': state['known_material_SHA256'],
        }
        atomic(args.out / 'index.json', result)
        state.update(status='known_uniform_body_reference_complete',
                     finished_unix_s=time.time(), seconds=time.time() - started,
                     index_SHA256=digest(args.out / 'index.json'))
        atomic(args.out / 'run.json', state)
        print(json.dumps(state), flush=True)
    except BaseException:
        state.update(status='failed', error=traceback.format_exc(), failed_unix_s=time.time())
        atomic(args.out / 'run.json', state)
        raise


if __name__ == '__main__':
    main()
