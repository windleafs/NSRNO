"""Finalize the completed CDF boundary from saved products, without wave solves.

The original native run, exception, work counters and deadline are immutable.
Truth is read only by the existing post-hoc evaluator after both estimates freeze.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from .low_frequency_0p5 import atomic, digest, SAMPLE


def link_exact(path, target):
    target = Path(target).resolve(strict=True)
    if path.exists():
        if path.resolve() != target:
            raise RuntimeError('An existing fixed product has different provenance')
    else:
        path.symlink_to(target)


def check_refs(refs, plan):
    expected = sorted((name, shot) for name in plan['families'] for shot in range(11))
    if len(refs) != 44 or sorted((r['family'], r['shot']) for r in refs) != expected:
        raise RuntimeError('Complete original44 predictions required; no new solves')
    for row in refs:
        if digest(row['path']) != row['SHA256']:
            raise RuntimeError('A completed source prediction changed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    allowed = Path('/home_nvme/zhuangyang/nsrno/out/v5/reconstruction_global_CDF_continuation_v1')
    if root != allowed.resolve():
        raise ValueError('Only the recorded scientific boundary can be finalized')
    receipt_path = root/'scientific_boundary_CPU_finalization.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['status'] == 'completed_saved_boundary_CPU_finalization':
            print(json.dumps(receipt), flush=True)
            return
        raise RuntimeError('Inspect the existing finalization; do not repeat truth evaluation')
    native_run_SHA = digest(root/'run.json')
    run = json.loads((root/'run.json').read_text())
    event = json.loads((root/'scientific_boundary_stop.json').read_text())
    workers = json.loads((root/'worker_processes.json').read_text())
    work = json.loads((root/'actual_native_work.json').read_text())
    if (run['pid'] != 3619749 or run['start_ticks'] != 11407646 or
            run['boot_id'] != 'c86bf89c-042b-40c7-98c5-51da9b5741d3' or
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() != run['boot_id'] or
            run['status'] != 'CDF_native_frequency_continuation_failed_for_inspection' or
            'Absolute trajectory deadline or cooperative stop reached' not in run['error'] or
            event['action'] != 'cooperative_stop_requested_at_complete_accepted_full_boundary' or
            not event['no_native_work_interrupted'] or
            event['native_pid'] != run['pid'] or event['native_start_ticks'] != run['start_ticks']):
        raise RuntimeError('This is not the recorded intentional scientific stop')
    if workers['worker_pids'] != [3619753, 3619754]:
        raise RuntimeError('Unexpected original worker identity')
    if any(Path('/proc', str(pid)).exists() for pid in [run['pid'], *workers['worker_pids'], event['controller_pid']]):
        raise RuntimeError('Wait until the original coordinator, workers and controller are gone')
    if (work['unfinished_native_phases'] != 0 or
            work['issued_adjoint_direction_passes'] != work['completed_adjoint_direction_passes'] or
            work['issued_physical_forward_passes'] != work['completed_physical_forward_passes'] or
            work['issued_tangent_direction_passes'] != 0):
        raise RuntimeError('Do not finalize incomplete original work')
    last = run['accepted_steps'][-1]
    if (last['objective'] != 'full_joint' or last['point'] != event['point'] or
            last['material_SHA256'] != event['accepted_material_SHA256']):
        raise RuntimeError('The scientific boundary is not the last accepted result')
    point = root/'steps'/f'point{last["point"]:02d}'
    cache = json.loads((point/'accepted_source_cache.json').read_text())
    candidate = point/f'candidate{run["active_attempt"]:02d}'
    accepted = json.loads((candidate/'actual_native_acceptance.json').read_text())
    if (not accepted['accepted'] or accepted['material_SHA256'] != last['material_SHA256'] or
            digest(candidate/'material.npy') != last['material_SHA256'] or
            cache['material_SHA256'] != last['material_SHA256'] or cache['source_count'] != 44 or
            cache['actual_native_RF'] != last['RF']):
        raise RuntimeError('Completed accepted candidate/cache provenance disagrees')
    plan = json.loads((root/'native_plan.json').read_text())
    check_refs(cache['RF_source_records'], plan)
    best = json.loads((root/'best_full_RF_incumbent.json').read_text())
    if digest(best['material_path']) != best['material_SHA256']:
        raise RuntimeError('The saved fullRF incumbent changed')
    check_refs(best['RF_source_records'], plan)
    prepared = json.loads((Path(run['inputs_root'])/'run.json').read_text())
    trajectory, final = root/'trajectory_final', root/'final'
    trajectory.mkdir(exist_ok=True)
    final.mkdir(exist_ok=True)
    for folder, material, control in [
            (trajectory, candidate/'material.npy', candidate/'control.npy'),
            (final, best['material_path'], best['control_path'])]:
        link_exact(folder/'material.npy', material)
        link_exact(folder/'control.npy', control)
    atomic(trajectory/'actual_native_acceptance.json', {
        'accepted_updates': run['accepted_updates'], 'actual_native_RF': {'loss': last['RF'], 'all44_sources': True},
        'RF_source_records': cache['RF_source_records'], 'native_TV': cache['parts']['TV'],
        'independent_prior_value': cache['prior'], 'material_SHA256': last['material_SHA256'],
        'full_RF_improved_from_initial': last['RF'] < prepared['initial_full_RF'],
        'source': 'completed_accepted_boundary_cache', 'not_a_joint_imaging_success_claim': True})
    atomic(final/'actual_native_acceptance.json', {
        'actual_native_RF': {'loss': best['loss'], 'all44_sources': True},
        'RF_source_records': best['RF_source_records'], 'material_SHA256': best['material_SHA256'],
        'native_TV': best['TV'], 'independent_prior_value': best['prior'],
        'full_original44_source_acceptance': True, 'joint_imaging_success_claimed': False})
    receipt = {'status': 'both_estimates_fixed_before_posthoc_truth_read', 'pid': os.getpid(),
        'unix_s': time.time(), 'native_run_SHA256': native_run_SHA,
        'intentional_scientific_boundary': event, 'original_actual_work': work,
        'original_created_unix_s': run['created_unix_s'], 'original_deadline_unix_s': run['deadline_unix_s'],
        'original_limits_unchanged': True, 'original_budget_closed_at_actual_counts': True,
        'unused_limits_not_reopened': True, 'new_wave_solves': 0, 'new_material_derivative_probes': 0,
        'accepted_low_updates': run['accepted_low_updates'], 'accepted_full_updates': run['accepted_full_updates'],
        'trajectory_material_SHA256': last['material_SHA256'], 'trajectory_RF': last['RF'],
        'final_material_SHA256': best['material_SHA256'], 'final_RF': best['loss'],
        'initial_full_RF': prepared['initial_full_RF'], 'joint_imaging_achieved': False}
    atomic(receipt_path, receipt)
    link_exact(trajectory/'stage0_initial_material.npy', prepared['initial_material_path'])
    with (trajectory/'maps.log').open('x') as log:
        subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material', str(trajectory/'material.npy'),
            '--sample', SAMPLE, '--out', str(trajectory/'maps'), '--estimate-label', 'CDF + residual prior: fixed boundary'],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=subprocess.STDOUT,
            check=True, timeout=120)
    with (trajectory/'increment.log').open('x') as log:
        subprocess.run([sys.executable, '-m', 'nsrno.posthoc_fixed_material_increment', '--candidate', str(trajectory),
            '--initial-material', prepared['initial_material_path'], '--initial-control', prepared['initial_control_path']],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=subprocess.STDOUT,
            check=True, timeout=60)
    same = best['material_SHA256'] == last['material_SHA256']
    original = best['material_SHA256'] == prepared['initial_material_SHA256']
    if not same and not original:
        # It can be the low4 point scored by the required full44 gradient.
        link_exact(final/'stage0_initial_material.npy', prepared['initial_material_path'])
        with (final/'maps.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material', str(final/'material.npy'),
                '--sample', SAMPLE, '--out', str(final/'maps'), '--estimate-label', 'Saved global fullRF incumbent'],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=120)
    else:
        maps_source = trajectory if same else Path(run['parent_root'])/'final'
        for suffix in ['png', 'metrics.json']:
            shutil.copy2(maps_source/f'maps.{suffix}', final/f'maps.{suffix}')
    atomic(root/'maps_provenance.json', {'trajectory_new_fixed_material_evaluation': True,
        'final_same_as_trajectory': same, 'final_is_original_incumbent': original,
        'no_intermediate_truth_feedback': True, 'native_run_unchanged': digest(root/'run.json') == native_run_SHA})
    if digest(root/'run.json') != native_run_SHA:
        raise RuntimeError('The original native receipt must remain immutable')
    receipt.update(status='completed_saved_boundary_CPU_finalization', finished_unix_s=time.time())
    atomic(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
