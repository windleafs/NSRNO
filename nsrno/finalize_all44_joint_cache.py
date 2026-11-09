"""Persist the completed accepted RF cache in the format used by later inversions."""
from pathlib import Path
import argparse
import json
from .low_frequency_0p5 import atomic, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path.cwd().resolve()/'out/v5'):
        raise ValueError('Dedicated nsrno computational output only')
    run = json.loads((root/'run.json').read_text())
    processes = json.loads((root/'worker_processes.json').read_text())
    if (run['status'] != 'all44_joint_tangent_stage_finished' or
        any(Path(f'/proc/{pid}').exists() for pid in [run['pid']]+processes['worker_pids'])):
        raise RuntimeError('Wait for the exact same coordinator and workers to exit')
    out = root/'final/actual_native_acceptance.json'
    if out.exists():
        print(json.dumps({'status': 'completed_cache_already_indexed', 'file': str(out)}))
        return
    plan = json.loads((root/'native_plan.json').read_text())
    candidate = json.loads((root/'candidate/actual_native_acceptance.json').read_text())
    if run['accepted_original_RF']:
        entries = [(family, shot) for family in plan['families'] for shot in range(11)]
        rows = []
        for index, (family, shot) in enumerate(entries):
            rank = index % len(processes['worker_pids'])
            path = root/'workers'/f'rank{rank:02d}'/'arrays/token0002'/f'{family}_{shot:02d}.npy'
            rows.append({'family': family, 'shot': shot, 'path': str(path.resolve()), 'SHA256': digest(path)})
        RF = {'loss': run['final_RF'], 'all44_sources': True, 'source_encoding': False}
        tv = candidate['TV_after']
    else:
        old = json.loads((Path(run['parent_root'])/'final/actual_native_acceptance.json').read_text())
        rows, RF, tv = old['RF_source_records'], old['actual_native_RF'], old['native_TV']
    if sorted((row['family'], row['shot']) for row in rows) != sorted(
            (family, shot) for family in plan['families'] for shot in range(11)):
        raise RuntimeError('Incomplete accepted original44 RF cache')
    if digest(root/'final/material.npy') != run['material_SHA256']:
        raise RuntimeError('Completed retained material changed')
    atomic(out, {'accepted': run['accepted_original_RF'], 'actual_native_RF': RF,
        'RF_source_records': rows, 'native_TV': tv, 'material_SHA256': run['material_SHA256'],
        'full_original44_source_acceptance': True, 'joint_imaging_success_claimed': False,
        'no_acoustic_work_added_by_cache_indexing': True})
    atomic(root/'direction_input_checksums.json', {name: digest(root/'arrays'/name) for name in
        ['control_directions.npy', 'physical_directions.npy']})
    print(json.dumps({'status': 'completed_all44_accepted_cache_indexed', 'accepted': run['accepted_original_RF'],
        'source_records': len(rows), 'material_SHA256': run['material_SHA256'],
        'native_RF': RF['loss'], 'new_acoustic_propagations': 0}), flush=True)


if __name__ == '__main__':
    main()
