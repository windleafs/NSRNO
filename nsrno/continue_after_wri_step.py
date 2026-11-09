"""One-shot result-gated route change; no signals, repeated probes or truth arrays.

The gate reads already produced post-hoc scalar metrics. These never enter
the pressure optimizer, its RF candidate selection, or a material objective.
If the complete result is not clearly stagnant, it stops for agent analysis.
"""
from pathlib import Path
import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from .reconstruct_pressure_wri import SOURCES, sha, live
from .wri_worker import atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-step', type=int, default=4)
    parser.add_argument('--source-pid', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pressure-out', type=Path, required=True)
    parser.add_argument('--timeout-hours', type=float, default=6.)
    args = parser.parse_args()
    source = args.source_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out/'watch.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'state.json').exists():
        raise RuntimeError('A result gate record already exists; inspect it instead of duplicating')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    command = Path(f'/proc/{args.source_pid}/cmdline')
    if not command.exists() or not command.read_bytes():
        raise RuntimeError('The expected running source identity is absent')
    identity = {'pid': args.source_pid, 'cmdline': command.read_bytes().decode().replace('\0', ' '),
        'start_ticks': Path(f'/proc/{args.source_pid}/stat').read_text().rsplit(')', 1)[1].split()[19],
        'boot_id': boot}
    prepared = {name: sha(Path(__file__).with_name(name)) for name in SOURCES}
    prepared[Path(__file__).name] = sha(Path(__file__))
    archive = out/'prepared_source'
    archive.mkdir()
    for name in prepared:
        shutil.copy2(Path(__file__).with_name(name), archive/name)
    atomic(out/'prepared_source_sha256.json', prepared)
    atomic(out/'source_identity.json', identity)
    start = time.time()
    state = {'status': 'waiting_complete_source_result', 'pid': os.getpid(), 'source_pid': args.source_pid,
        'source_root': str(source), 'source_step': args.source_step, 'boot_id': boot,
        'pressure_out': str(args.pressure_out.resolve()), 'created_unix_s': start,
        'signals_sent': False, 'patient_probes_started': False, 'pressure_launched': False,
        'criterion': 'actual c/rho RMS update <1 m/s/<1 kg m^-3 and both abs correlation changes <.05'}
    atomic(out/'state.json', state)
    try:
        while time.time()-start < args.timeout_hours*3600:
            if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != boot:
                raise RuntimeError('Source boot changed')
            run = json.loads((source/'run.json').read_text())
            if run['pid'] != args.source_pid or run['step'] != args.source_step:
                raise RuntimeError('Source coordinator or outer step changed')
            if live(args.source_pid):
                now = Path(f'/proc/{args.source_pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
                cmd = Path(f'/proc/{args.source_pid}/cmdline').read_bytes().decode().replace('\0', ' ')
                if now != identity['start_ticks'] or cmd != identity['cmdline']:
                    raise RuntimeError('Source process identity was reused')
            if run['status'] == 'failed':
                state.update(status='source_failed_requires_analysis', source_error=run.get('error'))
                atomic(out/'state.json', state)
                return
            folder = source/f'step{args.source_step:03d}'
            ready = (run['status'] == 'revised_step_finished' and not live(args.source_pid)
                and run.get('completed_step') == args.source_step
                and all((folder/name).exists() for name in
                    ['complete_sweep.json', 'rf_score.json', 'maps.metrics.json', 'material.npy', 'material_update.json']))
            if not ready:
                state.update(last_source_phase=run['status'], last_checked_unix_s=time.time())
                atomic(out/'state.json', state)
                time.sleep(20)
                continue
            for name, digest in prepared.items():
                if sha(Path(__file__).with_name(name)) != digest:
                    raise RuntimeError('Prepared pressure source changed before activation: '+name)
            previous = source/f'step{args.source_step-1:03d}'
            old = np.load(previous/'material.npy').astype(np.float64)
            current = np.load(folder/'material.npy').astype(np.float64)
            if old.shape != (3, 961, 1121) or current.shape != old.shape:
                raise RuntimeError('The reviewed reconstruction is not the complete native material')
            rms = np.sqrt(np.mean((current[:, 80:]-old[:, 80:])**2, axis=(-2, -1)))
            del old, current
            prior_metrics = json.loads((previous/'maps.metrics.json').read_text())
            metrics = json.loads((folder/'maps.metrics.json').read_text())
            changes = {}
            for key in ['c_m_s', 'density_kg_m3']:
                a = float(prior_metrics['maps'][key]['reconstruction']['correlation'])
                b = float(metrics['maps'][key]['reconstruction']['correlation'])
                changes[key] = {'previous': a, 'current': b, 'change': b-a}
            finite = all(math.isfinite(x['change']) for x in changes.values()) and np.isfinite(rms).all()
            stagnant = bool(finite and rms[0] < 1 and rms[1] < 1
                            and all(abs(x['change']) < .05 for x in changes.values()))
            rf = json.loads((folder/'rf_score.json').read_text())
            decision = {'source_step': args.source_step, 'source_root': str(source),
                'source_material_sha256': sha(folder/'material.npy'),
                'source_RF_sha256': sha(folder/'rf_score.json'), 'source_full_RF_loss': rf['primary_loss'],
                'physical_update_RMS_c_rho_alpha': rms.tolist(),
                'post_hoc_scalar_correlations': changes,
                'activate_pressure_route': stagnant,
                'reason': 'complete actual c/rho update remains stagnant under the changed dual state' if stagnant
                          else 'result falls outside the predeclared clear-stagnation condition; analyze before choosing route',
                'criterion': state['criterion'], 'truth_arrays_read': False,
                'post_hoc_metrics_used_only_for_research_route_decision': True,
                'optimizer_uses_truth': False, 'signals_sent': False, 'unix_s': time.time()}
            atomic(out/'decision.json', decision)
            if not stagnant:
                state.update(status='complete_result_requires_mechanism_analysis', decision=str(out/'decision.json'))
                atomic(out/'state.json', state)
                return
            argv = [sys.executable, '-m', 'nsrno.reconstruct_pressure_wri', '--out', str(args.pressure_out.resolve()),
                '--source-root', str(source), '--source-step', str(args.source_step),
                '--decision', str(out/'decision.json'), '--devices', '0', '1', '2',
                '--cap-gib', '14.5', '--iterations', '1', '--field-cg', '64', '--material-cg', '64']
            with (out/'pressure_controller.log').open('w', buffering=1) as stream:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='pressure_start_issued', pressure_launched=True, pressure_pid=process.pid,
                pressure_argv=argv, decision=str(out/'decision.json'), launched_unix_s=time.time())
            atomic(out/'state.json', state)
            time.sleep(3)
            state.update(status='pressure_controller_alive' if process.poll() is None else 'pressure_controller_exited_early',
                         pressure_returncode=process.poll())
            atomic(out/'state.json', state)
            return
        state.update(status='bounded_watch_timeout', waited_seconds=time.time()-start)
        atomic(out/'state.json', state)
    except BaseException:
        state.update(status='failed_without_launch' if not state['pressure_launched'] else 'failed_after_launch',
                     error=traceback.format_exc())
        atomic(out/'state.json', state)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
