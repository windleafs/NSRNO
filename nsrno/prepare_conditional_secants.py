"""Prepare real matched-duration finite-secant training views; CPU only, no new waves."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time
import traceback
import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter
from .native_gn_json import atomic


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def material_from_training_row(row):
    if row['role'] != 'train' or row['case'] != 'Neg_07_Left':
        raise ValueError('Only training materials may supply conditioning contexts here')
    if row.get('material3'):
        paths = [Path(row['material3'])]
        material = np.load(paths[0])
    else:
        paths = [Path(row['medium']), Path(row['alpha'])]
        material = np.concatenate((np.load(paths[0]), np.load(paths[1])[None]), 0)
    if material.shape != (3, 961, 1121):
        raise ValueError('Full native training material required')
    return material, {str(p.resolve()): digest(p) for p in paths}


def roi(material):
    return np.stack([gaussian_filter(np.asarray(field, np.float32), sigma=1.)[
        np.ix_(60 + 4 * np.arange(216), 178 + 4 * np.arange(192))] for field in material]).astype(np.float32)


def envelope_from_processed_RF(value):
    if value.ndim != 3 or value.shape[:2] != (11, 192) or value.shape[-1] not in (2401, 3601):
        raise ValueError('Complete existing processed LF RF required')
    n = value.shape[-1]; period = n / 40e6
    positive = fft.rfft(value, axis=-1, norm='forward')
    frequencies = fft.rfftfreq(n, 1 / 40e6)
    bins = np.flatnonzero((frequencies >= 1.5e6) & (frequencies <= 4.5e6))
    centre = round(3e6 * period); samples = math.ceil(6e6 * period)
    offsets = bins - centre
    spectrum = np.zeros((11, 192, samples), np.complex64)
    spectrum[..., offsets % samples] = math.sqrt(2.) * positive[..., bins]
    t = np.arange(samples, dtype=np.float64) * period / samples
    rotation = np.exp(2j * np.pi * (centre / period - 3e6) * t).astype(np.complex64)
    result = fft.ifft(spectrum, axis=-1, norm='forward') * rotation
    return np.stack((result.real, result.imag), 1).astype(np.float32), {
        'valid_samples': samples, 'sample_dt_s': period / samples,
        'selected_frequency_count': len(bins), 'carrier_hz': 3e6}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, default=Path('out/v5/raw_rf_window_balanced_data_v1/index.json'))
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('Preserve existing preparation; inspect, never overwrite implicitly')
    source = json.loads(a.index.read_text())
    train = [r for r in source['records'] if r['role'] == 'train']
    dev = [r for r in source['records'] if r['role'] != 'train']
    if len(train) != 112 or {r['case'] for r in train} != {'Neg_07_Left'}:
        raise ValueError('Training partition changed')
    if {r['case'] for r in dev} != {'Neg_35_Left'}:
        raise ValueError('Unexpected development partition')
    previous = json.loads((a.native_root / 'run.json').read_text())
    workers = json.loads((a.native_root / 'worker_processes.json').read_text())
    if previous['status'] != 'native_RF_L2_control_finished':
        raise RuntimeError('Complete previous reconstruction first')
    if any(pid and Path('/proc', str(pid)).exists()
           for pid in [previous['pid'], previous.get('child_pid'), *workers['worker_pids']]):
        raise RuntimeError('Previous native processes still exist')
    a.out.mkdir(); (a.out / 'contexts').mkdir(); (a.out / 'inference').mkdir()
    run = {'status': 'preparing_conditional_secants', 'pid': os.getpid(),
           'created_unix_s': time.time(), 'source_index_SHA256': digest(a.index),
           'source_SHA256': digest(__file__), 'GPU_work': False, 'new_wave_solves': 0,
           'development_material_truth_read': False, 'reserved_case_unread': 'Neg_47_Left'}
    atomic(a.out / 'run.json', run)
    started = time.time()
    try:
        edge = np.broadcast_to(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'), (961, 1121))
        edge_roi = gaussian_filter(np.asarray(edge, np.float32), sigma=1.)[
            np.ix_(60 + 4 * np.arange(216), 178 + 4 * np.arange(192))]
        np.save(a.out / 'ROI_known_edge.npy', edge_roi)
        reference_context = np.empty((3, 241, 281), np.float32)
        reference_roi = np.empty((3, 216, 192), np.float32)
        for j, value in enumerate((1540., 1000., .002)):
            reference_context[j] = value; reference_roi[j] = value
        np.save(a.out / 'reference_context.npy', reference_context)
        np.save(a.out / 'reference_ROI.npy', reference_roi)
        keep = ['id', 'role', 'case', 'kind', 'parent_id', 'pair_group', 'pair_sign',
                'changed_parameter', 'valid_samples', 'sample_dt_s', 'raw_rf', 'label',
                'raw_rf_SHA256', 'field_scale_mm']
        rows = []
        for row in source['records']:
            item = {k: row[k] for k in keep if k in row}
            if row['role'] == 'train':
                material, inputs = material_from_training_row(row)
                context = material[:, ::4, ::4].copy().astype(np.float32)
                context[2] -= edge[::4, ::4]
                destination = a.out / 'contexts' / (row['id'] + '.npy')
                np.save(destination, context)
                item.update(context=str(destination.resolve()),
                            context_SHA256=digest(destination), conditioning_material_inputs=inputs)
                del material, context
            item['view_id'] = item['id'] + '::' + str(item['valid_samples'])
            rows.append(item)
            if row.get('short_view'):
                short = {k: row['short_view'][k] for k in keep if k in row['short_view']}
                for k in ('context', 'context_SHA256', 'conditioning_material_inputs'):
                    if k in item:
                        short[k] = item[k]
                short['view_id'] = short['id'] + '::' + str(short['valid_samples'])
                rows.append(short)
        if len({r['view_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicated material-duration view')
        training_views = [r for r in rows if r['role'] == 'train']
        if len(training_views) != 168:
            raise ValueError('Expected 112 existing records and 56 actual short views')
        # Inference baseline is an already reconstructed estimate, not a truth array.
        matfile = a.native_root / 'step001/material.npy'
        if digest(matfile) != previous['material_SHA256']:
            raise RuntimeError('Protected current estimate changed')
        estimate = np.load(matfile); current_roi = roi(estimate); current_roi[2] -= edge_roi
        current_roi[:, :5] = np.asarray([1540., 1000., .002], np.float32)[:, None, None]
        context = estimate[:, ::4, ::4].copy(); context[2] -= edge[::4, ::4]
        np.save(a.out / 'inference/current_ROI.npy', current_roi)
        np.save(a.out / 'inference/current_context.npy', context)
        point = json.loads((a.native_root / 'accepted_point.json').read_text())
        token = point['native_RF_quadratic']['token']
        predictions = []; prediction_hashes = {}
        for shot in range(11):
            matches = list((a.native_root / 'workers').glob(f'rank*/RF/token{token:04d}/lf_{shot:02d}.npy'))
            if len(matches) != 1:
                raise RuntimeError('Missing unique archived current-material RF')
            prediction_hashes[str(matches[0].resolve())] = digest(matches[0])
            predictions.append(np.load(matches[0]))
        predicted, meta = envelope_from_processed_RF(np.stack(predictions))
        # CPU acquisition loader only reads known geometry/reference and measured RF.
        # predict/native_wave is never called.
        from .time_acquisition import TimeAcquisition
        acquisition = TimeAcquisition('lf', previous['sample'], device='cpu', chunk=128)
        measured, observed_meta = envelope_from_processed_RF(acquisition.observed.numpy())
        if meta != observed_meta:
            raise RuntimeError('Predicted and measured RF clocks differ')
        np.save(a.out / 'inference/current_predicted_envelope.npy', predicted)
        np.save(a.out / 'inference/current_observed_envelope.npy', measured)
        inference = {'sample': previous['sample'], **meta,
                     'baseline_material': str(matfile.resolve()), 'baseline_material_SHA256': digest(matfile),
                     'baseline_RF_archives_SHA256': prediction_hashes,
                     'baseline_RF_evaluation': point['evaluation'], 'baseline_RF_token': token,
                     'baseline_ROI': str((a.out / 'inference/current_ROI.npy').resolve()),
                     'baseline_context': str((a.out / 'inference/current_context.npy').resolve()),
                     'baseline_RF': str((a.out / 'inference/current_predicted_envelope.npy').resolve()),
                     'observed_RF': str((a.out / 'inference/current_observed_envelope.npy').resolve()),
                     'truth_arrays_used': False, 'new_wave_solves': 0}
        atomic(a.out / 'inference/current.json', inference)
        result = {'records': rows, 'training_records': 112, 'training_views': 168,
                  'training_case': 'Neg_07_Left', 'development_case': 'Neg_35_Left',
                  'training_RF_component_RMS_scale': source['training_RF_component_RMS_scale'],
                  'reference_context': str((a.out / 'reference_context.npy').resolve()),
                  'reference_ROI': str((a.out / 'reference_ROI.npy').resolve()),
                  'ROI_known_edge': str((a.out / 'ROI_known_edge.npy').resolve()),
                  'training_mean_material': source['training_mean_material'],
                  'native_current_inference': str((a.out / 'inference/current.json').resolve()),
                  'source_index_SHA256': digest(a.index), 'new_wave_solves': 0,
                  'new_material_derivatives': 0, 'reserved_case_unread': 'Neg_47_Left',
                  'development_context_truth_not_prepared': True}
        atomic(a.out / 'index.json', result)
        run.update(status='conditional_secant_data_prepared', training_views=168,
                   total_views=len(rows), index_SHA256=digest(a.out / 'index.json'),
                   seconds=time.time() - started, finished_unix_s=time.time())
        atomic(a.out / 'run.json', run); print(json.dumps(run), flush=True)
    except BaseException:
        run.update(status='failed', error=traceback.format_exc(), failed_unix_s=time.time())
        atomic(a.out / 'run.json', run); raise


if __name__ == '__main__':
    main()
