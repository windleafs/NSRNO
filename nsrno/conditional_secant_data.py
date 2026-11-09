"""Real RF pair loading and training-only finite-secant sampling."""
from functools import lru_cache
import numpy as np
import torch


@lru_cache(maxsize=32)
def array(path):
    return np.load(path)


def tissue_label(row, edge):
    value = array(row['label']).copy()
    value[2] -= edge
    value[:, :5] = np.asarray([1540., 1000., .002], np.float32)[:, None, None]
    return value


def rf_input(base, target, n, dt, scale):
    if base.shape != (11, 2, 192, n) or target.shape != base.shape:
        raise ValueError('Paired RF must share all shots, receivers and actual record clock')
    difference = (target.astype(np.float64) - base) / scale
    rms = float(np.sqrt(np.mean(difference ** 2)))
    value = np.zeros((11, 6, 192, 544), np.float32)
    value[:, :2, :, :n] = base / scale
    value[:, 2:4, :, :n] = difference / max(rms, 1e-12)
    value[:, 4, :, :n] = 1.
    value[:, 5, :, :n] = np.arange(n, dtype=np.float32)[None, None, :] * dt / 90e-6
    return value, rms


class TrainingPairs:
    def __init__(self, index):
        self.index = index
        self.rows = [r for r in index['records'] if r['role'] == 'train']
        if {r['case'] for r in self.rows} != {'Neg_07_Left'}:
            raise ValueError('Development material in training partition')
        self.edge = array(index['ROI_known_edge'])
        self.by_mode = {}
        self.groups = {kind: [] for kind in ['pure_c', 'pure_rho', 'alpha']}
        grouped = {}
        for row in self.rows:
            self.by_mode.setdefault(row['valid_samples'], []).append(row)
            if row.get('pair_group') and row['kind'] in self.groups:
                grouped.setdefault((row['pair_group'], row['valid_samples']), []).append(row)
        for members in grouped.values():
            if len(members) != 2:
                raise ValueError('Incomplete independent material pair')
            members = sorted(members, key=lambda r: r['pair_sign'])
            self.groups[members[0]['kind']].append(members)
        if any(not values for values in self.groups.values()):
            raise ValueError('All three independent material kinds required')
        self.scale = index['training_RF_component_RMS_scale']

    def sample(self, rng, step, profile=False):
        if profile:
            candidates = self.by_mode[541]
            baseline = candidates[0]
            target = next(r for r in candidates if r['parent_id'] != baseline['parent_id'])
            task = 'cross_material'
        elif step % 5 == 0:
            baseline = None; target = self.rows[int(rng.integers(len(self.rows)))]
            task = 'reference_to_material'
        elif step % 5 in (1, 2):
            kind = ['pure_c', 'pure_rho', 'alpha'][(step // 5 * 2 + step % 5 - 1) % 3]
            pair = self.groups[kind][int(rng.integers(len(self.groups[kind])))]
            baseline, target = pair if rng.random() < .5 else pair[::-1]
            task = kind
        else:
            mode = [361, 541][int(rng.integers(2))]
            candidates = self.by_mode[mode]
            baseline = candidates[int(rng.integers(len(candidates)))]
            different = [r for r in candidates if r['parent_id'] != baseline['parent_id']]
            target = different[int(rng.integers(len(different)))]
            task = 'cross_material'
        n, dt = target['valid_samples'], target['sample_dt_s']
        rf_target = array(target['raw_rf'])
        if baseline is None:
            rf_base = np.zeros_like(rf_target)
            base_roi = array(self.index['reference_ROI']).copy()
            context = array(self.index['reference_context']).copy()
        else:
            if baseline['valid_samples'] != n or baseline['sample_dt_s'] != dt:
                raise ValueError('Do not mix acquisition clocks')
            rf_base = array(baseline['raw_rf'])
            base_roi = tissue_label(baseline, self.edge)
            context = array(baseline['context']).copy()
        value, rms = rf_input(rf_base, rf_target, n, dt, self.scale)
        target_roi = tissue_label(target, self.edge)
        flip = not profile and rng.random() < .5
        if flip:
            value = np.ascontiguousarray(value[::-1, :, ::-1])
            base_roi = np.ascontiguousarray(base_roi[:, :, ::-1])
            context = np.ascontiguousarray(context[:, :, ::-1])
            target_roi = np.ascontiguousarray(target_roi[:, :, ::-1])
        return {'signal': value, 'baseline': base_roi, 'context': context,
                'target': target_roi, 'rms': rms, 'task': task,
                'baseline_id': baseline['id'] if baseline is not None else 'known_uniform_reference',
                'target_id': target['id'], 'record_samples': n}


def tensor_batch(sample, device):
    return tuple(torch.from_numpy(sample[key][None]).to(device)
                 for key in ['signal', 'baseline', 'context', 'target']) + (
                     torch.tensor([sample['rms']], device=device, dtype=torch.float32),)
