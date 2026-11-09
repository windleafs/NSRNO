"""Same unperturbed training baseline for both signs of each independent change."""
import numpy as np
from .conditional_secant_data import (TrainingPairs as OriginalPairs, array,
                                      tissue_label, rf_input, tensor_batch)


class TrainingPairs(OriginalPairs):
    def __init__(self, index):
        super().__init__(index)
        self.original_by_id_mode = {(row['id'], row['valid_samples']): row for row in self.rows
                                   if row['kind'] in ['original', 'coverage_base']}
        self.central = {}
        for kind, groups in self.groups.items():
            for members in groups:
                parent = members[0]['parent_id']; n = members[0]['valid_samples']
                base = self.original_by_id_mode.get((parent, n))
                if base is None:
                    choices = [r for r in self.rows if r['parent_id'] == parent
                               and r['valid_samples'] == n and r['kind'] in ['original', 'coverage_base']]
                    if len(choices) != 1:
                        raise ValueError('Missing unique unperturbed baseline: ' + str((parent, n)))
                    base = choices[0]
                self.central[(members[0]['pair_group'], n)] = base

    def evaluation_pairs(self):
        for kind, groups in self.groups.items():
            for members in groups:
                base = self.central[(members[0]['pair_group'], members[0]['valid_samples'])]
                for target in members:
                    yield kind, base, target

    def sample(self, rng, step, profile=False):
        # RNG calls and task frequencies match the preceding experiment.
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
            target = pair[1] if rng.random() < .5 else pair[0]
            baseline = self.central[(target['pair_group'], target['valid_samples'])]
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
                raise ValueError('Paired acquisition clocks differ')
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
