"""CPU direction from same-node receiver-space curvature and independent TV.

This does not accept a material candidate or propagate a wave. Acceptance
belongs to an exact simultaneous field/material objective over all data.
"""
import numpy as np
from .coupled_coordinates import pcg, dot


class JointMaterialDirection:
    def __init__(self, variables, blocks, data_gradient):
        self.vars = variables
        self.blocks = np.asarray(blocks, np.float64)
        self.data_gradient = np.asarray(data_gradient, np.float64)
        if self.blocks.shape != (6, *variables.shape) or self.data_gradient.shape != (3, *variables.shape):
            raise ValueError('Expected six local curvature maps and three gradient maps')
        if not np.isfinite(self.blocks).all() or not np.isfinite(self.data_gradient).all():
            raise RuntimeError('Nonfinite receiver-space material statistics')

    def apply(self, q):
        q = q*self.vars.unknown
        a, b, c, ab, ac, bc = self.blocks
        return np.stack([a*q[0]+ab*q[1]+ac*q[2],
            ab*q[0]+b*q[1]+bc*q[2], ac*q[0]+bc*q[1]+c*q[2]])*self.vars.unknown

    def solve(self, current, tv_weight=.001, damping_relative=.01, maximum=64,
              trust=(.03, .04, .10), callback=None):
        tv, gtv, htv, tvlocal = self.vars.tv(current, tv_weight)
        unknown = self.vars.unknown
        gradient = (self.data_gradient+gtv)*unknown
        right = -gradient
        diagonal_data = np.maximum(self.blocks[:3], 0)
        reference = np.array([max(float(np.median(a[unknown][a[unknown] > 0]))
            if np.any(a[unknown] > 0) else 0., 1e-18) for a in diagonal_data])
        base = damping_relative*np.maximum(diagonal_data, reference[:, None, None]*1e-4)
        base[:, ~unknown] = 1.
        target = np.asarray(trust, np.float64)/.1
        multipliers = np.zeros(3, np.float64)
        records = []

        def rms(q):
            return np.sqrt(np.mean(q[:, unknown]**2, axis=1))

        def preconditioner(shifts):
            damping = base+shifts[:, None, None]*unknown
            diagonal = np.maximum(diagonal_data+tvlocal[:3]+damping, 1e-30)
            scales = 1/np.sqrt(diagonal)
            cross = self.blocks[3:]+tvlocal[3:]
            a = cross[0]*scales[0]*scales[1]/1.00001
            b = cross[1]*scales[0]*scales[2]/1.00001
            c = cross[2]*scales[1]*scales[2]/1.00001
            determinant = np.maximum(1+2*a*b*c-a*a-b*b-c*c, 1e-12)
            inverse = np.stack([1-c*c, 1-b*b, 1-a*a, b*c-a, a*c-b, a*b-c])/determinant
            def pre(value):
                y = value*scales
                return np.stack([inverse[0]*y[0]+inverse[3]*y[1]+inverse[4]*y[2],
                    inverse[3]*y[0]+inverse[1]*y[1]+inverse[5]*y[2],
                    inverse[4]*y[0]+inverse[5]*y[1]+inverse[2]*y[2]])*scales
            return pre, damping, diagonal/np.maximum(inverse[:3], 1e-30)

        def linear(shifts, warm=None, label=''):
            pre, damping, _ = preconditioner(shifts)
            def operator(q):
                return self.apply(q)+htv(q)+damping*q
            initial = np.zeros_like(right) if warm is None else warm
            residual = right-operator(initial) if warm is not None else right
            correction, history = pcg(operator, residual, pre, maximum=maximum,
                tolerance=.01, callback=callback)
            solution = (initial+correction)*unknown
            residual = right-operator(solution)
            relative = np.sqrt(max(dot(residual, pre(residual)), 0.)/
                max(dot(right, pre(right)), 1e-300))
            records.append({'label': label, 'dual_shifts': shifts.tolist(),
                'iterations': len(history), 'global_preconditioned_relative_residual': float(relative),
                'RMS_normalized': rms(solution).tolist()})
            return solution, history

        direction, history = linear(multipliers, label='local_receiver_GN')
        raw_rms = rms(direction)
        _, _, effective = preconditioner(multipliers)
        effective_reference = np.array([float(np.median(a[unknown])) for a in effective])
        for sweep in range(6):
            for coordinate in range(3):
                if multipliers[coordinate] == 0 and rms(direction)[coordinate] <= target[coordinate]*1.001:
                    continue
                zero = multipliers.copy()
                zero[coordinate] = 0
                at_zero, _ = linear(zero, direction, f'trust{sweep}_{coordinate}_zero')
                if rms(at_zero)[coordinate] <= target[coordinate]:
                    multipliers, direction = zero, at_zero
                    continue
                low = 0.
                high = max(multipliers[coordinate], effective_reference[coordinate]*
                    max(rms(at_zero)[coordinate]/target[coordinate]-1, .1), 1e-20)
                high_direction = at_zero
                for bracket in range(24):
                    shifted = multipliers.copy()
                    shifted[coordinate] = high
                    high_direction, _ = linear(shifted, high_direction,
                        f'trust{sweep}_{coordinate}_bracket{bracket}')
                    if rms(high_direction)[coordinate] <= target[coordinate]:
                        break
                    high *= 4
                else:
                    raise RuntimeError('Unable to bracket a joint-material RMS multiplier')
                warm = high_direction
                for iteration in range(12):
                    mid = (low+high)/2
                    shifted = multipliers.copy()
                    shifted[coordinate] = mid
                    trial, _ = linear(shifted, warm, f'trust{sweep}_{coordinate}_bisect{iteration}')
                    value = rms(trial)[coordinate]
                    warm = trial
                    if value > target[coordinate]:
                        low = mid
                    else:
                        high, high_direction = mid, trial
                    if target[coordinate]*.99 <= value <= target[coordinate]:
                        break
                multipliers[coordinate], direction = high, high_direction
            values = rms(direction)
            active = multipliers > 0
            if np.all(values <= target*1.005) and np.all(values[active] >= target[active]*.98):
                break
        common = min(1., 1/max(float(np.max(rms(direction)/target)), 1e-30))
        direction *= common
        kind = 'same_node_receiver_GN_with_coupled_RMS_constraints'
        if dot(gradient, direction) >= 0:
            direction = -gradient/np.maximum(diagonal_data+tvlocal[:3]+base, 1e-30)
            direction *= min(1., 1/max(float(np.max(rms(direction)/target)), 1e-30))
            kind = 'common_scaled_diagonal_descent'
        proposal = current.copy()
        projection_scale = 1.
        for _ in range(16):
            trial = self.vars.project(current+.1*projection_scale*direction)
            actual = trial-current
            if dot(gradient, actual/.1) < 0 and np.all(rms(actual) <= np.asarray(trust)*1.01):
                proposal = trial
                break
            projection_scale *= .5
        return proposal, {'direction_kind': kind, 'CG_history': history,
            'linear_solves': records, 'trust_dual_shifts': multipliers.tolist(),
            'trust_RMS_r_beta_t': list(trust), 'raw_RMS_r_beta_t': (.1*raw_rms).tolist(),
            'projected_RMS_r_beta_t': rms(proposal-current).tolist(),
            'common_safety_scale': common, 'projection_common_scale': projection_scale,
            'projected_gradient_dot_step': dot(gradient, (proposal-current)/.1),
            'zero_material_direction': bool(not np.any(proposal-current)),
            'TV_before': tv, 'curvature_damping_reference': reference.tolist(),
            'off_node_data_curvature_omitted': True, 'truth_read': False,
            'material_accepted': False, 'acceptance_requires_joint_field_cost': True}
