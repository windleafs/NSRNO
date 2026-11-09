"""CPU coupled scalar-pressure material normal and RMS trust-region solve."""
import numpy as np
from .coupled_coordinates import pcg, dot
from .pressure_material_stencil import FIELDS, PAIRS


class PressureMaterialNormal:
    def __init__(self, variables, indices, packed_gram, local_right):
        self.vars = variables
        self.index = np.asarray(indices, np.int64)
        self.gram = packed_gram
        self.shape = tuple(variables.shape)
        self.size = int(np.prod(self.shape))
        if self.index.shape != (19, *self.shape) or self.gram.shape != (190, *self.shape):
            raise ValueError('Wrong native scalar-pressure normal shape')
        if local_right.shape != self.index.shape:
            raise ValueError('Wrong local right-hand side')
        self.rhs = self.fold(local_right)
        self.local = np.zeros((6, *self.shape), np.float64)
        cross_slot = {(0, 1): 3, (0, 2): 4, (1, 2): 5}
        for slot, (i, j) in enumerate(PAIRS):
            fi, fj = FIELDS[i], FIELDS[j]
            if i == j:
                self.local[fi] += self.fold_one(self.gram[slot], self.index[i])
            else:
                same = self.index[i] == self.index[j]
                if fi == fj:
                    self.local[fi] += self.fold_one(2 * self.gram[slot] * same, self.index[i])
                else:
                    self.local[cross_slot[(fi, fj)]] += self.fold_one(self.gram[slot] * same, self.index[i])
        self.local *= self.vars.unknown
        self.diagonal = np.maximum(self.local[:3], 0)

    def fold_one(self, value, index):
        return np.bincount(index.ravel(), weights=np.asarray(value).ravel(),
                           minlength=self.size).reshape(self.shape)

    def fold(self, local):
        out = np.zeros((3, *self.shape), np.float64)
        for i, field in enumerate(FIELDS):
            out[field] += self.fold_one(local[i], self.index[i])
        return out * self.vars.unknown

    def apply(self, q):
        q = np.asarray(q) * self.vars.unknown
        flat = q.reshape(3, -1)
        gathered = np.stack([flat[field, index] for field, index in zip(FIELDS, self.index)])
        local = np.zeros_like(gathered)
        temporary = np.empty(self.shape, np.float64)
        for slot, (i, j) in enumerate(PAIRS):
            np.multiply(self.gram[slot], gathered[j], out=temporary)
            local[i] += temporary
            if i != j:
                np.multiply(self.gram[slot], gathered[i], out=temporary)
                local[j] += temporary
        return self.fold(local)

    def solve(self, current, objective_change, tv_weight=.001, damping_relative=.01,
              maximum=64, trust=(.03, .04, .10), callback=None):
        """All trust shifts are solved inside the same coupled normal operator.

        objective_change(proposal) must evaluate the exact fixed-pressure
        equation-cost difference over every acquisition frequency. No linear
        surrogate is used as the acceptance objective.
        """
        tv, gtv, htv, tvlocal = self.vars.tv(current, tv_weight)
        unknown = self.vars.unknown
        gradient = (-self.rhs + gtv) * unknown
        right = -gradient
        reference = np.array([max(float(np.median(a[unknown][a[unknown] > 0]))
            if np.any(a[unknown] > 0) else 0., 1e-18) for a in self.diagonal])
        base = damping_relative * np.maximum(self.diagonal, reference[:, None, None] * 1e-4)
        base[:, ~unknown] = 1.
        target = np.asarray(trust, np.float64) / .1
        multipliers = np.zeros(3, np.float64)
        records = []

        def rms(q):
            return np.sqrt(np.mean(q[:, unknown] ** 2, axis=1))

        def preconditioner(shifts):
            damping = base + shifts[:, None, None] * unknown
            diagonal = np.maximum(self.diagonal + tvlocal[:3] + damping, 1e-30)
            scales = 1 / np.sqrt(diagonal)
            cross = self.local[3:] + tvlocal[3:]
            a = cross[0] * scales[0] * scales[1] / 1.00001
            b = cross[1] * scales[0] * scales[2] / 1.00001
            c = cross[2] * scales[1] * scales[2] / 1.00001
            determinant = np.maximum(1 + 2*a*b*c - a*a - b*b - c*c, 1e-12)
            inv = np.stack([1-c*c, 1-b*b, 1-a*a, b*c-a, a*c-b, a*b-c]) / determinant
            def apply(value):
                y = value * scales
                return np.stack([inv[0]*y[0] + inv[3]*y[1] + inv[4]*y[2],
                    inv[3]*y[0] + inv[1]*y[1] + inv[5]*y[2],
                    inv[4]*y[0] + inv[5]*y[1] + inv[2]*y[2]]) * scales
            return apply, damping, diagonal / np.maximum(inv[:3], 1e-30)

        def linear(shifts, warm=None, label=''):
            pre, damping, _ = preconditioner(shifts)
            def operator(q):
                return self.apply(q) + htv(q) + damping*q
            initial = np.zeros_like(right) if warm is None else warm
            residual = right - operator(initial) if warm is not None else right
            correction, history = pcg(operator, residual, pre, maximum=maximum,
                                       tolerance=.01, callback=callback)
            result = (initial + correction) * unknown
            residual = right - operator(result)
            relative = np.sqrt(max(dot(residual, pre(residual)), 0) /
                               max(dot(right, pre(right)), 1e-300))
            records.append({'label': label, 'dual_shifts': shifts.tolist(),
                'iterations': len(history), 'global_preconditioned_relative_residual': float(relative),
                'RMS_in_normalized_coordinates': rms(result).tolist()})
            return result, history

        direction, history = linear(multipliers, label='unconstrained_GN')
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
                high = max(multipliers[coordinate], effective_reference[coordinate] *
                    max(rms(at_zero)[coordinate]/target[coordinate] - 1, .1), 1e-20)
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
                    raise RuntimeError('Could not bracket a coupled RMS trust multiplier')
                warm = high_direction
                for iteration in range(12):
                    mid = (low + high)/2
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
        common_scale = min(1., 1/max(float(np.max(rms(direction)/target)), 1e-30))
        direction *= common_scale
        kind = 'coupled_RMS_constrained_pressure_GN'
        if dot(gradient, direction) >= 0:
            direction = -gradient / np.maximum(self.diagonal + tvlocal[:3] + base, 1e-30)
            direction *= min(1., 1/max(float(np.max(rms(direction)/target)), 1e-30))
            kind = 'common_scaled_diagonal_descent_fallback'
        accepted = current.copy()
        candidates = []
        for line in range(9):
            alpha = .5**line
            proposal = self.vars.project(current + alpha*.1*direction)
            actual_rms = rms(proposal-current)
            slope = dot(gradient, (proposal-current)/.1)
            if slope >= 0 or np.any(actual_rms > np.asarray(trust)*1.01):
                candidates.append({'alpha': alpha, 'slope': slope,
                    'actual_RMS': actual_rms.tolist(), 'accepted': False})
                continue
            equation = float(objective_change(proposal))
            tv_change = self.vars.tv(proposal, tv_weight)[0] - tv
            change = equation + tv_change
            ok = bool(np.isfinite(change) and change <= 1e-4*slope)
            candidates.append({'alpha': alpha, 'equation_cost_change': equation,
                'TV_change': tv_change, 'fixed_pressure_objective_change': change,
                'slope': slope, 'actual_RMS': actual_rms.tolist(), 'accepted': ok})
            if ok:
                accepted = proposal
                break
        return accepted, {'direction_kind': kind, 'CG_history': history, 'linear_solves': records,
            'raw_unconstrained_RMS_r_beta_t': (.1*raw_rms).tolist(),
            'trust_RMS_r_beta_t': list(trust), 'trust_dual_shifts': multipliers.tolist(),
            'common_safety_scale': common_scale, 'line_search': candidates,
            'accepted': bool(candidates and candidates[-1]['accepted']),
            'update_RMS_r_beta_t': rms(accepted-current).tolist(),
            'gradient_norm': np.sqrt(np.sum(gradient**2, axis=(-2, -1))).tolist(),
            'TV_before': tv, 'damping_reference_per_parameter': reference.tolist(),
            'truth_read': False, 'new_propagation_in_material_step': False}
