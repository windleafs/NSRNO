"""Warm receiver-dual continuation with a fixed shifted objective.

A changed material requires one actual current-K action on the old dual.
Only direction/image pairs carrying the current operator key are recycled.
"""
import math
import torch
from .recycled_receiver_solver import inner, solve_shifted


@torch.no_grad()
def solve_warm_recycled(K, rhs, shift, space, operator_key, warm_y, *,
        maximum_actions=12, relative_tolerance=.01, checkpoint=None):
    space.check(operator_key)  # Must precede the first expensive action.
    if not isinstance(maximum_actions, int) or maximum_actions < 1:
        raise ValueError('Budget includes the current-K warm-start action')
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError('A positive unchanged source penalty is required')
    if (warm_y.shape != rhs.shape or warm_y.dtype != rhs.dtype or
            warm_y.device != rhs.device or not rhs.is_complex() or
            not bool(torch.isfinite(rhs).all()) or
            not bool(torch.isfinite(warm_y).all()) or
            float(inner(warm_y, warm_y)) <= 1e-30):
        raise ValueError('A finite nonzero saved dual in the same receiver space is required')
    warm_y = warm_y.detach().clone()
    calls = 1
    Ky = K(warm_y, 1)
    if Ky.shape != rhs.shape or not bool(torch.isfinite(Ky).all()):
        raise RuntimeError('Invalid actual current-operator warm-start image')
    # This is a necessary initial-residual action, not an extra spectral probe.
    space.add(warm_y, Ky, operator_key)

    def emit(x, image, residual, actions):
        if checkpoint is not None:
            checkpoint({'y': x, 'accumulated_Ky': image, 'residual': residual,
                'completed_actions': actions, 'lambda_squared': shift,
                'operator_key': operator_key,
                'accumulated_image_is_not_final_native_field': True})

    emit(warm_y, Ky, rhs-Ky-shift*warm_y, 1)

    def action(value, index):
        nonlocal calls
        if calls >= maximum_actions or index+1 != calls+1:
            raise RuntimeError('Warm continuation actual-action accounting mismatch')
        calls += 1
        return K(value, calls)

    def save(packet):
        emit(packet['x'], packet['Kx'], packet['residual'], 1+packet['actions'])

    result = solve_shifted(action, rhs, shift, space, operator_key,
        initial=warm_y, initial_K=Ky, maximum_actions=maximum_actions-1,
        relative_tolerance=relative_tolerance, checkpoint=save)
    if calls != 1+result['actions'] or calls > maximum_actions:
        raise RuntimeError('Do not hide or replay a current-K action')
    emit(result['x'], result['accumulated_Kx'],
        rhs-result['accumulated_Kx']-shift*result['x'], calls)
    return {**result, 'actual_K_actions': calls, 'maximum_actual_K_actions': maximum_actions,
        'inner_CG_actions': result['actions'], 'fixed_lambda_squared': shift,
        'warm_image_recomputed_with_current_operator': True,
        'old_material_K_images_reused': False}

