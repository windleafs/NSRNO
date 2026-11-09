"""Depth primitive chart for a local coefficient on the unknown tissue rows.

C(a)=cumsum(a)/H and D(q)=H*diff(q), with the upper coupling primitive zero.
This is an invertible output coordinate, not an exact physical-ray model.
"""
import torch


def mean_depth_primitive(coefficient):
    if coefficient.shape[-2] < 1:
        raise ValueError('A nonempty unknown depth interval is required')
    return coefficient.cumsum(-2) / coefficient.shape[-2]


def coefficient_from_mean_primitive(primitive):
    if primitive.shape[-2] < 1:
        raise ValueError('A nonempty unknown depth interval is required')
    return primitive.shape[-2] * torch.cat(
        (primitive[..., :1, :],
         primitive[..., 1:, :] - primitive[..., :-1, :]), -2)
