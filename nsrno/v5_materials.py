"""Truth arrays for generators and independent evaluation; not for optimization."""
from pathlib import Path
import json
import numpy as np
from .v5_observations import canonical_id, AUGMENTATION_INDEX


def material_paths(sample, explicit=None):
    if explicit is not None:
        return [Path(explicit)]
    sample = canonical_id(sample)
    medium = Path(f'out/v5/forward_cache_v1/{sample}.medium.npy')
    if medium.exists():
        return [medium, Path(f'out/v5/uno_triparam_v1/alpha/{sample}.npy')]
    index = json.loads(AUGMENTATION_INDEX.read_text())
    paths = {Path(r['material3']).resolve() for r in index['records']
             if canonical_id(r['id']) == sample}
    if len(paths) != 1:
        raise KeyError(f'Expected one evaluation material for {sample}, found {len(paths)}')
    return list(paths)


def evaluation_material(sample, explicit=None):
    paths = material_paths(sample, explicit)
    if len(paths) == 2:
        result = np.concatenate([np.load(paths[0]), np.load(paths[1])[None]], 0)
    else:
        result = np.load(paths[0])
    if result.shape != (3, 961, 1121) or not np.isfinite(result).all():
        raise ValueError('Expected finite absolute c/rho/alpha on the native V5 grid')
    return result
