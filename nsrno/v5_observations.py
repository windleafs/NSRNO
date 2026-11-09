"""Resolve V5 observed IQ only; no material arrays are loaded by this module."""
from pathlib import Path
import json


AUGMENTATION_INDEX = Path('out/v5/uno_triparam_v1/augmentation_index.json')


def canonical_id(sample):
    return sample.rsplit('_', 1)[0] if sample.endswith(('_hf', '_lf')) else sample


def plane_observation(source, sample):
    if source not in ['lf', 'hf']:
        raise ValueError(source)
    index_path = Path('out/v5/forward_cache_3mhz_v1/index.json' if source == 'lf'
                      else 'out/v5/forward_cache_v1/index.json')
    index = json.loads(index_path.read_text())
    matches = [r for r in index['records'] if r['id'] == sample]
    if matches:
        if len(matches) != 1:
            raise ValueError(f'Ambiguous observed cache ID: {sample}')
        path = Path(matches[0]['iq'])
        return {'id': sample, 'iq': str((path if path.is_absolute() else index_path.parent/path).resolve()),
                'source_mhz': 3. if source == 'lf' else 7.5,
                'observation_index': str(index_path.resolve())}
    state = canonical_id(sample)
    augmented = json.loads(AUGMENTATION_INDEX.read_text())
    matches = [r for r in augmented['records']
               if r['id'] == f'{state}_{source}' and r['source_mhz'] == (3. if source == 'lf' else 7.5)]
    if len(matches) != 1:
        raise KeyError(f'Expected one {source} observation for {state}, found {len(matches)}')
    row = matches[0]
    if not row.get('original_time_solver'):
        raise ValueError('Expected the original time-domain observation')
    return {'id': state, 'record_id': row['id'], 'iq': str(Path(row['iq']).resolve()),
            'source_mhz': row['source_mhz'], 'angles_deg': row['angles_deg'],
            'absorption': row['absorption'], 'band_hz': row['band_hz'],
            'observation_index': str(AUGMENTATION_INDEX.resolve())}
