"""Atomic JSON for numerical reports, preserving scalar types as JSON values."""
import json
import numpy as np


def numeric_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported report value: {type(value).__name__}')


def atomic(path,value):
    content=json.dumps(value,indent=2,default=numeric_scalar)+'\n'
    temporary=path.with_suffix('.partial.json')
    temporary.write_text(content)
    temporary.replace(path)
