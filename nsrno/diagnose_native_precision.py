"""Distinguish changed base algebra from native floating-point repeatability."""
from pathlib import Path
from argparse import ArgumentParser
import json, time, hashlib
import numpy as np
import torch
from .time_acquisition import TimeAcquisition
from .native_time import native_wave


def main():
    p = ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('out/v5/native_tangent_v1'))
    p.add_argument('--device', type=int, default=0)
    a = p.parse_args()
    dest = a.out/'precision_review.json'
    if dest.exists():
        raise FileExistsError('Preserve measured precision review')
    original = json.loads((a.out/'run.json').read_text())
    machine = json.loads((a.out/'base_kernel_comparison.json').read_text())
    if not all(r['same_machine_code'] for r in machine['kernels']):
        raise ValueError('Base machine code changed; diagnose that first')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(a.device)
    torch.cuda.set_per_process_memory_fraction(1.5*2**30/
        torch.cuda.get_device_properties(a.device).total_memory, a.device)
    device = f'cuda:{a.device}'
    ac = TimeAcquisition(original['source'], original['sample'], device=device)
    signs = np.asarray(original['signs'], np.float32)
    source = torch.from_numpy(np.tensordot(signs, np.load(original['source_bank'], mmap_mode='r'), axes=(0,0))).to(device)
    reference = torch.from_numpy(np.tensordot(signs, ac.reference, axes=(0,0))).to(device)
    material = torch.from_numpy(np.load(original['material'])).to(device).requires_grad_(True)
    start = time.time()
    # Same original library, same inputs. Only checkpoint recording differs.
    raw_checkpoint = native_wave(material, source, ac.si, ac.sw, ac.ri, ac.rw, ac.config)
    with torch.no_grad():
        raw_plain = native_wave(material, source, ac.si, ac.sw, ac.ri, ac.rw, ac.config)
        r0 = (ac.response @ (raw_checkpoint.detach()-reference)).T
        r1 = (ac.response @ (raw_plain-reference)).T
        raw_error = float((raw_checkpoint.detach()-raw_plain).norm()/raw_checkpoint.detach().norm())
        rf_error = float((r0-r1).norm()/r0.norm())
    forward_limit = min(1e-4, max(2e-6, 5*rf_error))
    qualified = (original['tangent_adjoint_relative_error'] < 3e-4 and
                 original['base_prediction_relative_rms_difference'] <= forward_limit)
    result = {'purpose': 'numeric precision review; original failed record preserved',
              'truth_used': False, 'base_kernel_machine_code_identical': True,
              'same_original_library_checkpoint_vs_plain_raw_relative_rms': raw_error,
              'same_original_library_checkpoint_vs_plain_RF_relative_rms': rf_error,
              'tangent_base_vs_original_RF_relative_rms': original['base_prediction_relative_rms_difference'],
              'tangent_adjoint_relative_error': original['tangent_adjoint_relative_error'],
              'forward_tolerance': forward_limit,
              'tolerance_rule': 'max(2e-6, 5 times original-library measured RF repeatability), capped at 1e-4',
              'qualified_for_curvature_implementation': qualified,
              'seconds': time.time()-start,
              'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
              'interpretation': 'Identical arithmetic kernels with changed launch/checkpoint ordering; atomic source-add roundoff is a candidate mechanism, not a material recovery claim',
              'sources_sha256': {n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                  for n in ['native_time.cu','native_tangent.cu','native_tangent.py']}}
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
