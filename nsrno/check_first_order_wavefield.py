"""One necessary implementation qualification, not a material probe.

The new field system must equal the already established pressure equation,
and its field/preconditioner adjoints must be correct before a WRI solve.
No new wave simulation, RF perturbation, material derivative or truth access.
"""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import hashlib,json,time,resource
import numpy as np
import torch
from .conditioned_wave import FrequencyGeometry,ConditionedFrequencySystem,LearnedPreconditioner
from .dense_wave import learned_apply,projected_operator
from .first_order_wavefield import FirstOrderWavefield


def main():
    start=time.time();torch.set_num_threads(4);torch.manual_seed(20261106)
    ref=Path('out/v5/conditioned_lf_full_v1')
    progress=json.loads((ref/'progress.json').read_text())
    if '0:0:32' not in progress['completed']:raise RuntimeError('First actual reference block is not ready')
    index=json.loads(Path('out/v5/forward_cache_3mhz_v1/index.json').read_text())
    low=json.loads(Path('out/v5/low_frequency_3mhz_v1/index.json').read_text())
    meta=json.loads(Path('out/v5/time_triparam_v1/geometry.json').read_text())
    spec={**index['grid'],**index['acquisition'],**low['absorption'],'native_dt_s':meta['dt'],'pml':meta['pml']}
    geometry=FrequencyGeometry(spec)
    material=np.load('out/v5/reconstruction_geometry_v1_train_200/snapshots/stage0_step064/material.npy')
    system=ConditionedFrequencySystem(geometry,torch.from_numpy(material[:2]),torch.from_numpy(material[2]),91*40e6/2401)
    operator=FirstOrderWavefield(system)
    pressure=torch.from_numpy(np.array(np.load(ref/'reference/a00.npy',mmap_mode='r')[0]))
    pressure=geometry.pad_field(pressure)
    eliminated=operator(operator.from_pressure(pressure))[0]
    existing=projected_operator(system,pressure)
    schur_error=float((eliminated-existing).norm()/existing.norm().clamp_min(1e-30))
    def random_field():
        return torch.complex(torch.randn(3,geometry.H,geometry.W),torch.randn(3,geometry.H,geometry.W))*geometry.mask
    x,y=random_field(),random_field()
    def dot(a,b):return (a.to(torch.complex128).conj()*b.to(torch.complex128)).sum()
    with torch.no_grad():
        left,right=dot(operator(x),y),dot(x,operator.adjoint(y))
        adjoint_error=float(abs(left-right)/max(abs(left),abs(right),1e-30))
        checkpoint=torch.load('out/v5/forward_round8_dense_full/last.pt',map_location='cpu',weights_only=True)
        network=LearnedPreconditioner();network.load_state_dict(checkpoint['model']);network.requires_grad_(False);network.eval()
        forward,adjoint=operator.preconditioners(learned_apply(network,system),learned_apply(network,system,adjoint=True))
        left,right=dot(forward(x),y),dot(x,adjoint(y))
        preconditioner_error=float(abs(left-right)/max(abs(left),abs(right),1e-30))
    report={'qualified_for_actual_wavefield_inversion':max(schur_error,adjoint_error,preconditioner_error)<5e-4,
            'Schur_vs_existing_pressure_relative_error':schur_error,'field_adjoint_relative_error':adjoint_error,
            'block_preconditioner_adjoint_relative_error':preconditioner_error,
            'GPU_memory_GiB':0,'unknown_truth_read':False,'new_wave_simulations':0,
            'material_derivative_probes':0,'frequency_hz':system.frequency,
            'scope':'implementation equivalence and field adjoints only; NOT three-parameter imaging',
            'cpu_peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'sources_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                              for n in ['first_order_wavefield.py',Path(__file__).name]},
            'elapsed_seconds':time.time()-start}
    (ref/'first_order_qualification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    if not report['qualified_for_actual_wavefield_inversion']:raise RuntimeError('First-order implementation not qualified')


if __name__=='__main__':main()
