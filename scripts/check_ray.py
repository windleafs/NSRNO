"""Bounded checks of the new coordinate/phase implementation and one real record."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from nsrno.ray_propagation import RayPropagation

torch.set_num_threads(2)
torch.set_float32_matmul_precision("high")
spec = dict(medium_shape=[2,81,101], dx_m=.0002, x0_m=-.01, z0_m=-.002,
            receivers=8, pitch_m=.0002, angles_deg=[-8.,0.,8.], source_tref_s=[2e-6,1e-6,2e-6],
            ray_receiver_chunk=4)
operator = RayPropagation(spec).cuda()
medium = torch.empty((1,2,81,101), device="cuda")
medium[:,0] = 1500.
medium[:,1] = 1020.
medium.requires_grad_(True)
angles=torch.arange(3,device="cuda")
receivers=torch.arange(8,device="cuda")
rx = operator.receive_time(medium, receivers)
z,x=operator.z[None,:,None],operator.x[None,None,:]
expected = torch.sqrt(z.square()+(x-operator.rx[:,None,None]).square()).clamp_min(operator.dx/2)/1500.
rx_error=float((rx[0]-expected)[:,1:].abs().max())
tx=operator.transmit_time(medium, angles)
theta=operator.theta[:,None,None]
expected_tx=operator.tref[:,None,None]+(x*theta.sin()+z*theta.cos())/1540.+z/theta.cos()*(1/1500.-1/1540.)
tx_error=float((tx[0]-expected_tx).abs().max())
gradient=torch.autograd.grad(rx[:, :, 20:70, 20:80].sum()+tx[:, :, 20:70, 20:80].sum(),medium)[0]
assert rx_error < 1e-9 and tx_error < 1e-9
assert torch.isfinite(gradient).all() and gradient[:,0].abs().sum() > 0

# Move a single event by exactly one deposition-grid sample: all retained
# frequency bins must acquire their physical (carrier + baseband) phase shift.
source=torch.tensor([[[[1.]],[[.3]]]],device="cuda")
geometry=torch.ones(1,2,1,1,device="cuda")
dt=operator.duration/operator.nt
t=torch.tensor([[[[128*dt]]]],device="cuda")
a=operator._deposit(source,t,geometry)
b=operator._deposit(source,t+dt,geometry)
ca=torch.complex(a[:,:,0],a[:,:,1]); cb=torch.complex(b[:,:,0],b[:,:,1])
fa=torch.fft.fft(ca); fb=torch.fft.fft(cb)
frequency=torch.fft.fftfreq(640,d=operator.duration/640,device="cuda")+operator.fc
ratio=torch.exp(-2j*torch.pi*frequency*dt)
phase_error=float(((fb-fa*ratio).abs()/fa.abs().clamp_min(1e-8)).max())
assert phase_error < 1e-3
del medium,rx,tx,gradient,operator

root=Path('out/v5/forward_cache_v1')
index=json.loads((root/'index.json').read_text())
row=index['records'][0]
run=json.loads(Path('out/v5/forward_round3_slowness/run.json').read_text())
operator=RayPropagation(run['spec']).cuda()
medium=torch.from_numpy(np.load(root/row['medium'])).cuda()[None]
results={'uniform_receive_max_error_s':rx_error,'uniform_transmit_max_error_s':tx_error,
         'one_bin_delay_phase_relative_error':phase_error,'coordinate_input_gradient_finite_nonzero':True}
for mode in (False,True):
    torch.cuda.synchronize(); start=time.perf_counter()
    with torch.no_grad():
        features=operator(medium,angles=[5],receivers=list(range(0,192,24)),material=mode)
    torch.cuda.synchronize()
    results['material' if mode else 'reference']={
        'seconds_one_angle_eight_receivers':time.perf_counter()-start,
        'shape':list(features.shape),'rms':float(features.square().mean().sqrt()),
        'maximum':float(features.abs().max()),
        'float16_relative_rms':float((features-features.half().float()).square().sum().sqrt()/features.square().sum().sqrt())}
    assert torch.isfinite(features).all()
print(json.dumps(results,indent=2),flush=True)
Path('out/v5/forward_round4_checks').mkdir(exist_ok=True)
Path('out/v5/forward_round4_checks/implementation.json').write_text(json.dumps(results,indent=2)+'\n')
