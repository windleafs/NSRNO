"""CPU algebra/shape checks for the new RF statistics, not patient validation."""
import json
import torch
from .angular_spectral_features import angular_projection,BANDS


def main():
    mu=torch.tensor([-.99,-.88,-.67,-.31],dtype=torch.float64)
    w=torch.tensor([1.,2.,.5,1.5],dtype=torch.float64)
    a=torch.tensor([1+.2j,-.3+2j,.7-.4j],dtype=torch.complex128)[:,None]
    b=torch.tensor([.4-.8j,1+.3j,-.2+.1j],dtype=torch.complex128)[:,None]
    values=a+b*mu
    w0=w.sum()[None];w1=(w*mu).sum()[None];w2=(w*mu.square()).sum()[None]
    s0=(values*w).sum(-1)[:,None];s1=(values*w*mu).sum(-1)[:,None]
    energy=(values.abs().square()*w).sum(-1)[:,None]
    mean,contrast,*_=angular_projection(s0,s1,energy,w0,w1,w2,w0)
    center=w1/w0;var=w2/w0-center.square()
    mean_error=float((mean-(a+b*center)).abs().max())
    contrast_error=float((contrast-b*var/(var+1e-6).sqrt()).abs().max())
    constant=a.expand_as(values)
    c0=(constant*w).sum(-1)[:,None];c1=(constant*w*mu).sum(-1)[:,None]
    ce=(constant.abs().square()*w).sum(-1)[:,None]
    constant_contrast=float(angular_projection(c0,c1,ce,w0,w1,w2,w0)[1].abs().max())
    rng=torch.Generator().manual_seed(61107);signal=torch.randn(2401,generator=rng,dtype=torch.float64)
    ft=torch.fft.fft(signal);frequency=torch.fft.fftfreq(2401,1/40e6,dtype=torch.float64)
    spectral_errors={}
    for f0,bands in BANDS.items():
        masks=torch.stack([(frequency>=lo)&(frequency<hi) for lo,hi in bands])
        bandwise=torch.fft.ifft(ft[None]*2*masks).sum(0)
        full=torch.fft.ifft(ft*2*((frequency>=bands[0][0])&(frequency<bands[-1][1])))
        spectral_errors[str(f0)]=float((bandwise-full).abs().max())
    assert mean_error<1e-12 and contrast_error<1e-12 and constant_contrast<1e-12
    assert max(spectral_errors.values())<1e-12
    print(json.dumps({'CPU_only':True,'GPU_used':False,'patient_wave_solves':0,
        'mean_absolute_error':mean_error,'angular_component_absolute_error':contrast_error,
        'constant_signal_angular_component':constant_contrast,
        'disjoint_band_recombination_absolute_error':spectral_errors,
        'scientific_parameter_separability_claimed':False},indent=2))


if __name__=='__main__':main()
