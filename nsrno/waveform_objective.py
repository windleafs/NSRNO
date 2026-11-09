"""Phase-insensitive pulse arrival information for early FWI stages.

The reference-normalized envelope energy keeps absolute amplitudes. The final
stages use the original RF residual; envelope fitting alone is not inversion.
"""
import torch


def smooth_envelope(rf,sigma_us):
    n=rf.shape[-1]
    spectrum=torch.fft.fft(rf,dim=-1)
    h=torch.zeros(n,device=rf.device,dtype=rf.dtype);h[0]=1
    if n%2==0:h[1:n//2]=2;h[n//2]=1
    else:h[1:(n+1)//2]=2
    analytic=torch.fft.ifft(spectrum*h,dim=-1)
    envelope=torch.sqrt(analytic.real.square()+analytic.imag.square()+1e-20)
    # Zero-padding avoids treating the last and first arrival as adjacent.
    pad=max(1,round(5*sigma_us*40));padded=torch.nn.functional.pad(envelope,(pad,pad))
    frequency=torch.fft.rfftfreq(padded.shape[-1],1/40e6,device=rf.device)
    kernel=torch.exp(-.5*(2*torch.pi*frequency*sigma_us*1e-6).square())
    smooth=torch.fft.irfft(torch.fft.rfft(padded,dim=-1)*kernel,n=padded.shape[-1],dim=-1)
    return smooth[...,pad:pad+n]
