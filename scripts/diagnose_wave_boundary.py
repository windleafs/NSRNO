import json
from pathlib import Path
import torch
from nsrno.wave_operator import WaveResidualOperator
from nsrno.train_wave import sample
from nsrno.dataset import read_json,write_json

torch.set_num_threads(2); cache=Path('out/v5/forward_cache_v1'); index=read_json(cache/'index.json')
row=next(r for r in index['records'] if r['id']=='train_200')
spec={**index['grid'],'angles_deg':index['acquisition']['angles_deg'],'source_tref_s':row['source_tref_s']}
medium,_=sample(row,cache,5,360); results=[]
with torch.no_grad():
    for kind in ('shifted','free_space'):
        model=WaveResidualOperator(spec,8,kind).cuda(); data,gain,multiplier=model.setup(medium,6e6)
        ax,az,b,k0=data; inc=model.incident(6e6,5)
        rhs=-model.apply_operator(inc,(ax-1,az-1,b-(1-1j*model.sponge),k0))
        direction=gain*torch.fft.ifft2(multiplier*torch.fft.fft2(rhs,s=multiplier.shape))[:model.H,:model.W]
        boundary=model.sponge>0
        # Diagnose boundary incompatibility, including an analytical sponge taper.
        for tapered in (False,True) if kind=='free_space' else (False,):
            d=direction
            if tapered: d=d*torch.exp(-k0*48*model.dx/3*(model.sponge/2).pow(1.5))
            image=model.apply_operator(d,data); inner=(image.conj()*rhs).sum()
            cosine=inner.abs()/(image.abs().norm()*rhs.abs().norm())
            alpha=inner/image.abs().square().sum()
            result=dict(kind=kind,tapered=tapered,
                residual_energy_in_sponge=float(rhs[boundary].abs().square().sum()/rhs.abs().square().sum()),
                image_energy_in_sponge=float(image[boundary].abs().square().sum()/image.abs().square().sum()),
                alignment=float(cosine),step_magnitude=float(alpha.abs()),
                residual_after_step=float((rhs-alpha*image).abs().norm()/rhs.abs().norm()))
            results.append(result); print(json.dumps(result),flush=True)
write_json(Path('out/v5/forward_round5_wave_free_v1/boundary_diagnosis.json'),results)
