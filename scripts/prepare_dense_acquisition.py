"""Exact dense source spectra and explicit angle/frequency coverage metadata."""
from pathlib import Path
from argparse import ArgumentParser
import json,hashlib
import numpy as np
from scipy import fft
from nsrno.fullwave import load_generator

p=ArgumentParser(); p.add_argument('--out',type=Path,default=Path('out/v5/conditioned_dense_v1'))
p.add_argument('--last-bin',type=int,default=450); args=p.parse_args()
root=args.out; root.mkdir(parents=True,exist_ok=True)
old=json.load(open('out/v5/conditioned_wave_v1/index.json'))
gen=load_generator(); case=gen.geometry_case(); bins=np.arange(241,args.last_bin+1)
values=np.zeros((len(bins),11,len(case['x'])),np.complex64)
for a,angle in enumerate(gen.ANGLES):
    cols=np.flatnonzero((case['x']>=case['xe'][0])&(case['x']<=case['xe'][-1]))
    delay=case['x'][cols]*np.sin(np.deg2rad(angle))/case['c0']; case['delays']=delay-delay.min()
    _,_,wave=gen.bench.source_arrays(case)
    # n=24010 native samples has exactly the2401/40MHz RF duration.
    spectrum=fft.rfft(wave,n=24010,axis=-1,norm='forward',workers=1)
    values[:,a,cols]=spectrum[:,bins].T
np.save(root/'sources.npy',values)
spec={**old['spec'],'bins':bins.tolist(),'angle_indices':list(range(11)),
      'dense_band_hz':[float(bins[0]*40e6/2401),float(bins[-1]*40e6/2401)],
      'frequency_spacing_hz':40e6/2401,'reference_format':'reference/aXX.npy[freq,z,x]',
      'source_f0_hz':7.5e6}
index={**old,'spec':spec,'dense_acquisition_pairs':len(bins)*11,'frequency_coverage':'all adjacent RF bins in explicit spec; includes nominal passband and specified rolloff',
       'source_spectrum_method':'native400MHz FFT padded to24010, norm forward',
       'source_sha256':hashlib.sha256((root/'sources.npy').read_bytes()).hexdigest()}
(root/'index.json').write_text(json.dumps(index,indent=2)); (root/'spec.json').write_text(json.dumps(spec,indent=2))
print(json.dumps({'bins':len(bins),'angles':11,'pairs':len(bins)*11,'sources_shape':list(values.shape)}))
