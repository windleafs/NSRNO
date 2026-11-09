"""Recover known attenuation and exact acquisition spectra, without PyTorch."""
from pathlib import Path
import json, hashlib
import numpy as np
from nsrno.fullwave import load_generator, build_case

ROOT=Path('out/v5/conditioned_wave_v1')
CACHE=Path('out/v5/forward_cache_v1')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'alpha').mkdir(exist_ok=True)
    gen=load_generator()
    selected=json.load(open('out/v5/low_frequency_3mhz_v1/index.json'))['records']
    high=json.load(open(CACHE/'index.json'))
    rows={r['id']:r for r in high['records']}
    manifest=json.load(open('out/v5/manifest.json'))
    originals={r['id']:r for r in manifest['records']}
    case=gen.geometry_case(); fit=gen.absorption_model(case)
    bins=[270,360,420]; angles=[0,5,10]; frequencies=np.array(bins)*40e6/2401
    spectra={}
    for angle in angles:
        cols=np.flatnonzero((case['x']>=case['xe'][0])&(case['x']<=case['xe'][-1]))
        raw=case['x'][cols]*np.sin(np.deg2rad(high['acquisition']['angles_deg'][angle]))/case['c0']
        case['delays']=raw-raw.min()
        _,rx,wave=gen.bench.source_arrays(case)
        times=np.arange(case['nt'])*case['dt']
        basis=np.exp(-2j*np.pi*times[:,None]*frequencies[None,:])*case['dt']/(2401/40e6)
        transformed=wave@basis
        for j,bin_index in enumerate(bins):
            line=np.zeros(len(case['x']),np.complex64); line[cols]=transformed[:,j]
            spectra[f'b{bin_index}_a{angle}']=line
    np.savez(ROOT/'sources.npz',**spectra)
    spec={**high['grid'],**high['acquisition'],**fit,'pml':case['pml'],
          'native_dt_s':case['dt'],'native_nt':case['nt'],'source_cols':cols.tolist(),
          'source_c_lookup_zx':[int(np.floor(np.float32(case['z'][case['face']]-case['z'][case['pml']])/case['dx'])),
              int(np.floor(np.float32(case['x'][cols[0]]-case['x'][case['pml']])/case['dx']))],
          'bins':bins,'angle_indices':angles,'known_alpha':True}
    (ROOT/'spec.json').write_text(json.dumps(spec,indent=2))
    records=[]
    for selected_row in selected:
        row=rows[selected_row['id']]; path=ROOT/'alpha'/(row['id']+'.npy')
        if not path.exists():
            maps=build_case(gen,originals[row['id']])['maps']
            medium=np.load(CACHE/row['medium'],mmap_mode='r')
            if not (np.array_equal(medium[0],maps['sound_speed']) and np.array_equal(medium[1],maps['density'])):
                raise ValueError('Regenerated coefficients mismatch: '+row['id'])
            np.save(path,maps['alpha_coeff'].astype(np.float32))
        records.append({**row,'alpha':str(path.resolve()),'medium_exact_match_checked':True})
        print(json.dumps({'alpha_prepared':len(records),'id':row['id']}),flush=True)
    result={'spec':spec,'records':records,'source_cache':str(CACHE.resolve()),
            'generator_sha256':hashlib.sha256(Path(gen.__file__).read_bytes()).hexdigest(),
            'conditioning':'alpha explicitly supplied; eta depends analytically on c; no target wavefield input'}
    (ROOT/'index.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
