"""Full native RF score for saved materials; no target material arrays read."""
from pathlib import Path
import argparse,json,gc,time,subprocess
import numpy as np
import torch
from .time_acquisition import TimeAcquisition
from .rf_depth_weights import DepthWeights,WEIGHTS


def main():
    p=argparse.ArgumentParser();p.add_argument('material',type=Path)
    p.add_argument('--sample',default='train_200');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',type=int,default=0);p.add_argument('--cap-gib',type=float,default=1.1)
    a=p.parse_args();torch.set_num_threads(1)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(s.split(',')[0]):int(s.split(',')[1]) for s in rows.splitlines()}
    if used[a.device]+a.cap_gib*1024+768>16*1024:raise RuntimeError('Native RF score does not fit the whole-card budget')
    torch.cuda.set_device(a.device)
    torch.cuda.set_per_process_memory_fraction(a.cap_gib*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
    device=f'cuda:{a.device}';material=torch.from_numpy(np.load(a.material)).to(device)
    report={'material':str(a.material),'sample':a.sample,'truth_read':False,'all_shots':11,'receivers':192,
            'selection':'primary: mean LF+point full retained RF energy-normalized squared error; HF and geometry diagnostic',
            'groups':{},'device':a.device,'cap_GiB':a.cap_gib};start=time.time()
    with torch.no_grad():
        for source in ['point3','lf','hf']:
            acquisition=TimeAcquisition(source,a.sample,device=device,chunk=96)
            depth=DepthWeights(acquisition);energy=float(acquisition.observed.double().square().sum())
            error=0.;geom=0.;bins=np.zeros(5)
            for shot in range(11):
                difference=acquisition.predict(material,shot)-acquisition.observed[shot]
                error+=float(difference.double().square().sum())
                terms=depth.terms(difference,shot);values=np.array([float(t) for t in terms])
                geom+=.5*float(np.dot(WEIGHTS,values));bins+=values
                del difference,terms
            report['groups'][source]={'relative_RMS':float(np.sqrt(error/energy)),
                'normalized_error_squared':error/energy,'geometry_loss':geom,
                'geometry_window_relative_RMS':np.sqrt(bins).tolist(),'observed_energy':energy}
            del acquisition,depth;gc.collect();torch.cuda.empty_cache()
    report['primary_loss']=.25*sum(report['groups'][s]['normalized_error_squared'] for s in ['point3','lf'])
    report['three_group_global_loss']=sum(v['normalized_error_squared'] for v in report['groups'].values())/6
    report['three_group_geometry_loss']=sum(v['geometry_loss'] for v in report['groups'].values())/3
    report['peak_GiB']=torch.cuda.max_memory_allocated()/2**30;report['elapsed_seconds']=time.time()-start
    a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
