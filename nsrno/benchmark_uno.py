"""Measure the exported trained interface, including complete11-angle gradients."""
from pathlib import Path
import os,json,time,threading,subprocess
import numpy as np
import torch
from nsrno.uno_surrogate import UNOSurrogate
from nsrno.uno_data import atomic_json

def main():
    out=Path('out/v5/uno_round1_triparam'); started=time.monotonic()
    while not (out/'postprocess_complete.json').exists():
        if (out/'postprocess_error.json').exists(): raise RuntimeError((out/'postprocess_error.json').read_text())
        if time.monotonic()-started>14400: raise TimeoutError('Exported checkpoint unavailable')
        time.sleep(10)
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(14*2**30/torch.cuda.get_device_properties(0).total_memory)
    model=UNOSurrogate(checkpoint=out/'model_weights.pt')
    root=Path('out/v5/uno_triparam_v1'); rows=json.load(open(root/'index.json'))['records']
    observations=[]
    for source in [7.5,3.]:
        matched=[r for r in rows if r['parent_id']=='train_200' and r['source_mhz']==source]
        if matched: observations.append({'source_mhz':source,'iq':torch.from_numpy(np.load(matched[0]['iq'])).cuda()})
    z=torch.arange(961,device='cuda')*5e-5-.003; edge=torch.from_numpy(np.load(root/'known_edge_alpha.npy')).cuda()[None]
    alpha=torch.full((961,1121),.5,device='cuda'); alpha[z<.001-1e-7]=.002
    m=torch.stack((torch.full_like(alpha,1540),torch.full_like(alpha,1000),alpha+edge))
    alive=[True]; samples=[]
    def sample():
        while alive[0]:
            try:
                text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
                for line in text.splitlines():
                    pid,mem=line.split(',')
                    if int(pid)==os.getpid(): samples.append(float(mem)/1024)
            except Exception: pass
            time.sleep(.1)
    thread=threading.Thread(target=sample,daemon=True); thread.start()
    with torch.no_grad(): value=model(m)
    if value.shape!=(11,192,2401): raise RuntimeError('Exported RF shape mismatch')
    times=[]; torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(5):
            torch.cuda.synchronize(); tic=time.perf_counter(); value=model(m); torch.cuda.synchronize(); times.append(time.perf_counter()-tic)
    forward={'median_seconds':float(np.median(times)),'all_seconds':times,'allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    del value
    times=[]; torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        torch.cuda.synchronize(); tic=time.perf_counter(); objective,gradient=model.objective_and_gradient(m,observations)
        torch.cuda.synchronize(); times.append(time.perf_counter()-tic)
    norms=[float(gradient[k].norm()) for k in range(3)]
    if not bool(torch.isfinite(gradient).all()): raise RuntimeError('Exported model gradient nonfinite')
    material_gradient={'median_seconds':float(np.median(times)),'all_seconds':times,'allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'reserved_gib':torch.cuda.max_memory_reserved()/2**30,'input_gradient_norms':norms,'sources':len(observations),'angles_per_source':11}
    time.sleep(.25); alive[0]=False; thread.join(timeout=2)
    result={'device':torch.cuda.get_device_name(0),'checkpoint_step':model.checkpoint_step,'pure_neural':True,'physical_solver_used':False,
        'forward_full11':forward,'complete_three_parameter_gradient':material_gradient,
        'sampled_process_peak_gib':max(samples,default=0),'uniform_material_input_not_true_alpha':True,'native_shape':[3,961,1121]}
    atomic_json(out/'trained_benchmark.json',result); print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
