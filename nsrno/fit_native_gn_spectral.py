"""Fit a nonlocal CPU metric and assess a held-out, existing native HVP."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import hashlib
import json
import resource
import time
import numpy as np
from .native_gn_spectral import SpectralDataMetric,fit_symbol
from .native_gn_coordinates import dot


def assess(metric,folder):
    d=np.load(folder/'direction.npy')
    data=np.load(folder/'data_image.npy')
    total=np.load(folder/'total_image.npy')
    prediction=metric.apply(d)
    delta=prediction-data
    true_data=dot(d,data)
    predicted_data=dot(d,prediction)
    prior=dot(d,total-data)
    row=json.loads((folder/'progress.json').read_text())
    return {'checkpoint':str(folder),'native_HVP_number':row['hvp_count'],
        'true_data_curvature':true_data,'predicted_data_curvature':predicted_data,
        'predicted_to_true_data_curvature':predicted_data/true_data,
        'data_image_relative_error':np.sqrt(max(dot(delta,delta),0)/max(dot(data,data),1e-300)),
        'prior_curvature':prior,'predicted_total_curvature':predicted_data+prior,
        'true_total_curvature':true_data+prior,
        'existing_previous_metric_curvature':row.get('compact_secant',{}).get('sBs'),
        'predicted_to_true_total_curvature':(predicted_data+prior)/(true_data+prior)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--train-count',type=int,default=10)
    p.add_argument('--heldout-number',type=int,default=11)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ridge',type=float,default=.01)
    args=p.parse_args()
    if args.out.exists():raise RuntimeError('New output only; no implicit reruns')
    started=time.time()
    manifest=args.source_root/'step001/fixed_HVP_manifest.json'
    paths=[Path(p) for p in json.loads(manifest.read_text())['paths']]
    if not 0<args.train_count<args.heldout_number<=len(paths):raise ValueError('Need a genuinely held-out complete checkpoint')
    training=paths[:args.train_count]
    heldout=paths[args.heldout_number-1]
    symbol,fit=fit_symbol(training,args.ridge)
    shape=np.load(training[0]/'direction.npy',mmap_mode='r').shape
    mask=np.broadcast_to(np.arange(shape[1])[:,None]>=80,shape[1:])
    metric=SpectralDataMetric(symbol,mask)
    check=assess(metric,heldout)
    a=np.load(training[0]/'direction.npy')
    b=np.load(heldout/'direction.npy')
    Aa=metric.apply(a)
    Ab=metric.apply(b)
    denominator=np.sqrt(max(dot(a,Aa)*dot(b,Ab),1e-300))
    symmetry_error=abs(dot(a,Ab)-dot(Aa,b))/denominator
    if symmetry_error>1e-10 or min(dot(a,Aa),dot(b,Ab))<=0:
        raise RuntimeError('Fitted FFT implementation lost self-adjointness or positivity')
    args.out.mkdir(parents=True)
    np.save(args.out/'symbol.npy',symbol)
    report={'kind':'learned_nonlocal_data_normal_metric_from_existing_native_HVPs',
        'training_paths':[str(p) for p in training],'held_out':check,'fit':fit,
        'implementation_symmetry_relative_error':symmetry_error,
        'symbol_SHA256':hashlib.sha256((args.out/'symbol.npy').read_bytes()).hexdigest(),
        'source_files_SHA256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['native_gn_spectral.py','fit_native_gn_spectral.py']},
        'fixed_material_reference':json.loads((args.source_root/'fixed_model_reference.json').read_text()),
        'GPU_used':False,'new_patient_wave_solves':0,'material_truth_read':False,
        'new_reconstruction_performed':False,'seconds':time.time()-started,
        'peak_CPU_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
