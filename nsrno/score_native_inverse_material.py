"""All-shot native RF score of a fixed inferred material, without optimization."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import traceback
import numpy as np
from .reconstruct_native_gn import Pool,source_manifest
from .native_gn_json import atomic
from .pressure_resources import PressureResources
from .refine_native_gn_cpu import save_array


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--material',type=Path,required=True)
    p.add_argument('--reference-material',type=Path,
        help='Optional previous inferred material, scored sequentially against the same observations')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--after-root',type=Path,required=True)
    p.add_argument('--sample',default='train_200')
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=8.)
    args=p.parse_args();args.out=args.out.resolve()
    prior=json.loads((args.after_root/'run.json').read_text())
    if prior['status']!='feature_training_and_evaluation_finished':
        raise RuntimeError('Complete fixed-checkpoint evaluation before physical RF scoring')
    for pid in [prior['pid'],prior.get('child_pid')]:
        if pid is not None and Path(f'/proc/{pid}').exists():
            raise RuntimeError(f'Previous GPU pipeline process {pid} still exists')
    if args.out.exists():raise RuntimeError('Inspect existing score; no implicit replay')
    if not 0<args.cap_gib<=10:raise ValueError('Keep full context below 16 GiB per card')
    point_index=Path('out/v5/point_tx_3mhz_v1/index.json')
    point=json.loads(point_index.read_text())
    if args.sample not in {r['id'] for r in point['records']}:
        raise ValueError('Missing point acquisition: do not silently drop that observation group')
    material=np.load(args.material)
    if material.shape!=(3,961,1121) or material.dtype!=np.float32 or not np.isfinite(material).all():
        raise ValueError('Require finite absolute native float32 [c,rho,total-alpha]')
    if np.any(material[:2]<=0) or np.any(material[2]<0):raise ValueError('Invalid physical material')
    reference=None
    if args.reference_material is not None:
        reference=np.load(args.reference_material)
        if reference.shape!=material.shape or reference.dtype!=np.float32 or not np.isfinite(reference).all():
            raise ValueError('Invalid previous inferred material')
    args.out.mkdir();(args.out/'jobs').mkdir()
    manifest=source_manifest()
    manifest[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run={'status':'starting_native_RF_score','pid':os.getpid(),'sample':args.sample,
        'material':str(args.material.resolve()),
        'material_SHA256':hashlib.sha256(args.material.read_bytes()).hexdigest(),
        'source_sha256':manifest,'after_root':str(args.after_root.resolve()),
        'point_index_SHA256':hashlib.sha256(point_index.read_bytes()).hexdigest(),
        'unknowns':'fixed inferred c/rho/tissue-alpha; no updates',
        'source_groups':['lf','point3'],'shots_per_group':11,'receivers':192,
        'all_retained_RF':True,'native_shape':list(material.shape),'native_dx_m':50e-6,
        'devices':args.devices,'allocator_cap_GiB':args.cap_gib,'model_per_card_limit_GiB':16,
        'gradient_or_HVP_evaluations':0,'material_truth_read':False,
        'joint_reconstruction_success_claimed':False,'created_unix_s':time.time()}
    provenance=args.material.parent/'provenance.json'
    if provenance.exists():run['inference_provenance']=json.loads(provenance.read_text())
    if reference is not None:
        run['reference_inferred_material']=str(args.reference_material.resolve())
        run['reference_material_SHA256']=hashlib.sha256(args.reference_material.read_bytes()).hexdigest()
    atomic(args.out/'run.json',run);save_array(args.out/'material.npy',material)
    monitor=PressureResources(args.out);pool=None
    try:
        pool=Pool(args,monitor)
        result,_=pool.evaluate(material,'forward','fixed_inferred_material')
        atomic(args.out/'native_RF.json',result)
        if reference is not None:
            save_array(args.out/'previous_inferred_material.npy',reference)
            previous,_=pool.evaluate(reference,'forward','previous_inferred_material')
            atomic(args.out/'previous_native_RF.json',previous)
            run.update(reference_native_RF_loss=previous['loss'],
                change_from_previous_inferred_RF=result['loss']-previous['loss'])
        run.update(status='native_RF_scored',native_RF_loss=result['loss'],finished_unix_s=time.time())
        atomic(args.out/'run.json',run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',run);raise
    finally:
        if pool is not None:pool.close()
        monitor.log.close()


if __name__=='__main__':main()
