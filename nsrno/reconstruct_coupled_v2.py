"""Full-observation three-material inversion with a coded positive metric.

The gradient and acceptance use all original shots individually. Source-coded
shot-mean geometric curvature is a positive approximation, not the exact Hessian of
the source-dependent geometric objective. No material truth enters this file.
"""
from argparse import ArgumentParser
from pathlib import Path
import hashlib, json, multiprocessing as mp, os, queue, signal, subprocess, time, traceback
import numpy as np
import torch
from .coupled_coordinates_v2 import Coordinates, pcg, dot
from .coupled_worker_v2 import worker
from .time_acquisition import TimeAcquisition
from .encoded_time import prepare_bank
from .reconstruct_time import atomic_json


def save_numpy(path, value):
    partial = path.with_name(path.stem+'.partial.npy')
    np.save(partial, value)
    partial.replace(path)


def main():
    p = ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--initial-parameters', type=Path, required=True)
    p.add_argument('--sample', default='train_200')
    p.add_argument('--devices', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--sources', nargs='+', choices=['lf','hf','point3'], default=['point3','lf','hf'])
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--cg-iterations', type=int, default=8)
    p.add_argument('--cg-tolerance', type=float, default=.15)
    p.add_argument('--memory-limit-gib', type=float, default=4.75)
    p.add_argument('--minimum-host-available-gib', type=float, default=100.)
    p.add_argument('--chunk', type=int, default=96)
    p.add_argument('--radii', type=float, nargs=3, default=[.25,.5,.3])
    p.add_argument('--tv-weight', type=float, default=1e-3)
    p.add_argument('--precondition-mm', type=float, default=.5)
    p.add_argument('--damping-factor', type=float, default=.02)
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--qualification', type=Path, default=Path('out/v5/native_tangent_v1/precision_review.json'))
    a = p.parse_args()
    mem = dict(row.split(':',1) for row in Path('/proc/meminfo').read_text().splitlines())
    available = float(mem['MemAvailable'].split()[0])/2**20
    if available < a.minimum_host_available_gib:
        raise RuntimeError(f'Two cached codes require host memory guard: {available:.2f} GiB available')
    if len(set(a.devices)) != len(a.devices) or len(set(a.sources)) != len(a.sources):
        p.error('Devices and acquisition groups must be distinct')
    if len(a.devices) > len(a.sources):
        p.error('Do not create idle GPU workers')
    if not all(v > 0 for v in a.radii) or a.damping_factor <= 0:
        p.error('Trust radii and damping must be positive')
    if a.chunk < 1:
        p.error('Checkpoint chunk must be positive')
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/'run.json').exists():
        raise FileExistsError('Preserve prior inversion results')
    usage = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                     '--format=csv,noheader,nounits'],text=True)
    card_used = {int(row.split(',')[0]):int(row.split(',')[1]) for row in usage.splitlines()}
    if any(card_used[d]+a.memory_limit_gib*1024+768 > 16*1024 for d in a.devices):
        raise RuntimeError('No room inside the 16 GiB whole-card budget; wait for the existing jobs')
    qualification = json.loads(a.qualification.read_text())
    if not qualification['qualified_for_curvature_implementation']:
        raise RuntimeError('The native tangent has not been qualified')
    for name, digest in qualification['sources_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Tangent qualification is stale for {name}')
    initial = np.load(a.initial_parameters).astype(np.float32)
    coordinates = Coordinates(initial, a.radii, a.tv_weight, a.precondition_mm)
    u = coordinates.initial.copy()
    loads = [0.]*len(a.devices)
    assignments = [[] for _ in a.devices]
    for source in sorted(a.sources, key=lambda s:1.5 if s=='point3' else 1., reverse=True):
        rank = int(np.argmin(loads))
        assignments[rank].append(source)
        loads[rank] += 1.5 if source=='point3' else 1.
    files = ['reconstruct_coupled_v2.py','coupled_coordinates_v2.py','coupled_worker_v2.py','coupled_coordinates.py',
             'native_tangent.py','native_tangent.cu','native_time.py','native_time.cu',
             'time_acquisition.py','v5_observations.py','rf_depth_weights.py','reconstruct_time.py','encoded_time.py',
             'run_coupled_postprocess.py','launch_coupled_when_ready.py','launch_full_rf_from_wri.py']
    record = {'status':'preparing_source_banks','pid':os.getpid(),'sample':a.sample,
              'devices':a.devices,'sources':a.sources,'assignments':assignments,
              'memory_limit_GiB_per_worker':a.memory_limit_gib,
              'checkpoint_chunk':a.chunk,
              'card_used_MiB_before':card_used,
              'iterations':a.iterations,'cg_iterations':a.cg_iterations,'cg_tolerance':a.cg_tolerance,
              'trust_radii_q_c_Z_alpha':a.radii,'tv_weight':a.tv_weight,
              'precondition_mm':a.precondition_mm,'damping_factor':a.damping_factor,'seed':a.seed,
              'truth_used_by_optimizer':False,
              'unknowns':'independent native c, Z=c*rho and tissue alpha below 1 mm known gel',
              'target':'full unencoded geometric RF plus independent weak TV; all 11 shots and 192 receivers',
              'curvature':'mean of two fixed all-11-shot codes with dot=1; shot-mean geometric RF weights; positive approximation, not exact source-dependent Hessian',
              'preconditioner':'positive spatial Sobolev filter times fixed per-outer native-coordinate block scaling from current full gradient; no extra derivative probes',
              'damping_metric':'positive per-outer diagonal reciprocal to gradient block factors; gradient-scale approximation, not a measured Fisher diagonal',
              'minimum_host_available_GiB':a.minimum_host_available_gib,
              'checkpoint_selection':'accepted trajectory, full unencoded RF objective plus weak TV; no truth',
              'initial_parameters':str(a.initial_parameters.resolve()),
              'initial_parameters_sha256':hashlib.sha256(a.initial_parameters.read_bytes()).hexdigest(),
              'tangent_qualification':str(a.qualification.resolve()),
              'sources_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in files}}
    atomic_json(a.out/'run.json', record)
    for name in files:
        (a.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    save_numpy(a.out/'initial_parameters.npy', initial)
    save_numpy(a.out/'initial_material.npy', coordinates.material(u))
    uniform = np.zeros_like(initial)
    save_numpy(a.out/'uniform_initial_material.npy', coordinates.model.material(torch.from_numpy(uniform)).numpy())
    banks = {}
    for source in a.sources:
        cpu = TimeAcquisition(source, a.sample, device='cpu')
        banks[source] = str(prepare_bank(cpu))
        del cpu
    record.update(status='running', source_banks=banks)
    atomic_json(a.out/'run.json', record)
    ctx = mp.get_context('spawn')
    results = ctx.Queue()
    jobs = [ctx.Queue() for _ in a.devices]
    processes = [ctx.Process(target=worker, args=(rank,device,assignments[rank],a.sources,a.sample,
                 banks,jobs[rank],results,a.out,a.memory_limit_gib,a.chunk)) for rank,device in enumerate(a.devices)]
    for proc in processes:
        proc.start()
    atomic_json(a.out/'workers.json', {'workers':[{'rank':i,'pid':v.pid,'device':a.devices[i],
                'sources':assignments[i]} for i,v in enumerate(processes)]})
    stopped = [False]
    def request_stop(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    start = time.time()
    token, accepted = 0, 0
    best = {'objective':float('inf')}
    log = a.out/'evaluations.jsonl'
    rng = np.random.default_rng(a.seed)
    damping = None

    def emit(item):
        item['elapsed_seconds'] = time.time()-start
        with log.open('a') as f:
            f.write(json.dumps(item)+'\n')
        print(json.dumps(item), flush=True)
        return item

    def service(mode, payload=None, state_id=None, codes=None):
        nonlocal token
        token += 1
        for q in jobs:
            q.put((token,mode,payload,state_id,codes))
        pending = set(range(len(processes)))
        replies = []
        while pending:
            try:
                result = results.get(timeout=30)
            except queue.Empty:
                dead = [(i,p.exitcode) for i,p in enumerate(processes) if not p.is_alive()]
                if dead:
                    raise RuntimeError(f'Native curvature worker exited: {dead}')
                atomic_json(a.out/'wait_status.json', {'token':token,'mode':mode,
                    'pending_ranks':sorted(pending),'live_pids':[p.pid for p in processes],
                    'unix_s':time.time()})
                continue
            if result.get('error'):
                raise RuntimeError(result['error'])
            if result['token'] != token or result['rank'] not in pending:
                raise RuntimeError('Out-of-order curvature worker response')
            pending.remove(result['rank'])
            replies.append(result)
        replies.sort(key=lambda r:r['rank'])
        g = None
        if mode in ['gradient','hvp']:
            g = sum((r['gradient'].astype(np.float64) for r in replies),
                    np.zeros((3,961,1121),np.float64))
            if not np.isfinite(g).all():
                raise RuntimeError('Nonfinite native gradient')
        windows = {}
        for r in replies:
            windows.update({s:np.sqrt(v).tolist() for s,v in r['windows'].items()})
        return {'value':sum(r['value'] or 0. for r in replies),'gradient':g,'windows':windows,
                'quadratic':sum(r['quadratic'] or 0. for r in replies),
                'peak_GiB_per_worker':[r['peak_GiB'] for r in replies],
                'worker_seconds':[r['seconds'] for r in replies]}

    def save_best(q, item):
        if item['objective'] < best['objective']:
            best.update(objective=item['objective'],u=q.copy(),item=item.copy())
            save_numpy(a.out/'stage1_best_parameters.npy', coordinates.fields(q).astype(np.float32))
            save_numpy(a.out/'stage1_best_material.npy', coordinates.material(q))
            atomic_json(a.out/'stage1_best.json', item)

    def score_item(response, q, kind, step, **extra):
        tv, _, _ = coordinates.regularizer(q)
        item = {'kind':kind,'stage':1,'step':step,'data_loss':response['value'],
                'regularization':tv,'objective':response['value']+tv,
                'window_relative_rms':response['windows'],
                'peak_GiB_per_worker':response['peak_GiB_per_worker'],
                'worker_seconds':response['worker_seconds'],**extra}
        if not np.isfinite(item['objective']):
            raise RuntimeError('Nonfinite full RF objective')
        return emit(item)

    try:
        for iteration in range(1,a.iterations+1):
            material = coordinates.material(u)
            response = service('gradient', material)
            tv, gtv, tv_weight = coordinates.regularizer(u)
            gq = coordinates.transpose(u, material, response['gradient'])+gtv
            gp = coordinates.radii*gq
            current = score_item(response,u,'full_unencoded' if iteration==1 else 'full_gradient',accepted,
                                 outer_iteration=iteration,gradient_norm_by_coordinate=np.sqrt(np.sum(gq*gq,axis=(-2,-1))).tolist())
            if iteration == 1:
                save_best(u,current)
            if stopped[0]:
                break
            if dot(gp,gp) == 0:
                emit({'kind':'zero_numerical_gradient','outer_iteration':iteration,
                      'interpretation':'Not a material-recovery claim'})
                break
            if damping is None:
                damping = max(1e-10,a.damping_factor*np.sqrt(dot(gp,gp)/gp.size))
            codes = {s:rng.choice([-1.,1.],11).astype(np.float32) for s in a.sources}
            state_id = f'outer_{iteration}'
            service('prepare',material,state_id,codes)
            emit({'kind':'prepare_positive_curvature','outer_iteration':iteration,'damping':damping,
                  'codes_all_11_shots':{s:v.tolist() for s,v in codes.items()}})

            def metric(pvec):
                vq = coordinates.radii*pvec
                dm = coordinates.tangent(u,material,vq)
                result = service('hvp',dm,state_id)
                hq = coordinates.transpose(u,material,result['gradient'])
                data_inner = dot(dm.astype(np.float64),result['gradient'])
                mismatch = abs(data_inner-result['quadratic'])/max(abs(data_inner),abs(result['quadratic']),1e-30)
                if mismatch > .01 or data_inner < 0:
                    raise RuntimeError(f'Inconsistent or indefinite encoded normal product: {mismatch}, {data_inner}')
                emit({'kind':'actual_coupled_metric_product','outer_iteration':iteration,
                      'encoded_data_quadratic':result['quadratic'],'material_dot_JTWJ':data_inner,
                      'normal_product_relative_difference':mismatch,
                      'peak_GiB_per_worker':result['peak_GiB_per_worker'],
                      'worker_seconds':result['worker_seconds']})
                return coordinates.radii*(hq+coordinates.tv_metric(vq,tv_weight))+damping*coordinates.damping_metric(pvec)

            factors = coordinates.set_block_precondition(gp)
            emit({'kind':'block_preconditioner','outer_iteration':iteration,'positive_factors':factors,
                  'positive_LM_block_diagonal':(1/np.asarray(factors)).tolist()})
            direction,cg_history = pcg(metric,-gp,coordinates.precondition,
                maximum=a.cg_iterations,tolerance=a.cg_tolerance,
                callback=lambda row:emit({'kind':'cg_iteration','outer_iteration':iteration,**row}))
            rms = np.sqrt(np.mean(direction*direction,axis=(-2,-1)))
            trust_scale = min(1.,1/max(float(rms.max()),1e-30))
            direction *= trust_scale
            dq = coordinates.radii*direction
            if dot(gq,dq) >= 0 or not np.isfinite(dq).all():
                raise RuntimeError('The coupled direction is not a finite full-gradient descent direction')
            candidate_accepted = False
            for line in range(6):
                alpha = .5**line
                candidate = np.clip(u+alpha*dq,coordinates.lower,coordinates.upper)
                slope = dot(gq,candidate-u)
                if slope >= 0:
                    emit({'kind':'rejected_projected_direction','outer_iteration':iteration,'alpha':alpha,'slope':slope})
                    continue
                evaluation = service('score',coordinates.material(candidate))
                trial = score_item(evaluation,candidate,'line_search',accepted+1,outer_iteration=iteration,
                                   alpha=alpha,full_gradient_slope=slope,trust_scale=trust_scale,damping=damping)
                if trial['objective'] <= current['objective']+1e-4*slope:
                    u = candidate
                    accepted += 1
                    trial.update(kind='full_unencoded',step=accepted,accepted=True)
                    emit(trial)
                    save_best(u,trial)
                    candidate_accepted = True
                    if alpha < .25 or trust_scale < .5:
                        damping *= 2
                    elif alpha == 1. and trust_scale == 1.:
                        damping *= .5
                    break
                if stopped[0]:
                    break
            if not candidate_accepted:
                damping *= 10
                emit({'kind':'outer_step_rejected','outer_iteration':iteration,'next_damping':damping})
            if stopped[0]:
                break
        if 'u' not in best:
            raise RuntimeError('No complete full-observation candidate was evaluated')
        save_numpy(a.out/'final_parameters.npy',coordinates.fields(best['u']).astype(np.float32))
        save_numpy(a.out/'final_material.npy',coordinates.material(best['u']))
        result = {'status':'interrupted_at_safe_boundary' if stopped[0] else 'optimization_finished',
                  'accepted_updates':accepted,'best':best['item'],
                  'selection':'full unencoded RF plus weak TV; not truth; budget completion is not material recovery',
                  'seconds':time.time()-start}
        atomic_json(a.out/'stage1_result.json',result)
        record.update(status=result['status'],accepted_updates=accepted,seconds=time.time()-start)
        atomic_json(a.out/'run.json',record)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),seconds=time.time()-start)
        atomic_json(a.out/'run.json',record)
        raise
    finally:
        for q in jobs:
            q.put(None)
        for proc in processes:
            proc.join(timeout=30)
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)


if __name__ == '__main__':
    main()
