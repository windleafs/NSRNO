"""Wait for the SAME training, then fixed inference and separate CPU map evaluation."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback


def atomic(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    proc = Path('/proc', str(pid))
    try:
        stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        command = (proc/'cmdline').read_bytes().replace(bytes([0]), b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {'start_ticks':int(stat[19]), 'state':stat[0], 'command':command}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.train_root, args.out = args.train_root.resolve(), args.out.resolve()
    if not all(p.is_relative_to(Path.cwd().resolve()/'out/v5') for p in [args.train_root,args.out]):
        raise ValueError('All experiment records must remain inside nsrno')
    training = json.loads((args.train_root/'run.json').read_text())
    if training['status'] not in ['training_coefficient_chart_inverse','coefficient_chart_training_finished']:
        raise RuntimeError('Only the existing authorized chart training')
    evaluation = args.train_root/'evaluation'
    if args.out.exists() or evaluation.exists():
        raise RuntimeError('Inspect existing same finalization; never repeat implicitly')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if training['boot_id'] != boot:
        raise RuntimeError('Training boot changed')
    args.out.mkdir()
    names = [Path(__file__).name, 'evaluate_coefficient_chart_inverse.py',
             'evaluate_amplitude_angular_inverse.py', 'evaluate_time_material.py']
    hashes = {name:digest(Path(__file__).parent/name) for name in names}
    state = {'status':'waiting_for_same_chart_training', 'pid':os.getpid(),
        'start_ticks':identity(os.getpid())['start_ticks'], 'boot_id':boot,
        'created_unix_s':time.time(), 'training_root':str(args.train_root),
        'training_pid':training['pid'], 'training_start_ticks':training['start_ticks'],
        'training_deadline_unix_s':training['deadline_unix_s'],
        'waiting_deadline_unix_s':training['deadline_unix_s']+120,
        'inference_max_seconds':600, 'CPU_evaluation_max_seconds_per_map':120,
        'source_SHA256':hashes, 'coordinator_GPU_work':False,
        'new_training_launched':False, 'new_wave_solves':0,
        'new_acoustic_material_derivatives':0, 'reserved_case_unread':'Neg_47_Left',
        'development_truth_used_for_inference':False, 'joint_imaging_success_claimed':False}
    atomic(args.out/'run.json',state)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def unchanged():
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != boot:
            raise RuntimeError('Host boot changed')
        if any(digest(Path(__file__).parent/name) != sha for name,sha in hashes.items()):
            raise RuntimeError('Frozen inference/evaluation source changed')

    def child(command, env, label, maximum_seconds):
        unchanged()
        with (args.out/(label+'.log')).open('x') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
        info=identity(process.pid)
        state.update(status=label,child_pid=process.pid,child_start_ticks=info['start_ticks'],
            child_started_unix_s=time.time(),child_command=command)
        atomic(args.out/'run.json',state)
        while process.poll() is None:
            state['child_observed_unix_s']=time.time()
            atomic(args.out/'run.json',state)
            if time.time()-state['child_started_unix_s'] >= maximum_seconds:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    raise RuntimeError('Same timed-out child still live; inspect before any new work')
                raise RuntimeError(label+' exceeded its declared wall budget')
            time.sleep(2)
        state.update(child_returncode=process.returncode,child_finished_unix_s=time.time())
        atomic(args.out/'run.json',state)
        if process.returncode:
            raise RuntimeError(label+' failed; inspect the same retained log')
        state.update(child_pid=None,child_start_ticks=None)

    try:
        while True:
            unchanged()
            live=identity(training['pid'])
            if live is None:
                break
            if (live['start_ticks']!=training['start_ticks'] or
                ' -m nsrno.train_coefficient_chart_inverse ' not in live['command']):
                raise RuntimeError('Training process identity changed; do not guess or restart')
            progress_path=args.train_root/'progress.json'
            progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
            state.update(observed_unix_s=time.time(),training_still_live=True,
                         last_training_step=progress.get('step'))
            atomic(args.out/'run.json',state)
            if time.time()>=state['waiting_deadline_unix_s']:
                raise RuntimeError('Same training remains live after its full budget; inspect it')
            time.sleep(5)
        finished=json.loads((args.train_root/'run.json').read_text())
        if finished['status']!='coefficient_chart_training_finished':
            state.update(training_terminal_status=finished['status'],
                         training_last_checkpoint_step=finished.get('latest_checkpoint_step'))
            raise RuntimeError('Training exited before declared fixed-weight completion')
        if digest(finished['checkpoint_path'])!=finished['checkpoint_SHA256']:
            raise RuntimeError('Actual fixed final checkpoint differs')
        state.update(training_still_live=False,training_completed_steps=finished['completed_steps'],
                     checkpoint_SHA256=finished['checkpoint_SHA256'])
        atomic(args.out/'run.json',state)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                      '--format=csv,noheader'],text=True).strip()
        if apps:
            raise RuntimeError('GPU remains occupied; inspect before sequential inference')
        env=os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        child([sys.executable,'-m','nsrno.evaluate_coefficient_chart_inverse','--train-root',str(args.train_root),
               '--out',str(evaluation),'--max-seconds','600'],env,'fixed_final_EMA_inference',660)
        evaluated=json.loads((evaluation/'run.json').read_text())
        if evaluated['status']!='coefficient_chart_actual_predictions_finished':
            raise RuntimeError('Inference did not save its fixed complete predictions')
        # Development labels enter only these separate CPU processes AFTER inference exits.
        cpuenv=env.copy();cpuenv['CUDA_VISIBLE_DEVICES']=''
        summaries={}
        for name in ['train_200_reference','train_200_native_current']:
            folder=evaluation/name
            record=json.loads((folder/'prediction.json').read_text())
            if digest(folder/'material.npy')!=record['material_SHA256']:
                raise RuntimeError('Saved estimate changed before independent evaluation')
            child([sys.executable,'-m','nsrno.evaluate_time_material',str(folder/'material.npy'),
                   '--sample','train_200','--out',str(folder/'maps'),
                   '--estimate-label','Fixed final coefficient-chart UNO'],cpuenv,'CPU_maps_'+name,120)
            metric=json.loads((folder/'maps.metrics.json').read_text())
            summaries[name]={'material_SHA256':record['material_SHA256'],
                            'maps':metric['maps'],'post_hoc_only':True}
        atomic(args.out/'result_summary.json',{'estimates':summaries,
            'training_responses':evaluated.get('training_response_groups'),
            'inference_had_no_development_truth':True,
            'GPU_images_have_not_yet_been_viewed_by_agent':True,
            'joint_imaging_success_claimed':False})
        state.update(status='coefficient_chart_sequential_inference_and_CPU_maps_finished',
            finished_unix_s=time.time(),actual_native_predictions=2,
            separate_CPU_post_hoc_map_evaluations=2,manual_map_view_still_required=True)
        atomic(args.out/'run.json',state)
        print(json.dumps({k:state[k] for k in ['status','training_completed_steps',
            'actual_native_predictions','separate_CPU_post_hoc_map_evaluations']}),flush=True)
    except BaseException:
        state.update(status='coefficient_chart_finalization_stopped_for_inspection',
                     error=traceback.format_exc(),stopped_unix_s=time.time())
        atomic(args.out/'run.json',state)
        raise


if __name__=='__main__':
    main()
