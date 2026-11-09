"""Finish already-authorized forward assessment; do not launch an inaccurate inverse."""
from pathlib import Path
import os,sys,json,time,subprocess,traceback
from nsrno.dataset import write_json

def main():
    os.chdir('/home_nvme/zhuangyang/nsrno'); out=Path('out/v5/uno_round1_triparam'); started=time.monotonic()
    while not (out/'evaluation_complete.json').exists():
        if (out/'controller_error.json').exists(): raise RuntimeError((out/'controller_error.json').read_text())
        if time.monotonic()-started>14400: raise TimeoutError('Forward run did not finish within4 hours')
        time.sleep(10)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'.',
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for module in ['nsrno.uno_mean_reference','nsrno.plot_uno_round1']:
        subprocess.run([sys.executable,'-m',module],env=env,check=True)
    # Portable inference-only weights, without the large optimizer state.
    import torch
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    scales=json.load(open('out/v5/uno_triparam_v1/index.json'))['scales']
    torch.save({'model':saved['model'],'config':saved['settings']['network'],'scales':scales,'step':saved['step'],
        'pure_neural':True,'alpha_is_current_unknown_input':True},out/'model_weights.pt')
    result=json.load(open(out/'forward_evaluation.json'))
    status={'completed':True,'pure_neural_forward_complete':True,'figures_complete':True,
        'passed_for_inverse':result['passed_for_inverse'],'inverse_run':False,
        'next_action':'run gated three-parameter reconstruction and original physical replay' if result['passed_for_inverse'] else 'report forward/attenuation-response failure; do not run unreliable surrogate inversion'}
    write_json(out/'postprocess_complete.json',status); print(json.dumps(status),flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        out=Path('/home_nvme/zhuangyang/nsrno/out/v5/uno_round1_triparam'); out.mkdir(parents=True,exist_ok=True)
        write_json(out/'postprocess_error.json',{'error':traceback.format_exc()}); raise
