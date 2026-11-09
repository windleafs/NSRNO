"""End this owned CDF trajectory at a completed full-source accepted boundary.

No signal/kill, no source edit and no interruption of already issued waves.
The controller only requests the native driver's existing cooperative stop.
"""
from pathlib import Path
import argparse
import json
import os
import time


def identity(pid):
    p=Path('/proc',str(pid),'stat')
    if not p.exists():
        return None
    return int(p.read_text().rsplit(')',1)[1].split()[19])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    root=a.root.resolve()
    expected=Path('/home_nvme/zhuangyang/nsrno/out/v5/reconstruction_global_CDF_continuation_v1')
    if root!=expected.resolve():
        raise ValueError('Only the one owned and recorded CDF trajectory is allowed')
    pid,ticks=3619749,11407646
    boot='c86bf89c-042b-40c7-98c5-51da9b5741d3'
    reason='scientific_boundary_stop:squared_denoising_residual_has_antimode_minima;preserve_full_gradient_for_fixed_model_score_comparison'
    while True:
        run=json.loads((root/'run.json').read_text())
        if run['pid']!=pid or run['start_ticks']!=ticks or run['boot_id']!=boot:
            raise RuntimeError('Do not control an unrecognized experiment')
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=boot:
            raise RuntimeError('Boot identity changed; no action')
        if identity(pid)!=ticks or run['status'].endswith(('finished','failed_for_inspection')):
            print(json.dumps({'action':'none','reason':'original_native_process_terminal_or_gone','native_status':run['status']}),flush=True)
            return
        if (root/'stop_requested.flag').exists():
            print(json.dumps({'action':'none','reason':'an_existing_stop_request_is_already_present'}),flush=True)
            return
        accepted=[r for r in run['accepted_steps'] if r['objective']=='full_joint']
        if accepted:
            last=accepted[-1]
            point=last['point']
            job=json.loads((root/'active_job.json').read_text())
            work=json.loads((root/'actual_native_work.json').read_text())
            completed=(work.get('unfinished_native_phases')==0 and
                work.get('issued_adjoint_direction_passes')==work.get('completed_adjoint_direction_passes') and
                work.get('issued_physical_forward_passes')==work.get('completed_physical_forward_passes'))
            if (run['active_point']==point and job['point']==point and job['objective']=='full_joint' and
                job['kind']=='forward' and completed):
                cache=json.loads((root/'steps'/f'point{point:02d}'/'accepted_source_cache.json').read_text())
                if (cache['objective']!='full_joint' or cache['source_count']!=44 or
                    len(cache['RF_source_records'])!=44 or cache['material_SHA256']!=last['material_SHA256']):
                    raise RuntimeError('Do not stop without the complete accepted44source cache')
                # Recheck the top-level state just before requesting the flag.
                fresh=json.loads((root/'run.json').read_text())
                fresh_job=json.loads((root/'active_job.json').read_text())
                if fresh['active_point']!=point or fresh_job['token']!=job['token']:
                    time.sleep(1)
                    continue
                event={'action':'cooperative_stop_requested_at_complete_accepted_full_boundary',
                    'controller_pid':os.getpid(),'native_pid':pid,'native_start_ticks':ticks,
                    'boot_id':boot,'point':point,'accepted_material_SHA256':last['material_SHA256'],
                    'accepted_full_RF':last['RF'],'actual_work_at_boundary':work,
                    'unix_s':time.time(),'reason':reason,'no_native_work_interrupted':True,
                    'all_old_budgets_preserved':True,'no_source_file_modified':True}
                with (root/'stop_requested.flag').open('x') as stream:
                    stream.write(reason+'\n')
                (root/'scientific_boundary_stop.json').write_text(json.dumps(event,indent=2)+'\n')
                print(json.dumps(event),flush=True)
                return
        if time.time()>=run['deadline_unix_s']:
            print(json.dumps({'action':'none','reason':'original_deadline_reached'}),flush=True)
            return
        time.sleep(2)


if __name__=='__main__':
    main()
