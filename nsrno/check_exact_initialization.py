"""CPU-only preservation of RF-derived initial maps, never material truth."""
from pathlib import Path
import json,time
import numpy as np
import torch
from .exact_travel_parameters import ExactTravelTimeParameters
from .travel_time_parameters import TravelTimeParameters
from .reconstruct_time import Parameters


def main():
    torch.set_num_threads(1)
    paths=[Path('out/v5/reconstruction_point_env_v2_train_200/plane_fwi_initial_stage2_best_parameters.npy'),Path('out/v5/encoded_rf_v1/checkpoint032/stage0_best_parameters.npy')]
    reports=[];out=Path('out/v5/travel_exact_v1');out.mkdir(exist_ok=True)
    for path in paths:
        previous=torch.from_numpy(np.load(path));target=Parameters((961,1121),1).material(previous)
        item={'input':str(path),'truth_read':False,'scales':[]}
        old=TravelTimeParameters((961,1121),1);old_m=old.material(old.initialize(previous))
        item['old_maximum_material_error']=[float((a-b).abs().max()) for a,b in zip(old_m,target)];del old,old_m
        for step in [4,2,1]:
            start=time.time();model=ExactTravelTimeParameters((961,1121),step);packed=model.initialize(previous)
            item['scales'].append({'step':step,'seconds':time.time()-start,**model.initialization_report});del model,packed
        reports.append(item);(out/'initial_preservation.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(item),flush=True)


if __name__=='__main__':main()
