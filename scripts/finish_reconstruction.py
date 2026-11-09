from pathlib import Path
import json,hashlib,shutil
from datetime import datetime,timezone
root=Path('/home_nvme/zhuangyang/nsrno')
out=root/'out/v5/reconstruction_round1_train_200_gel1mm'; snapshot=out/'source_snapshot'
paths=['nsrno/reconstruct_material.py','nsrno/conditioned_surrogate.py','nsrno/conditioned_wave.py','nsrno/conditioned_data.py',
       'nsrno/wave_solver.py','scripts/evaluate_reconstruction_physics.py','scripts/evaluate_reconstruction_truth.py',
       'scripts/plot_reconstruction.py','scripts/launch_reconstruction_gel.py','scripts/postprocess_reconstruction.py',
       'docs/reconstruction_round1.md','docs/reconstruction_round1_design.md']
hashes={}
for relative in paths:
    source=root/relative; dest=snapshot/relative; dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest); hashes[relative]=hashlib.sha256(source.read_bytes()).hexdigest()
(out/'source_manifest.json').write_text(json.dumps({'saved_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes},indent=2))
readme=root/'README.md'; text=readme.read_text(); title='## 第一次实际声速—密度联合重建'
if title not in text:
    text+='\n\n'+title+'\n\n'
    text+='2026-11-05 已完成独立开发样本 train_200 的实际联合反演，详见[重建报告](docs/reconstruction_round1.md)。冻结第七轮模型，从均匀 c/rho 开始，已知真实衰减图，固定1 mm耦合层，优化200次。结果没有恢复有效组织结构：声速RMSE由53.70变为53.74 m/s，密度由49.86变为49.91 kg/m³。神经模型中的拟合误差0.00121，留出阵元0.98391；收敛物理复算的拟合误差0.25900。真实材料在神经模型中反而产生约0.506的拟合误差，当前代理目标尚不适合可靠材料反演。\n\n'
    text+='完整图、数组和记录位于 `out/v5/reconstruction_round1_train_200_gel1mm`；初始耦合层未固定的对照完整保留。入口为 `nsrno.reconstruct_material`。本次没有重新训练网络，未进行完整物理伴随反演，所有计算已完成。\n'
    readme.write_text(text)
print(json.dumps({'snapshot_files':len(paths),'report':str(root/'docs/reconstruction_round1.md')}))
