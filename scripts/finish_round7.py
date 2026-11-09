from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone
root=Path('/home_nvme/zhuangyang/nsrno'); out=root/'out/v5/forward_round7_conditioned'
snapshot=out/'source_snapshot'; snapshot.mkdir(exist_ok=True)
files=['nsrno/conditioned_wave.py','nsrno/conditioned_surrogate.py','nsrno/conditioned_data.py',
       'nsrno/train_preconditioner.py','nsrno/wave_solver.py','scripts/prepare_conditioned_data.py',
       'scripts/prepare_wave_teachers.py','scripts/check_conditioned_source.py',
       'scripts/summarize_conditioned_teachers.py','scripts/evaluate_round4_selected.py',
       'scripts/diagnose_learned_preconditioner.py','scripts/compare_hybrid_solver.py','scripts/check_conditioned_api.py',
       'docs/forward_round7_design.md','docs/forward_round7.md']
hashes={}
for relative in files:
    source=root/relative; dest=snapshot/relative; dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest); hashes[relative]=hashlib.sha256(source.read_bytes()).hexdigest()
(out/'source_manifest.json').write_text(json.dumps({'saved_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes},indent=2))
readme=root/'README.md'; text=readme.read_text()
text=text.replace('最新结果见 [材料相关传播时间训练报告]','完整 RF 基准结果见 [材料相关传播时间训练报告]')
heading='## 已知衰减条件下的可微波动算子与学习初始化'
if heading not in text:
    text+='\n\n'+heading+'\n\n'
    text+='2026-11-05 的 [第七轮报告](docs/forward_round7.md)记录新的 240 步训练。在 16 个独立病例验证记录、三个频率和三个角度的 144 个观测上，学习预条件器的 16 步通道频谱误差为 0.50882，固定 16 步为 0.75554，同一观测的第四轮模型为 0.65568。显式给定真实空间衰减图时，收敛参考误差为 0.00719；这与只输入 c/rho 的模型具有不同条件。它还不是完整 RF 或深层反演结果。\n\n'
    text+='新检查点为 `out/v5/forward_round7_conditioned/last.pt`，频谱可微接口是 `nsrno.conditioned_surrogate.ConditionedSurrogate`，要求显式传入 alpha 及固定采集参考。全程使用学习预条件器反而增加收敛计算量；将其作为 8 步初始波场后再固定求解，在 12 个固定监测观测上，平均 307 步降至 288 步，1.66 秒降至 1.57 秒。原第四轮完整 RF 基准保留。所有本轮任务已完成，没有后台训练。\n'
    readme.write_text(text)
print(json.dumps({'snapshot_files':len(files),'report':str(root/'docs/forward_round7.md'),'readme_updated':True}))
