"""Archive the final code and summarize the completed experiment, without new jobs."""
from pathlib import Path
import json,hashlib,tarfile,datetime

root=Path('/home_nvme/zhuangyang/nsrno')
training=root/'out/v5/forward_round8_dense_full'
inverse=root/'out/v5/reconstruction_round2_dense_train_200'
for path in (training/'training_complete.json',inverse/'completion.json',inverse/'postprocess_complete.json'):
    if not json.load(open(path))['completed']: raise RuntimeError(str(path))
report=root/'docs/forward_round8_dense.md'
body=report.read_text(encoding='utf-8-sig')
for name,folder in [('forward_error.png','forward_round8_dense_full'),
                    ('reconstruction.png','reconstruction_round2_dense_train_200'),
                    ('updates.png','reconstruction_round2_dense_train_200'),
                    ('rf_replay.png','reconstruction_round2_dense_train_200')]:
    body=body.replace(']('+name+')','](../out/v5/'+folder+'/'+name+')')
report.write_text(body,encoding='utf-8')
readme=root/'README.md'; content=readme.read_text()
marker='## 全部 11 角、密集有效频带与第二次实际联合重建'
if marker not in content:
    content+='''

## 全部 11 角、密集有效频带与第二次实际联合重建

2026-11-05 的[第八轮报告](docs/forward_round8_dense.md)记录已完成的密集训练及实际反演。采用原始 V5 的全部 11 角、192 阵元及 294 个相邻频点（4.015–8.896 MHz，包含名义滤波器过渡带），保留约 99.9958% 开发验证 RF 能量。训练轮换 32 条记录并完成全部频率/角度覆盖，未训练完整介质与采集的笛卡尔积。

固定独立开发样本 train_200 的全部 294×11 采集上，完整 RF 相对 RMS 在 32 次迭代时由 0.3900 降至 0.1631；64 次为 0.02904，物理残差收敛后为 0.006733。以上显式给定真实衰减图，不能与仅输入 c/rho 的模型混作相同条件。当前准确频域实现尚不比原时域求解器更快。

随后从均匀初值以收敛物理层及隐式伴随联合优化 c/rho，完成两遍全频带、196 次更新。实际成像失败：声速 RMSE 53.70→53.56 m/s，密度 49.86→49.99 kg/m³；原时域求解器独立复算的完整 RF 误差 0.9802→1.0378，18–36 μs 也未改善。三个监测频率的候选参数正演与时域结果相符，目前主要待解决的是材料优化未降低完整目标，不能据此断言密度不可恢复。

检查点为 `out/v5/forward_round8_dense_full/last.pt`，接口为 `nsrno.dense_surrogate.DenseSurrogate`，需要相同采集参考及显式 alpha。有限深度预测用 `truncated(..., iterations=64)`，默认接口使用收敛求解和隐式梯度；两种成本和精度应分别报告。重建图、原始数组、完整覆盖与独立复算记录在 `out/v5/reconstruction_round2_dense_train_200`。本轮全部计算已结束；以前的结果、3 MHz 数据和最终留出病例均保留。
'''
    readme.write_text(content)
files=sorted((root/'nsrno').glob('*.py'))+sorted((root/'scripts').glob('*.py'))
files += [readme,report,root/'docs/forward_round8_dense_design.md']
archive=root/'out/v5/round8_source_snapshot.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for path in files: tar.add(path,arcname=str(path.relative_to(root)))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'captured_after_run':True,'source_archive':str(archive.relative_to(root)),
    'source_sha256':{str(p.relative_to(root)):digest(p) for p in files},
    'source_archive_sha256':digest(archive),'trained_checkpoint_sha256':digest(training/'last.pt'),
    'completed_training':json.load(open(training/'training_complete.json')),
    'completed_reconstruction':json.load(open(inverse/'completion.json')),
    'selected_candidate':json.load(open(inverse/'selected.json')),
    'server_recovery':json.load(open(inverse/'server_recovery.json')),
    'jobs_complete':True}
for folder in (training,inverse):
    (folder/'final_record.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'source_archive':str(archive),'files':len(files),'selected':record['selected_candidate']},indent=2))
