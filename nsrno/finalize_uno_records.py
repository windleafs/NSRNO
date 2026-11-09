"""Publish the actual completed trial and archive final source/provenance."""
from pathlib import Path
import json,hashlib,tarfile,datetime

root=Path('/home_nvme/zhuangyang/nsrno'); data=root/'out/v5/uno_triparam_v1'; out=root/'out/v5/uno_round1_triparam'
for file in ['training_complete.json','evaluation_complete.json','postprocess_complete.json']:
    if not json.load(open(out/file))['completed']: raise RuntimeError(str(file))
evaluation=json.load(open(out/'forward_evaluation.json'))
if evaluation['passed_for_inverse']: raise RuntimeError('Inverse is now required before finalizing the full trial')
report=root/'docs/uno_round1_triparam.md'; body=report.read_text(encoding='utf-8-sig')
for name in ['memory.png','forward_results.png','hf_train_200_rf.png','lf_train_200_rf.png','alpha_response.png']:
    body=body.replace(']('+name+')','](../out/v5/uno_round1_triparam/'+name+')')
report.write_text(body,encoding='utf-8')
readme=root/'README.md'; text=readme.read_text(); marker='## 纯 UNO 三参数通道正演：第一轮'
if marker not in text:
    text+='''

## 纯 UNO 三参数通道正演：第一轮

2026-11-05 的[实际报告](docs/uno_round1_triparam.md)记录已完成的纯神经算子实现和6000步训练。`nsrno.uno_rf.ChannelUNO` 使用二维空间UNO、连续神经积分读出和接收域UNO，从原生c/rho/alpha预测完整复IQ及RF；在线没有波动方程迭代、物理伴随或参考内部波场。它是通道RF的UNO改型，不是原论文网络的原样复现。

全部780条高频记录的衰减标签已恢复，使用原有48条3MHz配对，另生成32训练/16开发验证材料状态、两种源的全部11角，共96条新源记录。数据位于 `out/v5/uno_triparam_v1`。真实组织衰减只用于有监督正演输入；三参数反演入口将其作为未知量，保持已知数值边缘吸收带及频率幂指数1.1。

实际训练PyTorch峰值分配0.912GiB、保留1.160GiB；最终权重完整11角RF正演约0.0677秒，两源共22角三参数目标梯度约1.025秒。318条开发源记录、每条完整11角的RF相对RMS为0.71887，18–36微秒1.00041，独立衰减差分1.02511。训练集平均波形基准为0.72310，当前模型主要学到浅表共同回波，尚不适合联合重建。显存和速度改善不能当作等精度的物理替代。

推理权重为 `out/v5/uno_round1_triparam/model_weights.pt`，由固定开发监测选取第4000步；完整训练到6000步。接口 `nsrno.uno_surrogate.UNOSurrogate` 可独立加载权重及其中的归一化常数。该检查点仅供研究复现，尚未达到反演条件。`nsrno.reconstruct_uno` 已实现全观测梯度、线搜索和包含初值的候选选择，但当前会因正演精度不达标而拒绝启动。没有本轮三参数重建结果。全部本轮计算已结束，原模型与最终留出病例保留。
'''
    readme.write_text(text)
names=['uno_rf','uno_data','prepare_uno_data','profile_uno_rf','generate_uno_pairs','uno_evaluate','train_uno_rf',
       'run_uno_round1','uno_surrogate','uno_mean_reference','reconstruct_uno','plot_uno_round1','run_uno_postprocess',
       'benchmark_uno','rf_representation','fullwave','generate_low_frequency','dataset','variants']
files=[root/'nsrno'/(name+'.py') for name in names]
files += [report,readme,root/'docs/superpowers/specs/2026-11-05-uno-triparameter-rf-design.md',
          out/'source_common_iq_1500/uno_rf.py',out/'source_common_iq_1500/validations.json']
archive=out/'source_snapshot.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for file in files: tar.add(file,arcname=str(file.relative_to(root)))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
index=json.load(open(data/'index.json')); augmented=json.load(open(data/'augmentation_index.json'))
allrows=index['records']+augmented['records']
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'captured_after_run':True,
    'first_trial_complete':True,'original_three_parameter_imaging_goal_achieved':False,'inverse_executed':False,
    'stop_reason':'6000-update budget completed; fullRF/lateRF/independent-alpha criteria all failed',
    'training':json.load(open(out/'training_complete.json')),'selected_step':evaluation['selected_step'],
    'forward_development_records':evaluation['records'],'angles_per_record':11,'rf_errors':evaluation['relative_rms'],
    'alpha_response_error':evaluation['paired_alpha_relative_rms'],'trained_benchmark':json.load(open(out/'trained_benchmark.json')),
    'dataset':{'source_records':len(allrows),'training_source_records':sum(r['role']=='train' for r in allrows),
        'development_source_records':sum(r['role']=='validation' for r in allrows),'new_materials':48,'new_source_records':96},
    'source_archive':str(archive.relative_to(root)),'source_sha256':{str(p.relative_to(root)):digest(p) for p in files},
    'source_archive_sha256':digest(archive),'inference_weights_sha256':digest(out/'model_weights.pt'),
    'data_index_sha256':digest(data/'index.json'),'augmentation_index_sha256':digest(data/'augmentation_index.json')}
(out/'final_record.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k not in ['source_sha256','trained_benchmark']},indent=2))
