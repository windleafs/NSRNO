# NSRNO

NSRNO 用 V5 乳腺数据检验固定单侧平面波采集下的声速和密度反演。项目使用独立的数据读取器与 UltraWave 物理参考。设计见 [双参数 RF 反演设计](docs/superpowers/specs/2026-11-04-nsrno-dual-parameter-rf-design.md)。

## 运行环境

在本目录下运行。CPU 数据读取与 UltraWave 仿真分别使用独立 Python 进程，避免在 GPU 仿真进程中载入 PyTorch。

```bash
cd /home_nvme/zhuangyang/nsrno
PY=/home/zhuangyang/miniconda3/envs/py310/bin/python
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
```

## 数据与零扰动复演

```bash
"$PY" -m nsrno.dataset --out out/v5/manifest.json
"$PY" -m nsrno.audit_data --manifest out/v5/manifest.json --out out/v5/data_audit.json

CUDA_VISIBLE_DEVICES=0 "$PY" -m nsrno.fullwave \
  --manifest out/v5/manifest.json --id train_000 --out out/v5/replay_example

"$PY" -m nsrno.compare_replay --manifest out/v5/manifest.json \
  --replay out/v5/replay_example --ids train_000 --out out/v5/replay_example/metrics.json
```

数据清单中的 RF 顺序为 `[angle,receiver,time]`，原生介质顺序为 `[z,x]`。数据读取返回原符号、原增益的通道 RF，以及原生标签裁剪后保存的声速和反射率代理 m。密度与衰减由原物理生成配方重建。复演沿用固定均匀参考相减、带通与重采样。输出记录来源哈希、模拟角度和每角耗时。对已有样本重新复演时，请使用新的输出目录。

开发病例为 Neg_07_Left 和 Neg_35_Left，变体按父样本继承病例。Neg_47_Left 是最终留出病例。V5 的旧验证切片可从清单追溯，模型选择须用独立病例分折。

## 局部物理先导

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m nsrno.local_probe simulate \
  --manifest out/v5/manifest.json --id train_000 --out out/v5/local_probe_example

"$PY" -m nsrno.local_probe analyze --manifest out/v5/manifest.json \
  --id train_000 --out out/v5/local_probe_example --replay out/v5/replay_example
```

在已知完整介质上，用同一个空间窗分别扰动 log(c) 和 log(ρ)，由 ±0.01 和 ±0.005 的全波中心差分比较两列雅可比的稳定性和夹角。在固定的三个角度上用独立生成的混合扰动检查线性近似和单参数串扰。该先导仅测量给定背景、空间窗和角度下的可分离性。首轮神经正演训练见下文。

## 神经 RF 正演代理

完整 RF 基准结果见 [材料相关传播时间训练报告](docs/forward_round4.md)，方法见 [第四轮实现说明](docs/forward_round4_method.md)。新分支按输入声速积分散射位置到各阵元的路径时间，再学习复数波形读出。开发验证病例全部 11 角的 18–36 μs 相对 RF RMS 从 0.99169 降至 0.85409，36 μs 以后仍未改善，完整 RF 预测尚未解决。用于后续正演开发的检查点是 `out/v5/forward_round4_material/best.pt`（第 70 轮），通过 `nsrno.forward_model.load_forward_model` 加载；在线计算仍包含传播聚合，不能将缓存后的训练速度当作在线推理速度。

[第三轮训练报告](docs/forward_round3.md)记录原生细节、非局部聚合及累计慢度查询的比较，保留第三轮检查点及结果。

[第二轮训练报告](docs/forward_round2.md)记录全局输出尺度、分窗输出尺度以及包含载频的复解析波形表示，保留独立模型与已有 V5 数据划分。

[首轮训练记录](docs/forward_round1.md)保留原始结果；[第二轮方法说明](docs/forward_round2_method.md)记录参数化和停止标准。


## 3 MHz 配对数据与波动方程神经算子

2026-11-05 的首批低频数据已完成：32 训练、16 独立病例验证记录，每条 11 角；原始 RF 在 `out/v5/low_frequency_3mhz_v1`，训练格式缓存在 `out/v5/forward_cache_3mhz_v1`。材料与对应 V5 逐项匹配，最终留出病例未使用。当前 `train_wave` 的频率抽样限定原高频 V5；低频训练需要使用低频采集参数与弛豫系数。

论文分析和结果见 [第五轮报告](docs/forward_round5.md)，推导与实现选择见 [第五轮设计](docs/forward_round5_design.md)。保留完整相对压缩率和相对密度成像目标。新 `nsrno.wave_operator` 实现显式变密度波动方程残差迭代、可学习预条件器和双弛豫频散；原型检查点为 `out/v5/forward_round5_wave_relax_v1/best.pt`，使用 `load_wave_operator` 加载。全部 270 验证记录每条一个固定频率/角度的误差为 0.65497，原第四轮为 0.64633，因此尚未替代第四轮 RF 基准。单频可微接口与完整宽带反演不是同一完成程度。


## 收敛误差与局部弛豫诊断

2026-11-05 的后续诊断见 [第六轮报告](docs/forward_round6.md)。两个开发病例各一个样本、三个频率、三个角度的 18 个观测上，同一第五轮方程充分求解后的合并通道谱误差从 0.68897 降到 0.33898，第四轮为 0.62372。该结果不是完整 RF 或全部验证切片的结果。单个验证观测的内部场对照表明，真实局部弛豫条件与当前共享系数近似之间仍有重要差别；报告记录了实际边界执行顺序和有限时间端点处理。

`nsrno.wave_solver.solve_frequency` 提供仅依赖现有模型和 c/rho 的收敛参考，返回实际残差和终止原因。它是数值参考，尚无隐式伴随反传，不能直接用于材料反演。原第四轮模型及所有训练检查点保留。本轮只完成诊断和参考实现，没有新的后台训练。


## 已知衰减条件下的可微波动算子与学习初始化

2026-11-05 的 [第七轮报告](docs/forward_round7.md)记录新的 240 步训练。在 16 个独立病例验证记录、三个频率和三个角度的 144 个观测上，学习预条件器的 16 步通道频谱误差为 0.50882，固定 16 步为 0.75554，同一观测的第四轮模型为 0.65568。显式给定真实空间衰减图时，收敛参考误差为 0.00719；这与只输入 c/rho 的模型具有不同条件。它还不是完整 RF 或深层反演结果。

新检查点为 `out/v5/forward_round7_conditioned/last.pt`，频谱可微接口是 `nsrno.conditioned_surrogate.ConditionedSurrogate`，要求显式传入 alpha 及固定采集参考。全程使用学习预条件器反而增加收敛计算量；将其作为 8 步初始波场后再固定求解，在 12 个固定监测观测上，平均 307 步降至 288 步，1.66 秒降至 1.57 秒。原第四轮完整 RF 基准保留。所有本轮任务已完成，没有后台训练。


## 第一次实际声速—密度联合重建

2026-11-05 已完成独立开发样本 train_200 的实际联合反演，详见[重建报告](docs/reconstruction_round1.md)。冻结第七轮模型，从均匀 c/rho 开始，已知真实衰减图，固定1 mm耦合层，优化200次。结果没有恢复有效组织结构：声速RMSE由53.70变为53.74 m/s，密度由49.86变为49.91 kg/m³。神经模型中的拟合误差0.00121，留出阵元0.98391；收敛物理复算的拟合误差0.25900。真实材料在神经模型中反而产生约0.506的拟合误差，当前代理目标尚不适合可靠材料反演。

完整图、数组和记录位于 `out/v5/reconstruction_round1_train_200_gel1mm`；初始耦合层未固定的对照完整保留。入口为 `nsrno.reconstruct_material`。本次没有重新训练网络，未进行完整物理伴随反演，所有计算已完成。


## 全部 11 角、密集有效频带与第二次实际联合重建

2026-11-05 的[第八轮报告](docs/forward_round8_dense.md)记录已完成的密集训练及实际反演。采用原始 V5 的全部 11 角、192 阵元及 294 个相邻频点（4.015–8.896 MHz，包含名义滤波器过渡带），保留约 99.9958% 开发验证 RF 能量。训练轮换 32 条记录并完成全部频率/角度覆盖，未训练完整介质与采集的笛卡尔积。

固定独立开发样本 train_200 的全部 294×11 采集上，完整 RF 相对 RMS 在 32 次迭代时由 0.3900 降至 0.1631；64 次为 0.02904，物理残差收敛后为 0.006733。以上显式给定真实衰减图，不能与仅输入 c/rho 的模型混作相同条件。当前准确频域实现尚不比原时域求解器更快。

随后从均匀初值以收敛物理层及隐式伴随联合优化 c/rho，完成两遍全频带、196 次更新。实际成像失败：声速 RMSE 53.70→53.56 m/s，密度 49.86→49.99 kg/m³；原时域求解器独立复算的完整 RF 误差 0.9802→1.0378，18–36 μs 也未改善。三个监测频率的候选参数正演与时域结果相符，目前主要待解决的是材料优化未降低完整目标，不能据此断言密度不可恢复。

检查点为 `out/v5/forward_round8_dense_full/last.pt`，接口为 `nsrno.dense_surrogate.DenseSurrogate`，需要相同采集参考及显式 alpha。有限深度预测用 `truncated(..., iterations=64)`，默认接口使用收敛求解和隐式梯度；两种成本和精度应分别报告。重建图、原始数组、完整覆盖与独立复算记录在 `out/v5/reconstruction_round2_dense_train_200`。本轮全部计算已结束；以前的结果、3 MHz 数据和最终留出病例均保留。


## 纯 UNO 三参数通道正演：第一轮

2026-11-05 的[实际报告](docs/uno_round1_triparam.md)记录已完成的纯神经算子实现和6000步训练。`nsrno.uno_rf.ChannelUNO` 使用二维空间UNO、连续神经积分读出和接收域UNO，从原生c/rho/alpha预测完整复IQ及RF；在线没有波动方程迭代、物理伴随或参考内部波场。它是通道RF的UNO改型，不是原论文网络的原样复现。

全部780条高频记录的衰减标签已恢复，使用原有48条3MHz配对，另生成32训练/16开发验证材料状态、两种源的全部11角，共96条新源记录。数据位于 `out/v5/uno_triparam_v1`。真实组织衰减只用于有监督正演输入；三参数反演入口将其作为未知量，保持已知数值边缘吸收带及频率幂指数1.1。

实际训练PyTorch峰值分配0.912GiB、保留1.160GiB；最终权重完整11角RF正演约0.0677秒，两源共22角三参数目标梯度约1.025秒。318条开发源记录、每条完整11角的RF相对RMS为0.71887，18–36微秒1.00041，独立衰减差分1.02511。训练集平均波形基准为0.72310，当前模型主要学到浅表共同回波，尚不适合联合重建。显存和速度改善不能当作等精度的物理替代。

推理权重为 `out/v5/uno_round1_triparam/model_weights.pt`，由固定开发监测选取第4000步；完整训练到6000步。接口 `nsrno.uno_surrogate.UNOSurrogate` 可独立加载权重及其中的归一化常数。该检查点仅供研究复现，尚未达到反演条件。`nsrno.reconstruct_uno` 已实现全观测梯度、线搜索和包含初值的候选选择，但当前会因正演精度不达标而拒绝启动。没有本轮三参数重建结果。全部本轮计算已结束，原模型与最终留出病例保留。
