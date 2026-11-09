from pathlib import Path
p=Path('/home_nvme/zhuangyang/nsrno/README.md'); text=p.read_text()
section='''
## 3 MHz 配对数据与波动方程神经算子

2026-11-05 的首批低频数据已完成：32 训练、16 独立病例验证记录，每条 11 角；原始 RF 在 `out/v5/low_frequency_3mhz_v1`，训练格式缓存在 `out/v5/forward_cache_3mhz_v1`。材料与对应 V5 逐项匹配，最终留出病例未使用。当前 `train_wave` 的频率抽样限定原高频 V5；低频训练需要使用低频采集参数与弛豫系数。

论文分析和结果见 [第五轮报告](docs/forward_round5.md)，推导与实现选择见 [第五轮设计](docs/forward_round5_design.md)。保留完整相对压缩率和相对密度成像目标。新 `nsrno.wave_operator` 实现显式变密度波动方程残差迭代、可学习预条件器和双弛豫频散；原型检查点为 `out/v5/forward_round5_wave_relax_v1/best.pt`，使用 `load_wave_operator` 加载。全部 270 验证记录每条一个固定频率/角度的误差为 0.65497，原第四轮为 0.64633，因此尚未替代第四轮 RF 基准。单频可微接口与完整宽带反演不是同一完成程度。
'''
if '## 3 MHz 配对数据与波动方程神经算子' not in text:
    p.write_text(text+'\n'+section)
