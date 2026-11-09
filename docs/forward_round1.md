# NSRNO 首轮神经正演代理

模型输入是同一 V5 仿真域上的声速和质量密度，两张图按 `[c, rho]` 排列，单位为 m/s 和 kg/m³，网格为 `[z,x]=[961,1121]`。输出为所选平面波角度的带符号通道 RF。默认输出全部 11 个角度，张量顺序为 `[batch,angle,receiver,time]`，每角 192 个阵元、2401 个时刻，采样率为 40 MHz。目标沿用原始 RF 的固定均匀参考相减和带通处理。模型条件包括两张介质图和固定配置中的发射角。

## 数据与模型

训练使用 `Neg_07_Left` 的 270 条基础记录和 240 条补充记录。验证使用 `Neg_35_Left` 的 270 条基础记录。切片、父样本及其变体跟随病例。最终留出病例为 `Neg_47_Left`。从原生仿真配方重建密度和声速，用存档声速和反射率标签核对配对。780 条介质和 RF 配对已准备完成。

编码器将每个 4×4 空间块重排到通道维，保留细尺度密度样值，同时输入轴向累计慢度和物理坐标。共享多尺度解码器以发射角和参考时钟为条件，在接收阵元和快时间网格上输出 640 个复基带样本。模型学习相对训练病例平均波形的变化，逆傅里叶变换恢复有符号 RF。平均波形和归一化尺度只使用训练病例的基础记录。640 点表示的最大相对 RMS 截取误差为 0.0911%。

训练按 0–8、8–18、18–36、36 μs 至记录末端四段平衡平方误差。每条记录每轮取两个轮换角度。按验证病例的 −8°、0°、+8° 的平均分窗相对 RMS 选择检查点。最终验证在同一验证病例上评估全部 11 个角度。

## 首轮结果

| 训练目标 | 最佳轮次 | 11 角验证相对 RF RMS | 四窗平均相对 RMS |
| --- | ---: | ---: | ---: |
| 训练病例的平均波形 | — | 0.728068 | 0.907798 |
| 固定训练分窗尺度 | 4 | 0.728069 | 0.907822 |
| 逐记录分窗尺度 | 8 | 0.728069 | 0.907800 |

固定分窗目标在平均波形基线处的损失有 94.0% 来自 10 条 `calibuni` 记录。逐记录版本使用当前目标 RF 的分窗能量与训练分窗尺度平方中较大者做分母，保留相同的全部 510 条训练记录。改变权重后，两次训练的验证结果都与平均波形相当。逐记录版本在训练病例基础记录上的相对 RF RMS 为 0.701512，平均波形同样为 0.701512。单条训练记录 `train_000` 的 0° 诊断中，128 次优化使分窗损失从 0.489 降至 0.420。梯度能改变单条波形，但本轮联合训练没有产生可度量的样本条件收益。现有结果还无法区分几何解码对传播相位的表达能力与联合优化难度各自的贡献。

## 复现训练

```bash
cd /home_nvme/zhuangyang/nsrno
PY=/home/zhuangyang/miniconda3/envs/py310/bin/python
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -m nsrno.prepare_forward --manifest out/v5/manifest.json \
  --out out/v5/forward_cache_v1 --workers 12
CUDA_VISIBLE_DEVICES=0 "$PY" -m nsrno.train_forward \
  --cache out/v5/forward_cache_v1 --out out/v5/forward_round1 \
  --epochs 10 --batch 4 --eval-batch 4 --angles-per-record 2 --workers 4
CUDA_VISIBLE_DEVICES=0 "$PY" -m nsrno.train_forward \
  --cache out/v5/forward_cache_v1 --out out/v5/forward_round1_balanced \
  --epochs 10 --batch 4 --eval-batch 4 --angles-per-record 2 --workers 4 \
  --loss-scaling record-window
```

已有训练状态时，使用相同配置加 `--resume` 从最新完整轮次恢复。各目录的 `best.pt` 为按验证选出的权重，`last.pt` 含优化器和学习率计划。逐轮结果在 `history.json`，全部验证角度的结果在 `validation_all_angles.json`。短拟合诊断在逐记录版本的 `fit_diagnostic.json`。

## 读取模型

```python
import numpy as np
import torch
from nsrno.forward_model import load_forward_model

device = "cuda:0"
model = load_forward_model("out/v5/forward_round1_balanced/best.pt", device)
medium = np.load("out/v5/forward_cache_v1/train_200.medium.npy")
medium = torch.from_numpy(medium)[None].to(device)
with torch.no_grad():
    rf = model(medium, angle_indices=[0, 5, 10])
# rf.shape == (1, 3, 192, 2401)
```

在可微调用中移除 `torch.no_grad()`，并对介质输入启用梯度。
