# 材料相关的散射位置—阵元传播表示

这是第四轮独立 NSRNO 实现的方法说明；实际训练结果另存报告。代码不调用 Geo-Flow 模型，不生成新的全波 RF 标签。

## 传播时间与相干聚合

对散射位置 q=(x,z)、接收阵元 r 和发射角 theta，总到达时间包含发射和接收两段：

\[
\tau_{a,r}(q;c)=T_a(q;c)+T_r(q;c).
\]

接收路径为阵元到 q 的直线；时间是沿线对 1/c 的积分。发射路径为从阵列平面以固定角度 theta 到达 q 的直线，起点 `x_entry=x-z*tan(theta)`。起点时钟固定为 `t_ref + x_entry*sin(theta)/1540`，路径时间再对输入慢度积分。固定参考声速对照将两段路径中的 c 统一替换为 1540 m/s，发射边界时钟保持不变。

接收积分通过每阵元 513 个角度的射线扇形网格计算，轴向间隔为原生 50 μm。存储和插值路径平均慢度偏差，几何距离单独精确计算；因此均匀介质的时间不受角度插值曲率影响。发射积分使用倾斜坐标和轴向累计积分。两者都是直线路径近似，未求解弯曲射线、绕射、多路径或多次散射。

原生 log(c)、log(rho) 减去 9×9 平滑值后形成细节 dc、dr。两个源特征分别为 `(-2*dc-dr)/0.01` 和 `dr/0.008`。第一项是高通 log 压缩率，第二项是高通 log 密度；只有在弱扰动下才能分别近似相对压缩率和相对密度扰动。它们采用单极和偶极的角向基函数。这一选择依据 [Variable density linear acoustic inverse problem](https://pubmed.ncbi.nlm.nih.gov/8879095/) 的一阶散射关系，最终幅值和波形由神经读出学习，不把整个前端当作精确 Born 正演器。

对每个原生散射样点，按总时间 tau 将源贡献沉积到复数时间序列，同时乘以 `exp(-i*2*pi*f_c*tau)` 保留载频相位。时间沉积网格为现有 IQ 网格的四倍，之后带限转换为原来的 640 点表示。散射样值不作随机稀疏抽取。几何权重包括二维距离扩散、Hanning 发射孔径及密度项的入射—出射方向内积；不显式包含完整的波场振幅与衰减传播。

## 神经读出

冻结第三轮第 70 轮模型，在其输出上增加晚期修正。新网络有 24152 个可训练参数：8 组复数脉冲滤波器，以及依赖平滑声速、密度、累计慢度、接收坐标、时间、发射角和发射参考时钟的条件权重。脉冲滤波支持相邻三个阵元、33 个 IQ 时间样点；条件权重在较粗的接收—时间网格计算后插值，避免将任意高频相位交给权重网络生成。

新分支的输出从 16 到 18 μs 平滑开启。初始化时脉冲滤波器为零，预测严格等于本轮冻结基线。这个时间门作用于 IQ 修正，解码回带限 RF 后仍可能存在很小的跨窗影响，因此最终继续报告所有时间窗。

两组共享从材料相关特征的训练记录统计得到的归一化尺度，共享初始化、目标函数、训练角度安排、停止条件和冻结基线。只有新分支聚合中的传播时间不同；固定声速对照的其他输入仍包含声速，不能将其称作完全不使用声速的模型。

## 缓存与在线计算

缓存只保存由现有 c/rho 确定的传播特征、平滑背景和冻结模型输出，RF 标签继续从原缓存读取。传播特征用 float16 保存；冻结输出用 float32 保存。缓存配置信息保存原数据索引、传播实现和冻结检查点的哈希。

旧模型的 bfloat16 输出对角度批量大小有数值敏感性。本轮冻结基线统一为单条记录、全部 11 个角度的 float32 计算，再选择所需角度。两组共用这一基线，新分支的增益由本轮基线的实际评估值计算。

训练时使用缓存加快学习；`load_forward_model(...).predict_iq(medium, angles)` 则在线计算传播特征并保留对材料输入的可微依赖。缓存后的训练更新耗时不等于在线推理耗时，后者仍需要计算所有散射贡献。计算图可微也不等于其材料导数已与全波物理导数一致。

## 运行与结果位置

服务器项目为 `/home_nvme/zhuangyang/nsrno`，环境为 `/home/zhuangyang/miniconda3/envs/py310/bin/python`。在项目目录运行：

```bash
python -m nsrno.prepare_ray --delay material --out out/v5/forward_ray_material_v1
python -m nsrno.prepare_ray --delay reference --out out/v5/forward_ray_reference_v1
python -m nsrno.train_ray --delay material --features out/v5/forward_ray_material_v1 --out out/v5/forward_round4_material
python -m nsrno.train_ray --delay reference --features out/v5/forward_ray_reference_v1 --out out/v5/forward_round4_reference
```

生成缓存可用 `--shards` 与 `--shard-id` 分配互不重叠的记录，已有完整记录直接复用。训练目录已有 `last.pt` 时须显式 `--resume`，配置必须相同。正式运行程序通过环境变量选择 GPU，并限制 BLAS/OMP 线程数。

主要代码：`nsrno/ray_propagation.py`、`nsrno/ray_model.py`、`nsrno/prepare_ray.py`、`nsrno/train_ray.py`。通过 `nsrno/forward_model.py` 的模型工厂加载，旧版检查点保持兼容。
