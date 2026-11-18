# 两侧低频、中央高频阵列的二维反演实验

这个实验用于检验一个具体假设：**如果低频侧阵确实能独立发射和接收，把它们从平面移动到目标两侧的弧线上，能否改善浅、深层的材料参数恢复，以及压缩性与密度之间的串扰？**

程序比较四种采集方式，生成带噪声的复数频域散射数据，再使用同一个双参数 Born 反演器重建。默认数据由变密度 Helmholtz 正演生成，保留多次散射；反演仍是一阶 Born 模型。它是一个可以检查几何、频率和模型误差的研究基线，不是特定诊疗一体探头的数字孪生，也没有实现 CUTE 或 IMPACT。

所有命令均从 **NSRNO 仓库根目录**执行。实验独立于历史 V5 数据加载路径，不需要下载原有训练数据，也不需要 GPU。

本 PR 已实际运行 quick、匹配 Born 对照、quick 联合网格细化和 study，结果图、误差与复现命令见 [实测示例](example_results/README.md)。弧形侧阵在这些配置中改善了深层恢复，但两种材料参数的误差仍大，示例保留了这些限制。

## 1. 安装和第一次运行

建议使用 Python 3.10 或更高版本，并创建独立环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r experiments/theranostic_arc/requirements.txt
```

Windows PowerShell 的环境激活命令是 `.venv\Scripts\Activate.ps1`；其余 Python 命令相同。

先检查配置和资源估计：

```bash
python -m nsrno.theranostic \
  --config experiments/theranostic_arc/quick.json \
  --out out/theranostic_quick \
  --dry-run
```

运行默认实验：

```bash
python -m nsrno.theranostic \
  --config experiments/theranostic_arc/quick.json \
  --out out/theranostic_quick \
  --threads 1
```

默认 `data_mode` 是 `fullwave`。`--threads 1` 限制相关数值库的线程数，便于控制运行资源；本实验使用 CPU 稀疏直接求解和显式灵敏度矩阵。运行时间和内存取决于网格、频率数、阵元数以及本机数值库，不能由文件名中的 `quick` 推断。

`--out` 必须是尚不存在或内容为空的目录；实际运行会拒绝向已有结果的非空目录写入。再次运行时换一个输出目录，例如 `out/theranostic_quick_seed2`。`--dry-run` 只打印规模估计，不进行正演，也不创建输出文件。

查看完整参数：

```bash
python -m nsrno.theranostic --help
```

运行针对该实验的测试：

```bash
python -m unittest discover -s tests -p 'test_theranostic.py' -v
```

## 2. 四种采集方式分别回答什么问题

| 场景名称 | 高频收发 | 低频收发 | 要检查的变化 |
| --- | --- | --- | --- |
| `central_hf` | 中央直线阵 | 无 | 单侧高频基线 |
| `central_dual` | 同一个中央高频阵 | 位于中央相同宽度内的理想低频阵 | 增加低频数据的作用 |
| `flat_dual` | 同一个中央高频阵 | 两侧平面低频阵，全部位于 `z=0` | 扩大投影孔径的作用 |
| `arc_dual` | 同一个中央高频阵 | 两侧弧形低频阵 | 在相同低频阵元数、频率和投影位置下，改变高度及朝向的作用 |

`central_dual` 是理想的频率对照：中央低频阵与中央高频阵是不同的虚拟端口，不能据此假设实际高频阵具有低频接收带宽。中央低频阵的阵元总数等于两侧低频阵元数之和。`flat_dual` 与 `arc_dual` 使用相同的低频阵元数量、宽度、频率和 `x` 坐标。

建议依次看这三个差值，而不是只看最后一幅图：

1. `central_dual` 相对 `central_hf`：增加低频观测后发生了什么？这也增加了观测数，不能完全归因于频率本身。
2. `flat_dual` 相对 `central_dual`：保持低频观测数量后，向两侧展开投影孔径是否有效？
3. `arc_dual` 相对 `flat_dual`：保持投影孔径后，弧形侧视位置和阵元朝向是否带来额外收益？这个差异同时包含传播距离和朝向变化，不能解释为单一“散射角效应”。

### 低频发射并不自动产生高频接收数据

本实验只使用同频率、同能力频带的收发组合。每个低频端口既能发射，也能在低频接收；左右侧阵之间的低频收发也包含在内。高频中央阵只参与高频观测。不同频率的观测在联合反演中共享同一组材料参数。

可以关闭侧阵的低频接收能力，检查硬件前提：

```bash
python -m nsrno.theranostic \
  --config experiments/theranostic_arc/quick.json \
  --out out/theranostic_no_side_receive \
  --no-side-receive \
  --threads 1
```

在当前互不重叠的低、高频频带下，关闭侧阵低频接收以后，`flat_dual` 和 `arc_dual` 只剩中央高频观测。程序不会把低频发射波凭空转换为高频中央阵可接收的信号。`central_dual` 的独立理想低频参考阵仍是频率对照。

因此，使用真实诊疗一体探头之前，必须确认侧面低频器件是否具有可用的接收链路、是否独立寻址，以及中央阵能否接收相应低频。只有低频大功率发射能力，并不足以实现这里的侧阵反演方案。

## 3. 几何模型和参数

坐标约定为 `x` 水平、`z` 向深部增加；长度使用 mm，频率使用 MHz，材料数组顺序为 `[z, x]`。中央高频阵位于 `z=0`。

侧阵的中心和法向为

$$
\mathbf r(\theta)=
\begin{bmatrix}R\sin\theta\\z_f-R\cos\theta\end{bmatrix},
\qquad
\mathbf n(\theta)=
\begin{bmatrix}-\sin\theta\\\cos\theta\end{bmatrix}.
$$

默认 $R=z_f=30\,\mathrm{mm}$，两侧角度分别取 $[-70^\circ,-45^\circ]$ 和 $[45^\circ,70^\circ]$。这些侧阵在成像平面内部分包围目标，外侧阵元的位置本身已经深入到 `z>0`。它不是把原来的平面探头单纯加宽，也不是完整环阵。

**如果实际探头的“两边低频、中间高频”位于高程方向，即垂直于所成像的 `x-z` 平面，那么这个二维几何不适用。** 不能把高程方向的阵元直接投影成这里的面内侧阵；这种情况需要三维声场模型。配置目前仅接受 `geometry.plane = "xz"`。

### 两个预设

| 配置项 | `quick.json` | `study.json` |
| --- | --- | --- |
| 中央高频阵元数 | 12 | 20 |
| 每侧低频阵元数 | 4 | 6 |
| 中央阵宽度 | 38.2 mm | 38.2 mm |
| 高频有限线宽度 | 0.35 mm | 0.35 mm |
| 低频有限线宽度 | 1.0 mm | 1.0 mm |
| 低频采样点 | 0.15、0.25 MHz | 0.25、0.40、0.55 MHz |
| 高频采样点 | 0.35、0.45 MHz | 0.65、0.80、1.00 MHz |
| 正演/逆算子基础网格间距 | 0.35 mm | 0.14 mm |
| 重建系数网格间距 | 1.0 mm | 0.8 mm |
| 两张材料图的系数总数 | 1,650 | 2,542 |
| 包含 PML 的基础网格节点数 | 46,990 | 290,260 |
| 每侧 PML 请求厚度 | 10 mm | 10 mm |
| 内置对比度系数 | 0.04 | 0.04 |
| 默认信噪比 | 40 dB | 40 dB |

物理计算区域均请求 `x∈[-34,34] mm`、`z∈[-4,40] mm`，未知参数仅位于 `x∈[-12,12] mm`、`z∈[2,34] mm`。网格和 PML 的整数单元取整会让实际计算边界略微向外延伸。ROI 外被明确假设为已知均匀耦合介质。

这些是稀疏虚拟阵元，不代表任意商用探头的阵元间距、阵元数量或机械结构。这里的“高频”是相对于低频组而言；`quick` 的高频仅为 0.35/0.45 MHz，`study` 的最高频率也只有 1 MHz。预设频率用于控制计算量，不是对用户实际诊断阵频谱的估计；替换为真实频段必须重新检查波长采样、网格收敛和资源预算。`study` 同时改变频率、阵元数和重建网格，所以它与 `quick` 的差异不能单独解释为某个因素的效果；需要隔离变量时应复制配置，只改变一个因素。

## 4. 正演、参数定义和 Born 反演

### 4.1 两个独立的材料参数

定义压缩性 $\kappa=1/(\rho c^2)$、逆密度 $\eta=1/\rho$，以及相对背景对比度

$$
m_\kappa=\frac{\kappa}{\kappa_0}-1,
\qquad
m_\eta=\frac{\eta}{\eta_0}-1.
$$

默认背景为 $c_0=1540\,\mathrm{m/s}$、$\rho_0=1000\,\mathrm{kg/m^3}$。重建时两张对比度图是独立未知量，不预设经验性的声速—密度对应关系。最后转换为

$$
\rho=\frac{\rho_0}{1+m_\eta},
\qquad
c=c_0\sqrt{\frac{1+m_\eta}{1+m_\kappa}}.
$$

因此，声速图看起来合理并不意味着密度或压缩性也准确；阅读结果时应同时检查两张原始对比度图及其串扰。

### 4.2 线性声学中的完整波场

采用 $e^{-i\omega t}$ 时间约定，归一化方程为

$$
\nabla\!\cdot\!\left[(1+m_\eta)\nabla p\right]
+k_0^2(1+m_\kappa)p=-s,
\qquad k_0=\omega/c_0.
$$

代码使用二阶通量形式离散、边界 PML 和稀疏 LU。逆密度影响离散通量，压缩性影响质量项。每个频率的同一介质矩阵在多个发射源之间共享分解。

`fullwave` 指求解非均匀介质中的完整线性声场，保留多次散射，以及参数到数据映射的非线性；它**不**表示非线性声传播、谐波产生或治疗声场热效应。

每个观测为已知背景参考相减后的复数声压：

$$
d_{tr}(\omega)=P_rp_t(m;\omega)-P_rp_t(0;\omega).
$$

这里 $P_r$ 是接收有限线上的加权平均。参考介质、源、接收位置和系统标定均视为已知；真实实验中的参考失配、直达波去除和电子系统响应需要另外处理。

### 4.3 Born 导数

设离散方程为 $A(m)p=-s$，背景解为 $A_0p_0=-s$。一阶扰动满足

$$
\delta p=-A_0^{-1}\delta A(m)p_0,
\qquad d\approx J_\kappa m_\kappa+J_\eta m_\eta.
$$

实现中的 $J$ 是本离散方程在零对比度处的导数，包含逆密度的面通量导数。构造它不需要远场近似。互易场乘积中不加复共轭，而对实数材料参数的目标函数求梯度时使用 $\operatorname{Re}(J^Hr)$；二者不能混用。

材料参数通过矩形 ROI 上的双线性基函数表示，ROI 外严格置零。这是显式的已知支撑先验，不是算法从数据里恢复出的背景。ROI 边界处允许不连续，内置模体在边界附近为零；导入自己的模体时也应检查边界。

### 4.4 阵元和激励的理想化

每个阵元用有限线段和 Gauss–Legendre 积分表示，默认 5 点积分；发射、接收采用互易的线段权重。阵元法向用于确定线段朝向，不额外叠加经验余弦方向图。有限宽度本身产生空间平均及其方向响应。

线源向两侧辐射，不包含刚性背衬、壳体遮挡、匹配层、透镜或实际机电转换函数。每个发射源归一化为**单位积分源强**，并不保证不同频率、不同宽度、不同几何下具有相同辐射能量或相同声功率。这里的比较不能直接转换为同等电压、同等功率或同等治疗剂量下的硬件性能比较。

模型也不包含组织衰减、弹性剪切波、非线性谐波、三维扩散或探头与组织的接触力学。PML 只吸收计算边界的出射波，不是组织衰减模型。

## 5. 两种数据模式的用途

### 默认：完整波场数据，Born 模型反演

```bash
python -m nsrno.theranostic \
  --config experiments/theranostic_arc/quick.json \
  --out out/theranostic_fullwave \
  --data-mode fullwave \
  --threads 1
```

内置模体直接在正演网格节点上计算，不先投影到逆问题的粗基函数。因此默认实验同时包含多次散射造成的 Born 误差，以及有限基函数不能准确表达连续模体的表示误差。数据和逆算子仍可使用同一基础差分网格；这并不等于已完成独立求解器验证。

默认对比度系数 0.04 只是弱对比度起点，不代表任何指定组织，也不能保证所有频率、目标尺寸和传播距离下 Born 误差都小。增大对比度、厚度或频率之后，要重新检查留出数据和完整波场回放。

### 对照：匹配离散和基函数的 Born 数据

```bash
python -m nsrno.theranostic \
  --config experiments/theranostic_arc/quick.json \
  --out out/theranostic_born_control \
  --data-mode born \
  --threads 1
```

这个模式有意让合成数据与逆算子的离散、参数基函数相匹配，用来检查几何条件、噪声、求解和正则化。它是一种受控的“匹配模型”实验；漂亮的结果本身不能证明对完整波场、真实组织或实际探头的有效性。

具体地，程序先在重建系数节点上采样模体，组成 `truth_coefficients`，再用 `J @ truth_coefficients` 生成干净数据。图像评价中的真值也使用这组系数经双线性基函数插值后的图，即 **basis-projected matched truth**，不会把连续模体与这组粗系数的表示差异算作 Born 对照的重建错误。`summary.json` 的 `image_truth` 明确记录当前评价对象，`continuous_truth_projection_error` 则另行报告粗基函数真值相对于原连续/导入模体的表示误差。

若 Born 对照已经恢复不佳，先排查观测不足、参数串扰、正则化和分辨率。若 Born 对照良好而默认 `fullwave` 明显变差，再考虑 Born 模型误差和表示误差，而不是直接增加正则化掩盖它们。

## 6. 网格误差：两种细化不能混为一谈

最短波长的网格点数为

$$
\mathrm{PPW}_{\min}
=\frac{c_{\min}}{1000\,f_{\max,\mathrm{MHz}}\,h_{\mathrm{mm}}}.
$$

程序检查背景及实际介质的最低波长采样。至少 8 PPW 是拒绝明显欠采样的保护阈值，**不是相位准确度保证**。二阶离散在长传播路径上仍可积累明显相位误差，而且不同几何的路径长度不同，可能造成有偏的几何比较。`quick` 和 `study` 都应作为试验配置，不能省略网格收敛检查。

### 6.1 同时细化数据和逆算子：网格收敛检查

`--grid-refine N` 把基础波场网格间距除以 `N`，同时作用于数据正演和逆算子。它保持重建基函数网格、阵元和随机种子不变。

```bash
python -m nsrno.theranostic \
  --config experiments/theranostic_arc/quick.json \
  --out out/theranostic_grid2 \
  --grid-refine 2 \
  --dry-run
```

```bash
python -m nsrno.theranostic \
  --config experiments/theranostic_arc/quick.json \
  --out out/theranostic_grid2 \
  --grid-refine 2 \
  --threads 1
```

与基础 `--grid-refine 1` 的结果比较，检查浅、深层指标、几何排序、复数据及留出误差是否稳定。有资源时继续比较 2 与 4；若仍在明显变化，就不能把差异解释为真实的几何优劣。

默认噪声以该次运行的中央参考信号 RMS 为尺度，所以细化后即使随机种子不变，绝对噪声幅值也可能略变。应查看输出中记录的噪声尺度；需要隔离纯数值误差时，可在配置副本中将 `experiment.snr_db` 设为 `null`，并保持两次运行其余参数相同。

### 6.2 只细化数据正演：离散失配压力测试

`--forward-refine N` 在基础网格之上，仅进一步细化合成数据的正演网格。逆算子仍保留基础网格。因此，下面这个命令检验的是**数据与逆模型的离散失配**，不能作为数值收敛证明：

```bash
python -m nsrno.theranostic \
  --config experiments/theranostic_arc/quick.json \
  --out out/theranostic_forward2 \
  --data-mode fullwave \
  --forward-refine 2 \
  --threads 1
```

当两个选项同时使用时，逆算子网格间距是原始 `spacing_mm / grid_refine`，数据正演间距进一步除以 `forward_refine`。`forward_refine` 应用于完整波场压力测试；匹配模型的 Born 对照必须保持它为 1，程序会拒绝 `born` 与 `forward_refine>1` 的组合。

### 6.3 资源保护

默认资源保护参数是

```text
--max-grid-nodes 400000
--max-jacobian-mb 768
```

它们分别限制计算网格规模和显式灵敏度矩阵的估计占用。二维网格间距减半后节点数约增至 4 倍，稀疏 LU 分解的内存、时间增长还取决于填充；因此这些上限不是总内存保证。

每次增加频率、阵元数、网格细化倍数或重建基函数数量，都先执行 `--dry-run`。例如 `study` 再细化一次通常会超过默认网格上限。只有确认本机资源足够后才显式提高上限；也可以缩小物理区域和 ROI，前提是仍保留所有阵元、目标、PML 及足够边界距离。不要通过粗化波场网格来绕过最高频率的采样要求。

## 7. 噪声、留出观测和共同正则化

### 同一频率使用共同噪声尺度

每个频率的噪声尺度由中央阵的干净参考散射信号决定，相应频率的不同几何使用同一个绝对噪声尺度。高频参考来自中央高频阵，低频参考来自理想中央低频阵。这样，不会把每个几何单独归一化为相同相对信噪比，抹掉其接收到的绝对信号强弱差别。

低频各几何还共享相同的标准化复高斯噪声样本，高频中央阵的数据块在场景之间直接复用；噪声样本本身不强制互易对称。复噪声的 RMS 为记录的 `noise_sigma`，由中央干净散射 RMS 乘以 `10**(-snr_db/20)` 得到。

`snr_db` 是生成数据的设置，不代表临床设备的噪声模型；当前不包含频率相关电子噪声、通道相关噪声、运动误差或标定漂移。`snr_db: null` 关闭加性噪声；此时求解权重仍使用中央信号 RMS，避免不同频率的绝对幅度失衡。

### 留出整组端口，避免互易方向泄漏

默认 `holdout_stride=4`。采用从 0 开始的阵元索引：当某个发射或接收端口满足 `index % stride == stride - 1` 时，该 Tx–Rx 观测留出，不参加反演。留出端口既不作为训练发射端，也不作为训练接收端。因此互易配对的两个方向同时留出，不能通过反向观测间接进入拟合。

这不等于每四个数据点只留出一个；因为同时排除了相关行和列，实际留出比例更大。留出数据误差检验对未参与训练的端口组合的预测能力，不等同于对新组织、新探头或新频率的泛化能力。

### 所有几何使用同一个反演尺度

设 $W$ 对每个频率按噪声尺度做白化；无噪声时使用中央信号 RMS。以 `central_dual` 训练灵敏度矩阵 $WJ$ 各个非零列二范数的中位数作为共同尺度 $S$。所有场景解同一形式的问题：

$$
\min_{-b\le m\le b}
\frac12\left\|\frac{WJm-Wd}{S}\right\|_2^2
+\frac{\lambda_0^2}{2}\|m\|_2^2
+\frac{\lambda_g^2}{2}\|Dm\|_2^2.
$$

其中 $m=[m_\kappa,m_\eta]$ 为实数系数，$D$ 是对两张图分别施加的每毫米一阶差分。默认 $\lambda_0=0.03$、$\lambda_g=0.08$、$b=0.3$，各场景共享相同设置，不用真值分别调参。这里的正则化参数依赖上述归一化约定，不能直接搬到历史脚本的不同目标函数中。

代码将复数观测的实部和虚部叠成实数系统，使用 SciPy `lsq_linear` 求解增广有界最小二乘，保留其停止状态、LSMR 迭代信息及独立的投影梯度/KKT 检查。`scipy_success` 与本实验的 `converged` 不是同一个字段：不能只凭 SciPy 的成功状态忽略独立最优性检查。接口行为见[官方 `lsq_linear` 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html)。

共同尺度并不消除所有实验差异：更多观测仍意味着更多数据约束，正则化仍会影响欠定问题的解。若要研究固定采集时间或固定总发射能量，需要进一步设计匹配预算的采集方案。

## 8. 输出文件和结果阅读顺序

一次运行的主要输出为：

| 路径 | 内容 |
| --- | --- |
| `resolved_config.json` | 合并命令行覆盖后的实际配置 |
| `summary.json` | 运行规模、真值约定、代码与环境来源、诊断信息和汇总 |
| `wave_diagnostics.json` | 各次波场求解的线性残差、分解耗时及 LU 非零项数等 |
| `metrics.csv` | 各场景的定量指标，便于跨运行比较 |
| `geometry.png` | 阵元、朝向及矩形重建 ROI 的几何关系，不包含模体图 |
| `reconstruction.png` | 真值和四种采集方式的重建图 |
| `sensitivity.png` | 双参数灵敏度及条件性诊断 |
| `<场景名>/reconstruction.npz` | 重建、真值、物理参数和坐标 |
| `<场景名>/responses.npz` | 各频率/孔径的数据、留出掩码、尺度和预测响应 |
| `<场景名>/sensitivity.npz` | 原始系数网格上的局部双参数灵敏度诊断数组 |
| `<场景名>/metrics.json` | 该场景的详细指标 |

`reconstruction.npz` 中，`x_mm`、`z_mm` 是评估坐标，`m_kappa`、`m_eta` 是重建对比度，`truth_m_kappa`、`truth_m_eta` 是对应真值。`speed_m_s`、`density_kg_m3` 及其 `truth_` 版本给出物理量；`model_x_mm`、`model_z_mm`、`coefficient_m_kappa`、`coefficient_m_eta` 保存原始反演系数网格。材料图的顺序均为 `[z,x]`。

统一在独立的 0.25 mm 评估网格上比较插值后的重建与该模式的真值，分别统计浅层 `z≤15 mm` 和深层 `z>15 mm`。这只是统一误差评估的采样间距，不是重建空间分辨率。`fullwave` 评价对象为连续/导入模体，`born` 评价对象为基函数插值后的匹配真值；应先读取 `summary.json` 的 `image_truth`，再比较指标。`continuous_truth_projection_error` 在两种模式中均记录粗基函数的表示误差。

`responses.npz` 使用 `block_00_*`、`block_01_*` 等键。每个块含 `frequency_mhz`、`aperture`、`clean`、`observed`、`holdout_mask`、`weight_scale`、`noise_sigma`、`born_prediction`、`nodal_born_truth` 和 `coarse_born_truth`；启用回放时还含 `fullwave_replay`。例如 `block_00_clean` 是该块的干净散射数据，`block_00_observed` 是加噪后的观测。响应以 Tx 优先、Rx 次之的顺序展平，频率和孔径应从对应键读取，不要假设块的固定顺序。

在 `fullwave` 模式中，`nodal_born_truth` 使用数据正演网格上的真实材料直接计算 Born 响应，不经过粗反演基函数；`coarse_born_truth` 使用 `J @ truth_coefficients`。前者与完整波场干净数据的差异帮助检查 Born 线性化误差，后者还包含粗基函数表示和可能存在的正演/逆算子网格失配。对应的 `truth_nodal_born_response`、`truth_coarse_basis_born_response` 指标保存在场景 `metrics.json` 中。匹配的 `born` 模式下两项都等于其干净数据，不是额外的独立验证。

`summary.json` 还记录 `truth_maximum_absolute_contrasts` 和 `truth_outside_inversion_bounds`。若后者为 `true`，当前先验边界排除了部分真实材料；这会在日志中提醒，不能把受边界截断的重建直接归咎于孔径。`provenance` 保存 Python/NumPy/SciPy 版本、Git 版本号与工作区修改状态、模块源码 SHA-256，以及导入模体的路径和 SHA-256；与 `resolved_config.json` 一起保留，便于审计和复现。

建议按以下顺序检查：

1. **几何及资源**：`geometry.png` 是否符合实际设想？侧阵是否真的位于目标侧面？配置是否用了希望测试的频率和阵元数？
2. **数值有效性**：波场线性求解残差、PPW、优化停止原因是否正常？细化网格后结果是否稳定？较小线性方程残差只能证明代数系统解得准确，不能证明连续波动方程的离散误差小。
3. **独立参数恢复**：同时看 $m_\kappa$、$m_\eta$，检查只有一种参数发生变化的目标是否在另一张图中产生伪影。
4. **浅层与深层指标**：不要只看整幅图平均误差。局部改善可能被大面积背景稀释，也可能以另一深度区域的恶化为代价。
5. **留出预测**：训练拟合变好而留出数据变差时，不能据此宣称几何更有效。
6. **完整波场回放**：需要时检查恢复出的介质能否解释实际完整波场观测。

回放命令：

```bash
python -m nsrno.theranostic \
  --config experiments/theranostic_arc/quick.json \
  --out out/theranostic_replay \
  --data-mode fullwave \
  --replay-fullwave \
  --threads 1
```

该选项对恢复的介质再次进行完整波场正演，并检查留出响应，会增加计算量。它是当前模型内的一致性检查，不是独立算法或真实数据验证。

### 如何解读灵敏度图

灵敏度诊断包含单个位置上两种材料参数对应的 $2\times2$ 条件 Fisher 信息。它假设**其他位置的参数已知**，用于观察该位置两列响应的强弱和相似程度。

它不能反映所有像素同时未知时的全局零空间、长距离串扰和完整后验协方差。因此，局部条件数改善不等于整幅图唯一可恢复，也不能当作实际误差条。必须结合网格收敛、实际重建与留出预测来判断。

本说明不预填哪种几何获胜；以所运行配置对应的输出为准。一个弱对比度合成模体上的改善，也不足以决定临床组织或实际探头的最终路线。

## 9. 使用自己的声速和密度模体

通过 `--phantom path/to/material.npz` 导入模体。NPZ 至少包含：

| 键 | 形状 / 单位 |
| --- | --- |
| `x_mm` | 严格递增的一维横向坐标，mm |
| `z_mm` | 严格递增的一维深度坐标，mm |
| `speed_m_s` | `[len(z_mm), len(x_mm)]`，m/s |
| `density_kg_m3` | `[len(z_mm), len(x_mm)]`，kg/m³ |

例如，下面生成一个声速目标和一个独立的密度目标，所有目标位于默认 ROI 内：

```python
import numpy as np

x_mm = np.linspace(-12.0, 12.0, 121)
z_mm = np.linspace(2.0, 34.0, 161)
xx, zz = np.meshgrid(x_mm, z_mm)

def bump(x0, z0, rx, rz):
    r2 = ((xx - x0) / rx)**2 + ((zz - z0) / rz)**2
    return np.maximum(1.0 - r2, 0.0)**2

speed = 1540.0 * (1.0 + 0.015 * bump(-4.0, 9.0, 2.5, 2.0))
density = 1000.0 * (1.0 + 0.025 * bump(3.0, 26.0, 3.0, 3.0))

np.savez_compressed(
    "material.npz",
    x_mm=x_mm,
    z_mm=z_mm,
    speed_m_s=speed,
    density_kg_m3=density,
)
```

然后运行：

```bash
python -m nsrno.theranostic \
  --config experiments/theranostic_arc/quick.json \
  --out out/theranostic_custom \
  --phantom material.npz \
  --threads 1
```

导入时从物理量计算

$$
m_\eta=\frac{\rho_0}{\rho}-1,
\qquad
m_\kappa=\frac{\rho_0c_0^2}{\rho c^2}-1.
$$

声速、密度必须有限且为正，坐标必须完整覆盖配置的 ROI，不进行未知区域外推。如果输入文件还包含 ROI 外区域，那里的材料必须是配置指定的均匀背景；程序会拒绝被截断的 ROI 外未知组织，而不是悄悄删除它。

`medium.contrast_scale` 只控制内置模体；导入 NPZ 时使用文件中的实际声速和密度。更强对比度、低声速区域和更细结构可能要求更细的波场网格、更密的反演基函数以及更合适的反演模型。检查所导入对比度是否落在反演约束 `[-bound,bound]` 内，否则真值会被先验边界排除。

## 10. 建议的研究推进顺序

第一步跑默认 `quick` 和匹配 Born 对照，确认四个场景、独立参数以及浅/深层指标可以完整产生。然后进行联合网格细化，在离散误差受控之后再解释几何排序。

第二步只改变低频侧阵角度、数量或频率中的一个因素，同时保持其他配置及正则化规则不变。对照 `central_dual → flat_dual → arc_dual`，判断改善主要来自低频、投影孔径还是侧视位置；使用多个噪声种子及几个不同模体检查稳定性。

第三步改变模体对比度和复杂程度，加入层状边界、相邻组织和浅层小结构，比较 Born 对照、完整波场数据、留出预测和完整波场回放。若完整波场结果随对比度明显崩溃，而匹配 Born 对照仍好，才有证据把工作重点转向超越一次线性化的反演，例如带频率递进和合适初始化的迭代波场方法。

第四步把真实探头的频率响应、接收能力、阵元数、位置和方向录入。若两侧低频只有共同发射通道、没有独立接收，或者实际弯曲发生在高程方向，先调整实验模型；不能沿用当前虚拟面内侧阵的结论。

本实验适合回答“这个采集几何在当前物理假设下是否值得继续研究”。它不能单独证明 CUTE、IMPACT 或 Born 衍射层析哪个路线总体更好；这些方法的被估计量、数据处理和先验不同，需要另建公平对照。

## 11. 研究背景和实现参考

1. Moghaddam, M.; Chew, W. C. **Variable density linear acoustic inverse problem.** *Ultrasonic Imaging*, 15(3), 255–266, 1993. [DOI: 10.1177/016173469301500305](https://doi.org/10.1177/016173469301500305)，[PubMed PMID: 8879095](https://pubmed.ncbi.nlm.nih.gov/8879095/)。该工作讨论密度与压缩性的同时反演、多频观测和正则化；其平面波入射及远场接收的推导假设，与本实验的有限阵元、离散 Helmholtz 导数有所不同。本实验不是该论文的逐项复现。
2. SciPy 官方文档：[scipy.optimize.lsq_linear](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html)。用于核对有界线性最小二乘接口、停止状态和内部 LSMR 诊断；实际运行所用 SciPy 版本记录于 `summary.json`。
