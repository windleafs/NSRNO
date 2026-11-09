2026-11-07 15:28（北京时间）更新。本轮 3000 次微调和全部评估已完成，父子进程实际退出；完整两例地图及全部开发衰减差分图已下载并实际查看。三参数联合重建仍失败，详细结果见 raw_rf_window_balanced_round1/RESULT.md。以下“活动训练”观察保留为历史，不应重启该作业。

固定最终完整 RMSE：train_200 的 c/rho/alpha 为 63.6606/63.6645/0.126784，val_059 为 66.5699/64.5781/0.139620，均差于均匀初值。训练独立响应相对 RMS 误差 0.43152/0.92630/0.93663；开发 alpha 误差 1.00309，增益 0.03908。10 个同材料窗口的平均 c/rho/alpha 预测差从 37.307/19.780/.09167 降至 20.973/8.482/.03703，仍不小，且没有带来材料恢复。不能声称窗口问题已完全排除。

后继为 RECYCLED_KRYLOV_GN_DESIGN.md 的真实曲率 Krylov 原生重建，已确认本轮全部退出后于 07:29:02 UTC 启动，协调 3198826；当前实际状态及资源见 NATIVE_RECYCLED_KRYLOV_PROGRESS.md。计划首先新增最多 8 个 HVP，保护已接受物理候选。目标保持 active。

以下保留 15:00 及更早的观察。

07:00:18 UTC 的实际观察：协调 3127614、训练 3127615 的 start_ticks 与启动时一致，已记录 825/3000 步，训练约 511.56 秒，进程树采样峰值仍为 776 MiB。近期独立改变项训练 EMA 相对 MSE c/rho/alpha=0.46555/0.85181/0.96770，仍不支持联合恢复；该统计不是固定最终权重评估。代码和数据未改，继续原预算。

同时完成了一项不占 GPU 的真实曲率重求解：旧同参考点的 15 个 HVP 在其张成的子空间内充分校正后，仅额外获得 0.00015378 的二次预测下降，完整空间残差仍为初始值的 81.95%。候选完整图已实际查看，三材料仍近乎均匀，因此没有新传播评分、没有修改物理最优候选。详见 native_saved_span_round1/RESULT.md；此项已完成，不是第二个活动 GPU 实验。

本轮实际启动于 2026-11-07 06:51:35 UTC。协调父 PID 3127614，start_ticks=10013607；训练子 PID 3127615，start_ticks=10013633。boot_id=8a3b92f8-8715-4be0-830e-bfe9886bb8d3。06:52:49 UTC 已记录 100/3000 步，训练约 63.48 秒，父子实际存在；进程树采样显存峰值 776 MiB（0.76 GiB），Torch 分配峰值 0.37421 GiB。没有并行 GPU 实验。

输出：协调 out/v5/raw_rf_window_balanced_experiment_v1；模型 out/v5/inverse_raw_rf_window_balanced_v1；配对缓存 out/v5/raw_rf_window_balanced_data_v1。仅 GPU 0，分配器上限 8 GiB，整个进程树每卡限制 16 GiB。预算 3000 次 AdamW 更新或 5400 秒先到，固定最终 EMA .995，不按开发结果选择权重。

为何做这轮：旧固定权重对清零 60 μs 后 RF 数值不敏感（声速图改变 0.21–0.36 m/s），但同时改有效标记及时间后改变 38–45 m/s。这个对照说明记录模式处理是一个可操作障碍，尚不能单独区分伪相关、坐标处理和聚合归一化。实际短记录的旧模型地图也很差，因此不能把这一问题当作全部失败的解释。原始结果见 raw_rf_inverse_round1/RESULT.md。

实现保留原始 RF UNO 的全部结构与 11 发射、192 接收，输出独立 c/rho/alpha 和完整 1–43 mm 材料图，不强制平滑 c。输入仍为 1.5–4.5 MHz 有效频带的复包络、有效标记和精确采样时刻。训练数据仍为 Neg_07_Left 的 112 个材料状态，开发为 Neg_35_Left 的 32 个状态，不是独立患者数；预留 Neg_47_Left 未读。

从旧 6000 步最终 EMA 开始，权重 SHA256=80660016ca898323339accc525960e847ed5c8d68b56ad1dc8d83fadccf507f2，优化器重置、独立约束乘子保留。奇数步执行原地图及一个独立参数响应任务，每四个奇数步加入已有轨迹对；偶数步对同一材料的真实 RF60/RF90 同时监督地图并加入软一致性，两项权重均为 1，沿用 [80 m/s,100 kg/m³,0.2 alpha] 的地图尺度。每步恰有一次优化更新。3000 步时两类任务各 1500 次。独立扰动对的两成员始终使用相同时间窗，长记录在该任务中以 0.5 概率换成真实短视图。

复用 56 个训练材料和 10 个开发材料的实际短记录，无新传播、无新频率、标签和全局 RF 幅度尺度均未改变。原缓存嵌套短视图误继承了长记录的 SHA256 与来源描述，已在训练开始前仅修正元数据；原索引、run 和执行过的准备程序均保存在缓存中，provenance_correction.json 记录前后哈希。修正后的训练索引 SHA256=b40c7b5b45fc64b36ac581ef33d63831dfa26bf74e9557bac34f629acc94fc9f。部署时文件曾放错到项目根目录，已移入实际 nsrno Python 包后才启动，没有重复训练。

训练后依次执行固定最终权重的全部材料预测与独立响应评估、10 个开发配对视图及 4 次合成控制推断，最后由 CPU 输出 train_200、val_059 的完整原生地图。各阶段必须等待并实际回收上一子进程。完整记录仍是默认评估输入。不为明显失败的材料图追加原生传播评分，也不能引用旧模型 RF 评分。

判据：两个记录长度的预测更接近，单独并不表示重建成功。必须同时看完整材料结构、逐深度误差和独立 rho/alpha 响应。如果只降低了时间窗敏感性而这些项目仍失败，则该问题不是充分解释，应转向传播几何、函数覆盖或原生物理优化，不能反复调整一致性权重来代替研究问题。

协调程序冻结 14 份源码：run_raw_rf_window_balanced_experiment.py、train_raw_rf_window_balanced.py、train_raw_rf_inverse.py、raw_rf_inverse_uno.py、uno_rf.py、inverse_uno.py、pressure_resources.py、evaluate_raw_rf_inverse.py、evaluate_raw_rf_window_views.py、summarize_raw_rf_inverse_results.py、evaluate_inverse_uno.py、evaluate_time_material.py、v5_materials.py、native_gn_json.py。训练及后续阶段完成前不得修改这些源码。

观察时读取协调与训练 run、progress、resource_summary、评估 run 以及实际 /proc 的 PID/start_ticks。不能仅凭旧 run.json、短暂 SSH 失败或缺一条新日志判断活动状态。CPU 结果汇总须在协调最终状态且父子实际退出后执行，输出另存 out/v5/raw_rf_window_balanced_comparison_v1。
