# 执行记录

- 初版保留了打包细节，但接收查询仍从粗特征网格插值。实现审查后改为 pixel_shuffle 恢复 50 μm 原生网格再查询。此前 local 29 轮、global 14 轮试运行保留于服务器的 forward_round3_*_coarse_query_pilot 目录，不计入正式对照。
- 正式版本 checkpoint spec 含 scatter_readout=native。新增可训练参数 local 26426、global 65918；两组均冻结原模型的 694786 参数。旧检查点仍可加载。
- 服务器约在 2026-11-04 12:42 UTC 重启，SSH 曾关闭连接，两组训练进程退出。13/15 轮的 history 文件未随进程退出更新，第一次恢复连接后曾误判为训练未中断，随后通过进程状态和 uptime 纠正。
- 对最后检查点进行了 ZIP CRC、模型加载、张量有限性检查，两组均通过；global 从第 13 轮、local 从第 15 轮恢复模型、优化器与学习率状态。恢复日志为各运行目录的 resume_after_reboot_1.log。原 train.log 含重启时未写回的空字节，保留为原始记录；指标使用原子写入的 history.json 和 checkpoint。
- 算子设计与分解谱算子的相关工作：Factorized Fourier Neural Operators, https://arxiv.org/abs/2111.13802 。本项目是独立实现，采用完整频率轴乘子和原生细节重排，不是该论文实验复现。
- 第三组增加由输入声速累计慢度确定的读取深度。均匀 1540、1500 m/s 的解析极限检查中，深度最大误差分别为 3.73e-9、9.31e-10 m；对均匀声速的导数相对误差分别为 1.75e-5、2.94e-5。只检查坐标与可微实现，没有重跑全波仿真。
- 正式 local、global 两组在第 60 轮按训练中期误差平台条件停止，选中第 0 轮。slowness 组完成预设 120 轮，选中第 70 轮；三组均完成基础训练集和开发验证集的全部 11 个角度评估。
- 对 slowness/best.pt 实施一次细节输入消融，结果保存为 detail_ablation.json。移除新分支的高通 c/rho 通道后，8–18 μs 的开发验证误差能量增加 12.09%。未继续追加架构、随机种子或训练预算。
