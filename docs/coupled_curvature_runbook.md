# 继续工作句柄

目标尚未完成；不得把下述数值核对或等待程序当成三参数成像。

## 当前真实运行（优先于下文历史记录）

耦合反演已于 2026-11-06 07:02 CST 实际启动：主 PID 1455582，工作 PID 1456325/1456326/1456327（GPU 0/1/2，point3/lf/hf），后处理 1455583，观察程序 1455584。实际参数为 `--chunk 96 --memory-limit-gib 4.25`。原生细场仍全自由，目标和真实梯度仍包含全部 11 次发射、192 接收。初始全数据梯度正在计算。启动器 1455578 正常退出；不要因它结束重启实验。

重算缓存从 128×7×961×1121 改为 96×7×961×1121 个 float32，GPU 缓存约 3.6→2.7 GiB，增加主机检查点存储。启动时 MemAvailable 63.90 GiB，首轮传播后约 28.02 GiB；整卡实测最高 14296 MiB（13.96 GiB）。完整耦合工作进程的实际反传峰值需等首个全数据响应，不能把 4.25 GiB 配置上限当作实测。

**自动审批已拒绝提前终止物理/走时两组旧优化。`retire_first_order_controls.py` 未执行，两组均继续原预算。除非收到用户针对这两组的明确新授权，不得重试该停止操作或用间接方式停止；广义研究目标不能替代这次审查要求。已通过异步问题请求具体授权，尚未收到答复；新方法现在能并行运行，因此目标不受该授权问题阻塞。**

为实现上述安全替代方案，仅替换了尚未开始 GPU 工作的 CPU 等待程序：1432050 已退出，1455578 使用 `--allow-concurrent-controls --memory-limit-gib 4.25 --chunk 96 --minimum-host-available-gib 48` 启动实际优化。两个旧优化及相位优化都未被停止。两组旧目录均不存在 `resource_decision.json`。

- 新耦合反演目录：`out/v5/reconstruction_coupled_v1_train_200`。
- 实际 CPU 等待 PID：1432050，`nsrno.launch_coupled_when_ready --initializer rf`。旧 PID 1419546 是已替换的 CPU 程序，没有启动过 GPU 反演。用 `/proc/PID/cmdline` 和 start_tick 核对；不能据一个 PID 文件、锁文件或安静日志重复启动。
- 等待条件：两组控制的原 main/worker/post 句柄释放、GPU 0/1/2 各自加 5.75 GiB 和 768 MiB 后不超过 16 GiB、MemAvailable 至少 32 GiB。它不会停止任何控制。
- 初始化：物理控制 `snapshots/stage0_step064` 中按完整 RF 选定并冻结的 `parameters.npy` / `material.npy`，不是相位初始化，也不读取真值。
- 真正优化 PID 仅在启动后写入 `workflow.json`；等待进程此后正常退出。此时应读实际首个完整梯度、第一组 `actual_coupled_metric_product`、内层 CG 和第一张接受材料图，核对新增运行的实际显存。
- 所有发射和接收进入真实全数据梯度及接受目标；编码仅用于正定近似曲率。不要将它称为精确几何 Hessian或三角度反演。

新增切线核 `native_tangent.cu` 保持当前已复核的哈希。首次 `native_tangent_v1/run.json` 因过严基态阈值记录 failed；`precision_review.json` 将此与同一旧库的实测重复性和相同机器码一起审查，`qualified_for_curvature_implementation=true`。不要因为原记录 failed 再跑旧检查或改 CUDA 代码去追求标志。切线/伴随内积差 2.67e-8，实际峰值 4.27 GiB；完整新耦合服务的峰值尚待实际读取。

当前相位主进程 1326143（工作进程 1326190/1326191，post 1326144，watcher 1326145）继续 64 次声速初始化、128 次完整 RF。第 32 次相位目标虽下降约 19.6%，c RMSE 变为 54.35、相关性 0.097，未恢复 c。alpha 完全不变，rho 随 Z/c 变化，不能叫三参数结果。

两控制：物理主 1003839 / workers 1004186、1004187 / post 1003840 / watcher 1097482；走时主 1075517 / workers 1075566、1075567 / post 1075518 / watcher 1097483。已看同次数 LF64+HF32 的实际三图，均接近均匀；密度相关性 −0.104/−0.097。计划在各自 HF64 完整记录后，依据实际图决定是否停止剩余重复一阶更新，保存 RF 选定最佳候选并写明预算改动。尚未停止，不能声称原 HF128 已完成。若执行停止，要核对 main、其 worker 的 PPID/start_tick 及 post/watch 的 out 参数，先保存稳定快照，停止 post/watch 后再停止该优化进程与其子进程，避免遗留 GPU 进程。

新后处理 `run_coupled_postprocess` 仅先评价最终三图。若图未恢复，不增加原求解器复算。正面结果还必须做原求解器全部发射复算、独立 alpha 配对和另一开发材料；Neg_47 留出不用于方法选择。
