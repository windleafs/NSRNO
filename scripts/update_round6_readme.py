from pathlib import Path
p=Path('/home_nvme/zhuangyang/nsrno/README.md'); text=p.read_text()
section='''
## 收敛误差与局部弛豫诊断

2026-11-05 的后续诊断见 [第六轮报告](docs/forward_round6.md)。两个开发病例各一个样本、三个频率、三个角度的 18 个观测上，同一第五轮方程充分求解后的合并通道谱误差从 0.68897 降到 0.33898，第四轮为 0.62372。该结果不是完整 RF 或全部验证切片的结果。单个验证观测的内部场对照表明，真实局部弛豫条件与当前共享系数近似之间仍有重要差别；报告记录了实际边界执行顺序和有限时间端点处理。

`nsrno.wave_solver.solve_frequency` 提供仅依赖现有模型和 c/rho 的收敛参考，返回实际残差和终止原因。它是数值参考，尚无隐式伴随反传，不能直接用于材料反演。原第四轮模型及所有训练检查点保留。本轮只完成诊断和参考实现，没有新的后台训练。
'''
if '## 收敛误差与局部弛豫诊断' not in text: p.write_text(text+'\n'+section)
