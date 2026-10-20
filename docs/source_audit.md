# 原项目源码审查记录

审查对象：`/home/szu/Project/active_view_g1/gimbal_tvr`

- `AGENTS.md`：在 `/home/szu/Project` 下未找到。
- Git 分支：`g1-gimbal-4dof-body-xy`。
- HEAD：`4e5507b Restart distance GRPO with dense rewards and timed Stop verification`。
- 工作区：审查时有 1605 个 Git 状态项，包含已删除旧训练/RL 输出和未提交新增文件；原项目没有被本阶段修改。
- 新仓库与原项目并列，原项目只作为只读来源。

## 推理闭包结论

实际入口是原 `src/gimbal_tvr/monotone_stop_policy.py:MonotoneInference.predict`。当前保留 checkpoint 的 `variant=visual_stop_residual`，因此加载 `VisualStopResidualPolicy`。它继承 `MonotoneStopPolicy`，其运动头是冻结的 `MagnitudePolicy`；Stop logit 由单调距离头加 residual 得到。`MagnitudePolicy` 又依赖 `relative_visual_policy.py` 的 geometry、DINO `VisualEncoder`、ImageNet 预处理和动作历史编码。`dedicated_policy.py:ActionPolicy/Inference` 是同一仓库的早期纯视觉策略，但没有被当前 checkpoint 的 variant 使用。

运行时真正读取的文件是 checkpoint、DINOv2 源码、DINOv2 ViT-B/14 权重、当前 RGB、目标 RGB和动作历史。训练数据、Isaac/USD、GRPO reward、执行器和目标位姿不进入离线 policy `predict`。原代码的 DINO 路径和权重路径是绝对路径，部署版已改为显式传入路径；原缓存没有 git metadata，因此 DINO commit 无法确认，改为 vendoring 实际源目录并记录 tree hash。

原始代码的仿真相机与 `joint_controller.py` 是 Isaac Xform 实现，不被部署包冒充为真机。原始 `grpo_stop_hold.py` 包含仿真执行器调用；部署包只保留需要真实状态输入的 Stop verifier，缺少状态时返回无法验证。
