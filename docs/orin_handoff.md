# 给下一台电脑 AI 的 Stage B 交接提示词

请在 `/home/.../g1-active-view` 仓库继续工作，先阅读：

- `README.md`
- `docs/model_contract.md`
- `docs/migration_inventory.md`
- `docs/validation_report.md`
- `models/manifest.json`
- `configs/execution.real.example.yaml`
- `configs/calibration.example.yaml`

Stage A 已完成离线推理整理。不要导入 Isaac，不要重训 VA-SFT，不要启动 GRPO，不要改变九动作表，不要连接机器人，直到先完成以下审查：

1. 查询 Orin JetPack/L4T、aarch64、Python、CUDA、PyTorch、OpenCV、内存和实际相机接口。
2. 执行 `scripts/prepare_models.py` 校验 LFS 权重和 SHA256。
3. 用真实相机做只读采集和回放，确认 RGB/BGR、分辨率、延迟、曝光和内参。
4. 实现独立的真实相机、云台编码器和安全状态适配器；默认动作仍为禁用。
5. 重新标定 pan/tilt 零位、方向、限位和动作步长，记录到新的真机配置，不覆盖 `execution.real.example.yaml`。
6. 在 shadow mode 对比 `raw_action` 与保护层动作，禁止发布运动命令。
7. 只有在人工审查真实图像和硬件信号后，才进行单步低速云台测试；身体六动作需要单独的 G1 安全审批。

可执行的当前基线命令和输入在 `README.md` 与 `docs/validation_report.md` 中，不能用自己生成的输出代替原实现基准。
