# Stage A 验证报告

## 已执行命令

```text
unitree_sim_env/bin/python scripts/check_environment.py
unitree_sim_env/bin/python scripts/prepare_models.py
PYTHONPATH=src unitree_sim_env/bin/python scripts/replay_baseline.py
PYTHONPATH=src unitree_sim_env/bin/python -m pytest -q tests
```

## 结果

- 模型文件存在、不是 Git LFS 指针、字节数和 SHA256 均通过。
- DINOv2 源码由仓库内 vendored 目录加载；未调用 `torch.hub`。
- `pytest`：4 passed。
- 真实仿真样例 `normal_empty_history`：原实现和部署包的 logits、概率、动作 id/名称、DINO cosine similarity、SIFT 内点数逐项一致，最大差异 0（float 输出完全相同）。
- 真实仿真样例 `normal_history_8`：逐项一致，最大差异 0。
- 合成全黑样例 `no_features_stop`：用于无特征和 Stop 分支，原实现与部署包逐项一致；它不是现实任务照片，不能代替真实样例。
- 低头恢复测试：缺少 pitch/时间/安全信号时返回 `cannot_verify_missing_recovery_state`；提供明确状态并持续 3.1 秒后，才允许纯保护层产生 `GimbalPitchUp` 覆盖。
- Stop hold 测试：缺少稳定性样本返回 `missing_stability_samples`，不会填零判定成功。

基准输入文件为 `datasets/fixtures/current_082_xpos.png`、`target_082.png` 和 `blank.png`；其原始输出保存在 `tests/fixtures/original_output.json`，部署重放结果保存在 `tests/fixtures/deployment_output.json`。

## 未完成

当前本阶段在现有 `unitree_sim_env` 中完成了 `PYTHONPATH` 离线导入和运行验证。创建临时独立 venv 后执行本地 `pip install --no-deps --no-build-isolation -e` 时，被工具自动审批服务以 401/超时拒绝，命令未执行，不能把安装标记为通过；`pyproject.toml` 已提供标准安装声明，下一台电脑可在独立环境中执行。没有 Orin、TensorRT、FP16、真实相机或硬件执行验证。DINOv2 原缓存没有 git commit，因此只能固定 vendored source tree hash，不能提供真实 commit。
