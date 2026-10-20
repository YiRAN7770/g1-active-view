# Stage A 验证报告

本报告对应本地最终提交（用 `git rev-parse HEAD` 获取）。原项目
`/home/szu/Project/active_view_g1/gimbal_tvr` 在提取后未修改；本仓库没有远程
推送。

## A. 代码提取和数值一致性

原始推理入口是
`src/gimbal_tvr/monotone_stop_policy.py:MonotoneInference.predict`。基准输入、
原始输出和部署输出分别是：

- `tests/fixtures/baseline.json`
- `tests/fixtures/original_output.json`
- `tests/fixtures/deployment_output.json`

基准包含真实仿真图像 `current_082_xpos.png` 与 `target_082.png`、空历史、完整
8 步动作历史，以及相同全黑图的 Stop 分支。原始实现和部署包在三组样例的
动作 id/名称、logits、概率、DINO cosine similarity 和 SIFT 内点数上逐项一致，
最大绝对差为 0（浮点输出也完全相同）。`pytest` 在当前工作站执行结果为
`10 passed`。

实际执行：

```text
/home/szu/Software/anaconda3/envs/unitree_sim_env/bin/python \
  -m pytest -q /home/szu/Project/active_view_g1/g1-active-view/tests
```

结果：10 passed；仅有 vendored DINOv2 在没有 xFormers 时的性能 warning。

### “无特征 Stop、相似度 1.0”核查

该样例位于 `baseline.json` 的 `no_features_stop`：current 和 target 都是同一
张全黑图。相似度的来源是 DINO 特征的 cosine similarity，不是 SIFT 默认值；
相同输入经过相同编码器得到 `1.0` 是预期结果。SIFT 内点为 0，因为全黑图没有
可匹配的局部特征；SIFT 几何分支按原代码走零特征分支，策略仍原样输出 Stop。

这不等于现实任务成功。`OfflineRuntime` 对离线推理始终返回
`task_success=false`，且没有 Stop 保持的真实速度、漂移、碰撞和跌倒信号时，
`verify_stop_hold` 返回 `missing_stability_samples` 或
`missing_physical_signals`。因此“原始策略 Stop”“图像/匹配有效性”“Stop 保持
验证”“任务成功”四件事被分别记录，不能用 Stop 或相似度 1.0 代替任务成功。

边界测试实际覆盖：

| 项目 | 输入/条件 | 实际结果 |
|---|---|---|
| 相同无纹理图 | 两张相同全黑图 | raw action=Stop，DINO sim=1.0，SIFT inliers=0，`task_success=false` |
| 不同无纹理图 | 全黑图与均匀灰图 | SIFT inliers=0，图像有效性为 false，返回失败原因，不报告成功 |
| 无效输入 | 不存在的图像路径 | 在策略前抛出 `FileNotFoundError`/`ValueError` |
| 旧帧 | recovery 连续调用相同 timestamp | 返回 `invalid_or_stale_observation`，拒绝覆盖 |
| episode reset | 空历史及 `OfflineRuntime.reset()` | 空历史使用 8 个 padding id；reset 清除恢复窗口、冷却和计数 |
| 最近 8 步 | 9 步输入及完整 8 步基准 | 仅保留最近 8 步；基准输出逐项一致 |
| 低头恢复缺状态 | 缺 pitch、timestamp、步长或安全信号 | `cannot_verify_missing_recovery_state`，不覆盖 |
| 低头恢复有效状态 | 3 个持续观测、3.1 秒、低相似度且不确定 | 仅保护层覆盖为 `GimbalPitchUp`，并带审计字段 |
| Stop 保持缺状态 | 空样本或信号缺字段 | `verified=false`，不填零、不判成功 |
| 权重缺失 | 删除模型文件 | `MISSING` |
| 权重哈希错误 | 修改一个字节 | `CHECKSUM_MISMATCH` |
| LFS 指针误用 | 写入 LFS pointer 文本 | `LFS pointer, fetch required` |

SIFT 无匹配会使几何证据缺失，但不能仅凭 0 个内点断言任务失败，也不能忽略
该证据失效；部署输出同时暴露 `visual_inliers`、`valid`、`failure_reason`、
`image_validity` 和 `task_success`。模块没有时间戳输入时，旧帧检测责任留给
未来相机适配层；当前 recovery 接口在调用方提供 timestamp 时会拒绝重复旧帧。

## B. 独立安装和交付

### 工作站回归

当前工作站环境记录在 `docs/environment_workstation.md`：Python 3.11.15、
PyTorch 2.13.0+cu130、NumPy 1.26.4、Pillow 11.2.1、OpenCV 4.11.0；
`torch.cuda.is_available()` 为 False。本次验证使用 CPU，未改变系统环境。

已执行模型校验、仓库外离线推理和基准回放，命令入口会从脚本位置解析仓库，
不读取原项目 `PYTHONPATH`，也不调用 `torch.hub`。DINOv2 源码和权重均来自
仓库内路径。

### 独立 venv 安装

曾尝试在不继承 site-packages 的全新 venv 中执行：

```bash
rm -rf /tmp/g1-active-view-venv
/home/szu/Software/anaconda3/envs/unitree_sim_env/bin/python -m venv /tmp/g1-active-view-venv
/tmp/g1-active-view-venv/bin/python -m pip install --no-deps --no-build-isolation -e \
  /home/szu/Project/active_view_g1/g1-active-view
```

工具审批服务在该命令执行阶段返回 401/连接超时，命令未执行且 venv 未创建。
因此独立 `pip install -e .` 和安装后推理**未标记为通过**。这不是依赖或算法
失败；需要人工在允许执行本地 venv 命令的机器上运行上面的命令，再执行 README
中的 `check_environment.py`、`prepare_models.py`、`infer_offline.py` 和
`replay_baseline.py`。没有绕过审批，也没有修改系统 Python、CUDA 或驱动。

### 本地克隆交付验证

为避免使用工作目录中未跟踪文件，使用最终提交创建了新的本地克隆：
`/tmp/g1-active-view-final-clone`（跨设备硬链接克隆失败后改用
`git clone --no-local`，没有复制工作目录文件）。克隆 HEAD 通过
`git rev-parse HEAD` 与当前仓库一致，工作树干净。

在 `/tmp` 工作目录执行了：

```bash
/home/szu/Software/anaconda3/envs/unitree_sim_env/bin/python \
  /tmp/g1-active-view-final-clone/scripts/prepare_models.py
/home/szu/Software/anaconda3/envs/unitree_sim_env/bin/python \
  /tmp/g1-active-view-final-clone/scripts/infer_offline.py \
  --current /tmp/g1-active-view-final-clone/datasets/fixtures/current_082_xpos.png \
  --target /tmp/g1-active-view-final-clone/datasets/fixtures/target_082.png --history '[]'
/home/szu/Software/anaconda3/envs/unitree_sim_env/bin/python \
  /tmp/g1-active-view-final-clone/scripts/replay_baseline.py
/home/szu/Software/anaconda3/envs/unitree_sim_env/bin/python \
  -m pytest -q /tmp/g1-active-view-final-clone/tests
```

结果：模型校验 `MODEL_CHECK_PASS`；离线推理输出 `MoveXPos`、相似度
`0.8598791360855103`、`task_success=false`；回放与原始 JSON 一致；测试
`10 passed`。克隆中的两个权重均为实际 LFS 对象而非指针，`git lfs fsck`
返回 `OK`。

Git 管理检查：`git ls-files` 覆盖源码、配置、fixtures、文档、vendored
DINOv2、模型指针和校验脚本；`git lfs ls-files` 覆盖两个权重；无仓库外软链接；
`.gitignore` 排除了虚拟环境、缓存、训练输出、ROS build/install/log、rosbag
和本机密钥。当前分支没有远程发布授权，未执行 push。

## C. 尚未验证的能力

- 独立 venv 的 `pip install -e .` 及其安装后推理：被本次工具审批服务阻塞，
  按上文命令待人工执行。
- NVIDIA GPU、Orin JetPack/L4T、aarch64、TensorRT、FP16 和性能指标：未验证。
- 真实相机、云台编码器、G1 身体控制、串口、ROS/DDS、Unitree SDK 和任何运动：
  未连接、未初始化、未发送命令。
- 真机成功阈值和标定参数：没有发明或填入；配置中的未标定项保持显式未配置。

DINOv2 缓存没有 Git metadata，原始 commit 为 `unknown`。仓库固定实际 vendored
源码，不自动下载；`scripts/hash_tree.py` 定义源码树哈希：按 POSIX 相对路径
排序，逐项写入 `relative_path_bytes + NUL + file_sha256_digest`，排除
`__pycache__/**`、`*.pyc` 和 `*.pyo`。原始和部署 `dinov2/` 子树哈希均为
`1bd32991d59172509305a98afc73f91e4b42532c36bf01ae70e2d3335ed7488a`。

完整原/部署文件 SHA256、模型选择证据（`va_sft_577_best.pt` 对应原
`va_sft_577_best/checkpoint/best.pt`，manifest 标注 `round11_low_head_replay`、
577 条轨迹、`visual_stop_residual`）见 `docs/source_provenance.json`。
