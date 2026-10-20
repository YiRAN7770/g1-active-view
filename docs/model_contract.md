# 模型契约（由源码核实）

## 推理闭包和组合关系

原实现的真实入口是 `src/gimbal_tvr/monotone_stop_policy.py:MonotoneInference.predict`。当前 `best.pt` 的 `variant` 是 `visual_stop_residual`，因此实际组合为：

```text
VisualStopResidualPolicy
  -> MonotoneStopPolicy
      -> MagnitudePolicy（前 8 个运动动作）
  -> 可学习 Stop residual（只修正 Stop logit）
  -> argmax
```

`MagnitudeInference` 和 `RelativeInference` 是历史/替代入口，当前 checkpoint 不由它们加载。`dedicated_policy.py` 提供原始 DINOv2 编码器、预处理、动作历史编码；部署包将同样逻辑拆到 `vision/dinov2_encoder.py`、`vision/sift_geometry.py` 和 `policies/visual_stop_residual.py`。

## checkpoint

`best.pt` 是 PyTorch `torch.save` 字典，键为 `policy`、`actions`、`variant`、`metadata`。网络定义必须使用 `VisualStopResidualPolicy`；加载时执行 `map_location`，并严格检查九动作表和 variant。权重 SHA256 见 `models/manifest.json`。

## DINOv2

- 变体：`dinov2_vitb14`，ViT-B/14，`pretrained=False` 后加载本地 `dinov2_vitb14_pretrain.pth`。
- 源码：`third_party/dinov2`，工作站缓存显示 package `0.0.1`。
- 原缓存没有 `.git` 元数据，commit 无法确认；因此本仓库 vendored 了实际被调用的源码，并记录 source tree SHA256 `1bd32991d59172509305a98afc73f91e4b42532c36bf01ae70e2d3335ed7488a`。不得把未知 commit 写成已知 commit。
- 特征：`forward_features` 的 `x_norm_clstoken` 与 `x_norm_patchtokens` 拼接；patch token 转为 `[B,768,H/14,W/14]` 后 adaptive average pool 到 4×4，最终 768×17=13056 维。
- 部署不调用 `torch.hub`，不访问网络。

## 图像输入和相似度

DINO 输入是 RGB 图像：PIL 转 RGB，双线性 resize 到 280×210，转 CHW，float32，像素除以 255，随后按 ImageNet mean `[.485,.456,.406]`、std `[.229,.224,.225]` 归一化。SIFT 分支不是 DINO 输入：用 OpenCV 读取/转换灰度图，再 resize 到 640×480。

当前相似度为 DINO 特征的 cosine similarity：`cos(current_feature, target_feature)`，范围裁剪到 `[-1,1]`。SIFT 几何特征不参与这个 cosine，而参与动作分类器。无 SIFT 特征、少于 8 个匹配点、基础矩阵 RANSAC 不足 8 个内点或单应矩阵失败时，几何分支按源码使用零数组；不会伪造匹配。SIFT 使用 `nfeatures=1500`、`contrastThreshold=.02`、BFMatcher 双向 Lowe ratio `.8`、Fundamental RANSAC 阈值 2、置信度 .995、Homography RANSAC 阈值 3。

## SIFT 几何特征

匹配坐标差除以 `[640,480]`，加入匹配数/内点数/内点比例/灰度平均差、delta 分位数、距离分位数、3×3 空间网格中位移和计数、9 个采样点单应流；最终 `GEO_DIM=61`，对非有限值替换为 0 并裁剪 `[-2,2]`。动作网络使用排除索引 `[0,1,2,18..40 每隔3]` 后的 `FLOW_INDICES`。

## 动作表

checkpoint id 从 0 开始，id 9 是历史 padding，不是动作：

| id | 名称 | 仿真语义（源码） | 真机单位/方向 |
|---:|---|---|---|
| 0 | MoveXPos | G1 身体平移正 X | 待真机坐标标定 |
| 1 | MoveXNeg | G1 身体平移负 X | 待真机坐标标定 |
| 2 | MoveYPos | G1 身体平移正 Y | 待真机坐标标定 |
| 3 | MoveYNeg | G1 身体平移负 Y | 待真机坐标标定 |
| 4 | BodyYawLeft | G1 身体左转 | 待真机坐标标定 |
| 5 | BodyYawRight | G1 身体右转 | 待真机坐标标定 |
| 6 | GimbalPitchUp | 云台俯仰上抬 | 待真机零位/方向标定 |
| 7 | GimbalPitchDown | 云台俯仰下压 | 待真机零位/方向标定 |
| 8 | Stop | 策略停车提议 | 只由真实安全层确认 |

仿真配置给出了 5° 云台单步和身体控制时长，但这些数值没有复制到真机配置。`action_space.py` 中的五动作旧词表不是当前 checkpoint 的动作契约。

## 历史、随机性和后处理

动作历史取最近 8 个已执行动作。原 GRPO rollout 在动作执行成功后 `history.append(action)`；超出云台限位或执行异常的分支不追加。部署离线接口只接受调用方提供的历史，不自动假设执行成功。模型 `eval()`、`torch.inference_mode()`，动作选择为 logits `argmax`；没有采样、温度、mask 或随机后处理。低头恢复是独立保护层：只有真实 pitch、时间、相似度、概率、视觉内点和安全状态齐全时才可覆盖；缺少状态返回 `cannot_verify_missing_recovery_state`。

## Stop 区别

策略 `Stop` 是模型输出；保护层覆盖可以把动作改为 `GimbalPitchUp`，不能把 Stop 当作已停车；故障停车和真实执行器急停属于未实现的硬件层。3 秒稳定验证需要真实速度、yaw/pitch 漂移、碰撞和跌倒信号，缺失信号时必须返回无法验证。
