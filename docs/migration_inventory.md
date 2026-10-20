# Stage A 迁移清单

## 已迁移

- `monotone_stop_policy.py` 的 `VisualStopResidualPolicy`、Stop residual 和推理输出。
- `magnitude_visual_policy.py` 的 MagnitudePolicy 结构和前 8 动作输出。
- `relative_visual_policy.py` 的 SIFT geometry 特征和 `FLOW_INDICES`。
- `dedicated_policy.py` 的 DINOv2 编码、RGB 预处理和历史编码。
- `low_head_recovery.py` 的纯状态保护逻辑。
- 577 路径 checkpoint、DINOv2 源码和 DINOv2 权重。
- 一组真实仿真 RGB 输入、目标图像和原实现输出基准。

## 排除

Isaac Lab、Isaac Sim、USD、仿真环境、仿真机器人控制、训练脚本、GRPO optimizer、专家采集器、训练数据集、DDS/ROS/Unitree SDK、相机打开逻辑、串口和运动命令都没有成为部署包运行依赖。

## 真机阶段 B 必须新增

1. 真实相机采集适配器：RGB 格式、分辨率、时间戳、曝光、内外参和延迟。
2. 真实云台读写适配器：pan/tilt 编码器、零位、方向、限位、速度闭环、watchdog 和急停。
3. 若保留九动作，新增 G1 身体运动适配器：平移、yaw、IMU、接触、碰撞和跌倒反馈。
4. 真实 Stop 稳定信号适配器：线速度、yaw 速率、XY/yaw/pitch 漂移、碰撞/跌倒。
5. 真机标定和回放数据；仿真 5°、0.4 m/s 等参数不得直接使用。
6. Orin 设备、JetPack/L4T、CUDA、Python、PyTorch/OpenCV 版本查询和内存/延迟基线。

Stage A 的 `adapters/interfaces.py` 明确抛出未实现错误，没有假装提供硬件后端。
