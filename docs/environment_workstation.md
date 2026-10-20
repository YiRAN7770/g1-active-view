# 当前工作站验证环境

核实时间：2026-10-20（Asia/Shanghai）。

| 项目 | 实际值 |
|---|---|
| Python | 3.11.15（unitree_sim_env） |
| PyTorch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| OpenCV | 4.11.0 |
| NumPy | 1.26.4 |
| Pillow | 11.2.1 |
| CUDA runtime | 13.0 build；`torch.cuda.is_available()` 为 False |
| 验证解释器 | `/home/szu/Software/anaconda3/envs/unitree_sim_env/bin/python` |

另一个训练环境为 Python 3.11.16、PyTorch 2.8.0+cu128、NumPy 2.4.6，但没有 OpenCV；本部署基线使用上表的现有 unitree_sim_env，不修改系统环境。

这些是当前电脑基线，不是 Orin 安装要求。Orin 版本必须在 Stage B 根据 JetPack/L4T 和架构重新查询。
