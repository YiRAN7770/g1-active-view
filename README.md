# G1 Active View：Stage A 部署仓库

这个仓库是当前 G1+双轴云台视觉动作策略的离线推理包。它只读取图像、目标图像和动作历史，输出九动作策略结果；不会导入 Isaac、打开相机、打开串口、初始化机器人 SDK 或自动执行动作。Stage A 不包含真实硬件驱动，也不声称已经适配 Orin 或真机。

## 当前模型

`models/va_sft_577_best.pt` 是原项目封装的 577 条专家轨迹 VA-SFT checkpoint，变体为 `visual_stop_residual`。DINOv2 ViT-B/14 源码已放在 `third_party/dinov2`，权重为 `models/dinov2_vitb14_pretrain.pth`。两个文件都由 `models/manifest.json` 校验；Git LFS 在首次提交前已配置。

## 安装和当前电脑验证

标准安装方式（在准备好的独立环境中执行）：

```bash
python -m pip install -e .
```

以下命令从仓库外也可以执行：

```bash
cd /tmp
/home/szu/Software/anaconda3/envs/unitree_sim_env/bin/python /home/szu/Project/active_view_g1/g1-active-view/scripts/check_environment.py
/home/szu/Software/anaconda3/envs/unitree_sim_env/bin/python /home/szu/Project/active_view_g1/g1-active-view/scripts/prepare_models.py
PYTHONPATH=/home/szu/Project/active_view_g1/g1-active-view/src \
  /home/szu/Software/anaconda3/envs/unitree_sim_env/bin/python \
  /home/szu/Project/active_view_g1/g1-active-view/scripts/infer_offline.py \
  --current /home/szu/Project/active_view_g1/g1-active-view/datasets/fixtures/current_082_xpos.png \
  --target /home/szu/Project/active_view_g1/g1-active-view/datasets/fixtures/target_082.png \
  --history '[]'
PYTHONPATH=/home/szu/Project/active_view_g1/g1-active-view/src \
  /home/szu/Software/anaconda3/envs/unitree_sim_env/bin/python \
  /home/szu/Project/active_view_g1/g1-active-view/scripts/replay_baseline.py
PYTHONPATH=/home/szu/Project/active_view_g1/g1-active-view/src \
  /home/szu/Software/anaconda3/envs/unitree_sim_env/bin/python -m pytest \
  /home/szu/Project/active_view_g1/g1-active-view/tests
```

推理默认使用仓库内模型路径，可用环境变量覆盖：`G1_ACTIVE_VIEW_CHECKPOINT`、`G1_ACTIVE_VIEW_DINO_SOURCE`、`G1_ACTIVE_VIEW_DINO_WEIGHTS`。所有路径缺失或校验不一致时会失败，不会下载替代文件。

## 输出契约

`infer_offline.py` 输出原始 logits、概率、原始动作、保护层动作、相似度、SIFT 内点数、覆盖审计、有效性和推理耗时。动作历史最多 8 个，编码为 checkpoint 内部动作表，episode 开始用 id 9 填充。Stage A 的执行器接口只抛出“未实现”，不会发送动作。

详细契约见 `docs/model_contract.md`，迁移清单见 `docs/migration_inventory.md`。
