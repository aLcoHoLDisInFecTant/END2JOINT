# End2Joint — 基于神经网络的机器人逆运动学求解

从末端位姿 + 实际关节角映射到 control 信号（action），左右臂联合训练。

## 项目结构

```
End2Joint/
├── actibot_sdk/          # 机器人 SDK（ROS2 控制、URDF 模型、配置文件）
├── data/                 # 数据集
│   ├── 0525_workflow_120/           # 120 episodes 训练数据（LeRobot 格式）
│   ├── 0525_workflow_120_action_fk/ # 训练集 FK 结果（45列，含 ee + state + action）
│   ├── 0602_test_for_net/           # 测试数据（5 episodes）
│   └── 0602_test_for_net_action_fk/ # 测试集 FK 结果
├── example/              # FK 计算工具
│   ├── actibot_fk.py                 # FK 核心引擎（Pinocchio）
│   ├── compute_fk_batch.py           # 基于 observation.state 的 FK
│   ├── compute_fk_action.py          # 基于 action 的 FK（推荐，输出 45 列）
│   └── README.md
├── docs/                 # 文档（报告、实验计划书、环境配置等）
├── ik_net/               # 标准 IK 网络
│   ├── config.py                     # 超参数配置
│   ├── dataloader.py                 # 数据加载（单文件 ee + state + action）
│   ├── model.py                      # ResidualMLP 模型
│   ├── train.py                      # 训练循环（含 run_stage 通用训练原语）
│   ├── train_joint.py                # 真实 + 操作包络合成 联合训练
│   ├── evaluate.py                   # 测试评估
│   ├── predict.py                    # 推理 + 保存结果
│   ├── models/                       # 模型检查点
│   │   ├── 0629_groot85_model/       # 基线（部署）
│   │   ├── 0630_envelope_model/      # 操作包络联合训练候选
│   │   └── archive/                  # 历史检查点（0611/0616/0616_moredata）
│   └── save_real/                    # 真机验证日志
├── ik_net_robust/        # 改进版：噪声注入 + FK 修正
│   ├── config.py                     # 超参数配置
│   ├── dataloader.py                 # 数据加载 + 噪声注入
│   ├── model.py                      # ResidualMLP 模型
│   ├── train.py                      # 训练循环
│   ├── evaluate.py                   # 测试评估
│   ├── fk_utils.py                   # FK 修正函数
│   ├── predict.py                    # 自回归推理 + FK 修正
│   └── test_ar.py                    # 自回归精度测试
```

## 模型

| 模型 | 输入 | 输出 | 核心特点 |
|------|------|------|---------|
| **ik_net** | 26D ee_pose + state_prev | **14D action** | 单源数据，直接预测 |
| **ik_net_robust** | 26D ee_pose + state_prev | 14D action | 噪声注入 + FK 修正 |

## 数据格式

`compute_fk_action.py` 输出（45 列，单文件全部数据）：

| 类别 | 列数 | 列名 | 用途 |
|------|------|------|------|
| 元信息 | 3 | episode_index, frame_index, timestamp | 帧标识 |
| **action 关节** | 14 | L_sh_pitch ~ R_wr_pitch | 训练目标 y |
| **state 关节** | 14 | state_L_sh_pitch ~ state_R_wr_pitch | 输入 prev |
| 夹爪 | 2 | gripper_L, gripper_R | 保存用 |
| **末端位姿** | 12 | eeL_xyzrpy + eeR_xyzrpy | 输入 ee_pose |

训练样本：`X = [ee_pose_t(12), state_{t-1}(14)] → 26D`, `y = action_t → 14D`

## 环境配置

```bash
conda create -n actibot_sdk python=3.12 -y
conda activate actibot_sdk
conda install pinocchio=3.6.0 -c conda-forge -y
pip install torch scikit-learn matplotlib pandas pyarrow
```

## 快速开始

```bash
# 1. 计算 FK
python example/compute_fk_action.py data/0602_test_for_net -o data/0602_test_for_net_action_fk

# 2. 训练
python ik_net/train.py

# 3. 推理
python ik_net/predict.py --data data/0602_test_for_net_action_fk
python ik_net/predict_ar.py --data data/0602_test_for_net_action_fk

# 4. FK 修正
python ik_net_robust/predict.py --fkfix --fkfix-step 5

# 5. 真机验证分析
python ik_net/analyze_real_error.py
```

## 精度

| 模式 | Joint MAE | 单帧耗时 |
|------|-----------|---------|
| ik_net（离线单步） | ~1.0° | 60 μs |
| ik_net（自回归） | ~5.8° | 60 μs |
| ik_net_robust +噪声 | ~2.9° | 60 μs |
| **+FK修正(每5帧)** | **~0.6°** | **92 μs** |
| **+FK修正(每帧)** | **~0.36°** | 170 μs |
| 真机 pred vs real | **~0.6°** | — |
