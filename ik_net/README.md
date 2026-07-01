# IK-Net: 基于神经网络的逆运动学求解

用 MLP 从末端位姿 + 上一帧实际关节角，直接预测 **control 信号（action）**。

## 输入输出

**输入** (26 维)：
```
[eeL_xyzrpy(6), eeR_xyzrpy(6), state_L_prev(7), state_R_prev(7)]
```

**输出** (14 维)：
```
[action_L(7), action_R(7)]
```

### 数据来源

| 输入部分 | 来源 | 说明 |
|---------|------|------|
| `ee_pose_t` | `action_fk` | 从 action 控制信号算的 FK（期望末端位姿） |
| `state_{t-1}` | `state_fk` | 从 observation.state 取的实际关节角 |
| `y = action_t` | `action_fk` | 目标值 = 应发送的 control 信号 |

`state_{t-1}` 输入帮助模型感知机器人当前实际位置，避免控制信号与真实状态脱节。

## 网络架构

```
输入: 26D
  │
  ├── Linear(26→400) + ReLU + Dropout(0.1)
  ├── Linear(400→300) + ReLU + Dropout(0.1)
  ├── Linear(300→200) + ReLU + Dropout(0.1)
  ├── Linear(200→100) + ReLU + Dropout(0.1)
  ├── Linear(100→50) + ReLU + Dropout(0.1)
  │
  └── Linear(50→14) → 输出 14D (action_t)
```

- 总参数量：~22 万
- 直接预测模式：`action_t = MLP(ee_pose_t, state_{t-1})`

## 项目结构

```
ik_net/
├── config.py              # 超参数配置
├── dataloader.py          # 数据加载（双源：action FK + state FK）
├── model.py               # ResidualMLP 模型
├── train.py               # 训练循环（含 run_stage 通用训练原语）
├── train_joint.py         # 真实 + 操作包络合成 联合训练
├── evaluate.py            # 测试评估
├── predict.py             # 推理测试 + 保存预测结果
├── fk_utils.py            # Pinocchio FK 封装
├── models/                # 模型检查点
│   ├── 0629_groot85_model/    # 基线（部署）
│   ├── 0630_envelope_model/   # 操作包络联合训练候选
│   └── archive/               # 历史检查点（0611/0616/0616_moredata）
├── save_real/             # 真机验证日志
├── requirements.txt
└── README.md
```

## 安装与运行

```bash
conda activate actibot_sdk
pip install torch scikit-learn matplotlib pandas pyarrow
```

### 训练

```bash
python ik_net/train.py
```

### 推理

```bash
# 单步预测
python ik_net/predict.py --data data/0602_test_for_net_action_fk

# 保存预测结果
python ik_net/predict.py --data /path/to/data --save ./results

# 自回归推理
python ik_net/predict_ar.py --data /path/to/data
```

### 回放

```bash
python ik_net/playback_predictions.py ./results/predictions.parquet
```

## 关键配置

| 参数 | 作用 |
|------|------|
| `hp["ckpt_dir"]` | 模型权重路径 |
| `hp["ckpt_name"]` | 模型文件名 |
| `paths["data_dir"]` | action FK 训练数据 |
| `paths["joint_state_dir"]` | state FK 数据（提供 prev_joints） |
| `paths["extra_data_dirs"]` | 额外训练数据 |
