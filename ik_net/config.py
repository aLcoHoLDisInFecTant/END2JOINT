"""IK-Net 超参数配置

使用方法:
  from config import hp, paths, data_config
"""

import os

# 项目根目录：自动取本文件所在目录(ik_net)的上一级，免去硬编码绝对路径
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════
#  超参数 (HYPERPARAMETERS) — 与模型训练相关，常修改
# ═══════════════════════════════════════════════════════════════

hp = {

    # ── 数据 ──
    "input_dim": 26,          # 输入: [eeL_xyzrpy(6), eeR_xyzrpy(6), prev_joints_L(7), prev_joints_R(7)]
    "output_dim": 14,         # 输出: [joints_L(7), joints_R(7)] — control 信号 (action)

    # ── 模型架构 ──
    "hidden_dims": [400, 300, 200, 100, 50],   # 各隐藏层神经元数
    "dropout": 0.1,                             # Dropout 比率
    "use_residual": False,                 # True = 残差连接, False = 直接预测

    # ── 训练 ──
    "batch_size": 256,          # 批大小
    "learning_rate": 1e-3,      # 初始学习率
    "weight_decay": 1e-5,       # L2 正则化系数
    "num_epochs": 10000,          # 最大训练轮数
    "lr_step": 50,              # 学习率衰减间隔 (epoch)
    "lr_gamma": 0.5,            # 学习率衰减系数
    "patience": 100,             # 早停容忍轮数
    "target_joint_deg": 0.1,    # 目标关节角平均误差（度），达标即停
    "ckpt_dir": os.path.join(_PROJECT_DIR, "ik_net/0629_groot85_model"),  # 模型权重 + 归一化参数路径
    "ckpt_name": "0629_model.pt",

    # ── 数据划分 ──
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    # ── 随机种子 ──
    "seed": 42,
}


# ═══════════════════════════════════════════════════════════════
#  路径配置 (PATHS) — 与环境相关，很少修改
# ═══════════════════════════════════════════════════════════════

paths = {
    "data_dir": os.path.join(_PROJECT_DIR, "data/0525_workflow_120_action_fk"),
    "extra_data_dirs": [
        os.path.join(_PROJECT_DIR, "data/my_dataset_groot_action_fk")
        ],
    # 均匀全空间合成（已证实对小 MLP 适得其反，保留作对照）
    "synthetic_data_dir": os.path.join(_PROJECT_DIR, "data/synthetic_fullspace_fk"),
    # 操作包络合成（径向扩展真实流形外缘，覆盖 ep001 类 OOD；供 train_joint.py）
    "envelope_data_dir": os.path.join(_PROJECT_DIR, "data/synthetic_envelope_fk"),
    "results_dir": os.path.join(_PROJECT_DIR, "ik_net/0629_groot85_model"),
    "staged_results_dir": os.path.join(_PROJECT_DIR, "ik_net/0630_synth_pretrain_model"),
    "joint_results_dir": os.path.join(_PROJECT_DIR, "ik_net/0630_envelope_model"),
    "project_dir": _PROJECT_DIR,
    "urdf_path": os.path.join(_PROJECT_DIR, "actibot_sdk/robot_description/v3/urdf/v3_urdf_251121-2.urdf"),
}


# ═══════════════════════════════════════════════════════════════
#  两阶段训练超参 (STAGED: 合成预训练 → 真实微调)
# ═══════════════════════════════════════════════════════════════

staged = {
    # scaler 在「真实 train」上拟合（与部署/基线一致，保留任务区分辨率）；
    # 合成预训练复用同一冻结 scaler（其全空间输入会落在较大归一化幅度，可学）。
    "pretrain_epochs": 200,        # Stage A: 合成全空间预训练
    "pretrain_lr": 1e-3,
    "pretrain_lr_step": 50,
    "pretrain_lr_gamma": 0.5,
    "pretrain_patience": 30,       # 以合成 val 关节 MAE 早停

    "finetune_epochs": 4000,       # Stage B: 真实任务微调
    "finetune_lr": 5e-4,           # 足以恢复 in-distribution 精度，又不至于抹掉基座
    "finetune_lr_step": 150,       # 比单阶段更缓的衰减，避免 lr 过早归零（上轮 1.18° 卡死的主因）
    "finetune_lr_gamma": 0.5,
    "finetune_patience": 200,      # 以真实 val 关节 MAE 早停
    "ckpt_name": "0630_model.pt",
}


# ═══════════════════════════════════════════════════════════════
#  数据字段名 (DATA FIELDS) — 与数据列名相关
# ═══════════════════════════════════════════════════════════════

data_config = {
    "col_eeL": ["eeL_x", "eeL_y", "eeL_z", "eeL_roll", "eeL_pitch", "eeL_yaw"],
    "col_eeR": ["eeR_x", "eeR_y", "eeR_z", "eeR_roll", "eeR_pitch", "eeR_yaw"],
    "col_action_l": ["L_sh_pitch", "L_sh_roll", "L_sh_yaw",
                     "L_el_pitch", "L_el_roll", "L_wr_yaw", "L_wr_pitch"],
    "col_action_r": ["R_sh_pitch", "R_sh_roll", "R_sh_yaw",
                     "R_el_pitch", "R_el_roll", "R_wr_yaw", "R_wr_pitch"],
    "col_state_l": ["state_L_sh_pitch", "state_L_sh_roll", "state_L_sh_yaw",
                    "state_L_el_pitch", "state_L_el_roll", "state_L_wr_yaw", "state_L_wr_pitch"],
    "col_state_r": ["state_R_sh_pitch", "state_R_sh_roll", "state_R_sh_yaw",
                    "state_R_el_pitch", "state_R_el_roll", "state_R_wr_yaw", "state_R_wr_pitch"],
}
