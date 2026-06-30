"""IK-Net-Robust 超参数配置

使用方法:
  from config import hp, paths, data_config
"""

import os

# 项目根目录：自动取本文件所在目录(ik_net_robust)的上一级，免去硬编码绝对路径
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
#  超参数 (HYPERPARAMETERS) — 与模型训练相关，常修改
# ═══════════════════════════════════════════════════════════════

hp = {

    # ── 数据 ──
    "input_dim": 26,          # 输入: [eeL_xyzrpy(6), eeR_xyzrpy(6), state_prev_L(7), state_prev_R(7)]
    "output_dim": 14,         # 输出: control 信号 (action)

    # ── 模型架构 ──
    "hidden_dims": [400, 300, 200, 100, 50],
    "dropout": 0.1,
    "use_residual": False,

    # ── 噪声注入（抗自回归误差累积）──
    "noise_prob": 0.5,
    "noise_scale": 0.05,

    # ── 训练 ──
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "num_epochs": 10000,
    "lr_step": 50,
    "lr_gamma": 0.5,
    "patience": 100,
    "target_joint_deg": 0.1,
    "ckpt_dir": os.path.join(_PROJECT_DIR, "ik_net_robust/results"),
    "ckpt_name": "best_model.pt",

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
    "extra_data_dirs": [],
    "results_dir": os.path.join(_PROJECT_DIR, "ik_net_robust/results"),
    "project_dir": _PROJECT_DIR,
    "urdf_path": os.path.join(_PROJECT_DIR, "actibot_sdk/robot_description/v3/urdf/v3_urdf_251121-2.urdf"),
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
