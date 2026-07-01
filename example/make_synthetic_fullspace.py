#!/usr/bin/env python3
"""生成「全空间合成 FK 数据」用于 IKNET 泛化预训练。

动机（见 IKNET_TEST_REPORT_zh.md §7.4.7）：IKNET 只在任务录制数据上训练，
关节空间大片区域属 OOD，纯网络在未见大幅度动作上末端误差大。用关节限位内
随机采样的全空间构型 + FK 末端位姿，给网络一个覆盖全工作空间的粗解基座。

关键设计——prev_joints 的 δ 分布：
  dataloader 把 `state` 列右移一帧作为 prev（state_prev[t] = state[t-1]）。
  本脚本令 state == action == q_t，并以「随机游走」构造每条 episode，
  因此 prev_t = q_{t-1}，单步游走步长即 prev→target 的 δ。
  真机单步 δ 仅 ~0.003–0.005 rad，故步长按混合分布抽样（主体真机小尺度，
  叠加中等尾巴覆盖 open-loop staleness / OOD 跳变），既保留全空间 endpose
  覆盖，又让 prev 分布贴近部署。

输出 45 列 parquet，schema 与 example/compute_fk_action.py 完全一致，
文件名 episode_{i:06d}_action_fk.parquet，可直接被 ik_net/dataloader.py 读取。

用法:
  conda activate actibot_sdk   # 或 env -u PYTHONPATH .venv/bin/python ...
  python example/make_synthetic_fullspace.py \
      --num-episodes 4000 --frames 20 \
      --out-dir data/synthetic_fullspace_fk --seed 42
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_example_dir = os.path.abspath(os.path.dirname(__file__))
if _example_dir not in sys.path:
    sys.path.insert(0, _example_dir)
_project_dir = os.path.abspath(os.path.join(_example_dir, ".."))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from actibot_fk import Arm_IK
from pinocchio.rpy import matrixToRpy

# ── 列定义：与 compute_fk_action.py 保持一致 ──
COL_JOINTS_L = [f"L_{n}" for n in ("sh_pitch", "sh_roll", "sh_yaw",
                                   "el_pitch", "el_roll", "wr_yaw", "wr_pitch")]
COL_JOINTS_R = [f"R_{n}" for n in ("sh_pitch", "sh_roll", "sh_yaw",
                                   "el_pitch", "el_roll", "wr_yaw", "wr_pitch")]
COL_STATE_L = [f"state_{n}" for n in ("L_sh_pitch", "L_sh_roll", "L_sh_yaw",
                "L_el_pitch", "L_el_roll", "L_wr_yaw", "L_wr_pitch")]
COL_STATE_R = [f"state_{n}" for n in ("R_sh_pitch", "R_sh_roll", "R_sh_yaw",
                "R_el_pitch", "R_el_roll", "R_wr_yaw", "R_wr_pitch")]
COL_GRIPPER = ["gripper_L", "gripper_R"]
COL_POS_L = ["eeL_x", "eeL_y", "eeL_z"]
COL_RPY_L = ["eeL_roll", "eeL_pitch", "eeL_yaw"]
COL_POS_R = ["eeR_x", "eeR_y", "eeR_z"]
COL_RPY_R = ["eeR_roll", "eeR_pitch", "eeR_yaw"]
COL_META = ["episode_index", "frame_index", "timestamp"]

ALL_COLS = COL_META + COL_JOINTS_L + COL_JOINTS_R + COL_STATE_L + COL_STATE_R \
           + COL_GRIPPER + COL_POS_L + COL_RPY_L + COL_POS_R + COL_RPY_R

URDF_PATH = os.path.join(_project_dir,
    "actibot_sdk/robot_description/v3/urdf/v3_urdf_251121-2.urdf")

# 14D 臂关节在 19D 简化模型中的索引：左 5-11，右 12-18
ARM_IDX = list(range(5, 12)) + list(range(12, 19))

# 混合 δ 分布（逐关节、逐步抽样的高斯 σ，单位 rad）：
#   70% ~ N(0, 0.005)  ~0.3°  匹配真机单步（主体 in-distribution）
#   25% ~ N(0, 0.03)   ~1.7°  数步 staleness
#    5% ~ N(0, 0.10)   ~5.7°  open-loop horizon=8 累积 / 快速动作 / OOD 跳变
DEFAULT_MIX_WEIGHTS = [0.70, 0.25, 0.05]
DEFAULT_MIX_SIGMAS = [0.005, 0.03, 0.10]


def get_joint_limits(ik):
    """从 Pinocchio 简化模型按 14 臂关节顺序读取限位（权威，随 URDF 同步）。"""
    m = ik.reduced_robot.model
    lo = np.asarray(m.lowerPositionLimit)[ARM_IDX].astype(np.float64)
    hi = np.asarray(m.upperPositionLimit)[ARM_IDX].astype(np.float64)
    # 对非有限限位回退到 ±π
    lo = np.where(np.isfinite(lo), lo, -np.pi)
    hi = np.where(np.isfinite(hi), hi, np.pi)
    return lo, hi


def sample_steps(rng, shape, weights, sigmas):
    """逐元素从高斯混合抽样游走步长。shape=(n_steps, 14)。"""
    comp = rng.choice(len(weights), size=shape, p=weights)
    sig = np.asarray(sigmas)[comp]
    return rng.normal(0.0, 1.0, size=shape) * sig


def make_episode_q(rng, q0, n_frames, lo, hi, weights, sigmas):
    """构造一条 episode 的关节轨迹 (n_frames, 14)，起点 q0 由调用方给定。

    q_t = clip(q_{t-1} + step_t, lo, hi)，step 为混合 δ（即 prev→target 的 δ）。
    """
    q = np.empty((n_frames, 14), dtype=np.float64)
    q[0] = np.clip(q0, lo, hi)
    if n_frames > 1:
        steps = sample_steps(rng, (n_frames - 1, 14), weights, sigmas)
        for t in range(1, n_frames):
            q[t] = np.clip(q[t - 1] + steps[t - 1], lo, hi)
    return q


def load_real_configs(real_dirs):
    """读取真实 action_fk 的 14D action 关节，返回 (M,14)、均值 μ、逐关节 min/max。"""
    cols = COL_JOINTS_L + COL_JOINTS_R
    import glob
    frames = []
    for d in real_dirs:
        for f in sorted(glob.glob(os.path.join(d, "episode_*_fk.parquet"))):
            frames.append(pd.read_parquet(f)[cols].values.astype(np.float64))
    if not frames:
        raise FileNotFoundError(f"未在 {real_dirs} 找到 episode_*_fk.parquet")
    R = np.vstack(frames)
    return R, R.mean(axis=0), R.min(axis=0), R.max(axis=0)


def envelope_start(rng, R, mu, rmin, rmax, lo, hi, p):
    """操作包络起点：径向扩展 + 对随机少数关节的定向远扩展。

    1) 径向：q0 = μ + α ⊙ (q_real − μ)（保留关节相关性，覆盖流形近邻外缘）；
       α 逐关节 = 1 + (alpha_max−1)·U^skew（skew>1 偏向 1，多数关节接近真实）。
    2) 定向远扩展（解决「真实范围窄但需远探」的关节，如 R_sh_yaw 达 11.7σ）：
       以 prob p_ext 随机选 k 个关节，把其值改采于「扩展操作范围」
       [μ − E·(μ−rmin), μ + E·(rmax−μ)]（截断到 URDF 限位）。只扩少数关节 →
       仍贴近「一臂/少数关节越界」的真实结构，不退化为均匀立方体。
    3) ε 小噪声填密度。
    """
    q_real = R[rng.integers(len(R))]
    u = rng.random(14) ** p["alpha_skew"]
    alpha = 1.0 + (p["alpha_max"] - 1.0) * u
    q0 = mu + alpha * (q_real - mu)

    if rng.random() < p["p_ext"]:
        k = rng.integers(1, p["max_ext_joints"] + 1)
        E = p["ext_factor"]
        elo = np.clip(mu - E * (mu - rmin), lo, hi)
        ehi = np.clip(mu + E * (rmax - mu), lo, hi)
        for j in rng.choice(14, size=k, replace=False):
            q0[j] = rng.uniform(elo[j], ehi[j])

    q0 = q0 + rng.normal(0.0, p["jitter"], size=14)
    return np.clip(q0, lo, hi)


def fk_batch(ik, q14):
    """对 (N,14) 关节批量 FK，返回 (N,12) = [eeL_xyz, eeL_rpy, eeR_xyz, eeR_rpy]。"""
    N = q14.shape[0]
    ee = np.empty((N, 12), dtype=np.float64)
    q19 = np.zeros(19)
    for i in range(N):
        q19[5:12] = q14[i, :7]
        q19[12:19] = q14[i, 7:]
        T_l, T_r = ik.get_fk_solution(q19)
        ee[i, 0:3] = T_l[:3, 3]
        ee[i, 3:6] = matrixToRpy(T_l[:3, :3])
        ee[i, 6:9] = T_r[:3, 3]
        ee[i, 9:12] = matrixToRpy(T_r[:3, :3])
    return ee


def build_episode_df(ep_idx, q14, ee, dt):
    """组装 45 列 DataFrame（action == state == q；gripper=0）。"""
    N = q14.shape[0]
    data = {
        "episode_index": np.full(N, ep_idx, dtype=np.int64),
        "frame_index": np.arange(N, dtype=np.int64),
        "timestamp": (np.arange(N) * dt).astype(np.float64),
    }
    for j, col in enumerate(COL_JOINTS_L):
        data[col] = q14[:, j]
    for j, col in enumerate(COL_JOINTS_R):
        data[col] = q14[:, 7 + j]
    for j, col in enumerate(COL_STATE_L):           # state == action（完美跟踪）
        data[col] = q14[:, j]
    for j, col in enumerate(COL_STATE_R):
        data[col] = q14[:, 7 + j]
    data["gripper_L"] = np.zeros(N, dtype=np.float64)
    data["gripper_R"] = np.zeros(N, dtype=np.float64)
    for j, col in enumerate(COL_POS_L):
        data[col] = ee[:, 0 + j]
    for j, col in enumerate(COL_RPY_L):
        data[col] = ee[:, 3 + j]
    for j, col in enumerate(COL_POS_R):
        data[col] = ee[:, 6 + j]
    for j, col in enumerate(COL_RPY_R):
        data[col] = ee[:, 9 + j]
    return pd.DataFrame(data, columns=ALL_COLS)


def main():
    ap = argparse.ArgumentParser(description="生成全空间合成 FK 数据（混合 δ 随机游走）")
    ap.add_argument("--num-episodes", type=int, default=4000)
    ap.add_argument("--frames", type=int, default=20, help="每条 episode 帧数")
    ap.add_argument("--out-dir", default=os.path.join(_project_dir, "data/synthetic_fullspace_fk"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dt", type=float, default=1.0 / 30.0)
    ap.add_argument("--urdf", default=URDF_PATH)
    ap.add_argument("--mix-weights", type=float, nargs=3, default=DEFAULT_MIX_WEIGHTS,
                    help="混合 δ 三分量权重（小/中/大）")
    ap.add_argument("--mix-sigmas", type=float, nargs=3, default=DEFAULT_MIX_SIGMAS,
                    help="混合 δ 三分量 σ（rad）")
    ap.add_argument("--mode", choices=["uniform", "envelope"], default="envelope",
                    help="起点分布：uniform=全空间均匀（已证实对小 MLP 适得其反）；"
                         "envelope=操作包络径向扩展（推荐，覆盖 ep001 类外缘）")
    ap.add_argument("--real-dirs", nargs="+", default=[
                        os.path.join(_project_dir, "data/0525_workflow_120_action_fk"),
                        os.path.join(_project_dir, "data/my_dataset_groot_action_fk")],
                    help="envelope 模式：真实流形来源目录")
    # 默认为经验证的最佳配置（in-dist 6.96mm、ep000 8.8mm、ep001 median 19mm）
    ap.add_argument("--alpha-max", type=float, default=3.5, help="envelope 径向扩展上限")
    ap.add_argument("--alpha-skew", type=float, default=1.5, help="α 偏置指数（>1 偏向 1）")
    ap.add_argument("--jitter", type=float, default=0.03, help="envelope 起点噪声 σ(rad)")
    ap.add_argument("--p-ext", type=float, default=0.35, help="对少数关节做定向远扩展的概率")
    ap.add_argument("--max-ext-joints", type=int, default=2, help="单样本最多远扩展的关节数")
    ap.add_argument("--ext-factor", type=float, default=3.0, help="远扩展的操作范围倍数 E")
    args = ap.parse_args()

    weights = np.asarray(args.mix_weights, dtype=np.float64)
    weights = weights / weights.sum()
    sigmas = args.mix_sigmas

    print("加载 FK 引擎 ...")
    ik = Arm_IK(args.urdf)
    lo, hi = get_joint_limits(ik)
    print(f"混合 δ: weights={list(np.round(weights, 3))} sigmas={sigmas}")

    rng = np.random.default_rng(args.seed)

    # ── 起点分布 ──
    if args.mode == "envelope":
        R, mu, rmin, rmax = load_real_configs(args.real_dirs)
        p = {"alpha_max": args.alpha_max, "alpha_skew": args.alpha_skew, "jitter": args.jitter,
             "p_ext": args.p_ext, "max_ext_joints": args.max_ext_joints, "ext_factor": args.ext_factor}
        print(f"envelope 模式: 真实帧 {len(R)}, 径向 alpha_max={args.alpha_max}/skew={args.alpha_skew}, "
              f"远扩展 p={args.p_ext}/k≤{args.max_ext_joints}/E={args.ext_factor}, jitter={args.jitter}")
        def start_fn():
            return envelope_start(rng, R, mu, rmin, rmax, lo, hi, p)
    else:
        print("uniform 模式: 全空间均匀起点")
        def start_fn():
            return rng.uniform(lo, hi)

    os.makedirs(args.out_dir, exist_ok=True)
    total = args.num_episodes * args.frames
    print(f"生成 {args.num_episodes} episodes × {args.frames} 帧 = {total} 样本 → {args.out_dir}")

    for ep in range(args.num_episodes):
        q14 = make_episode_q(rng, start_fn(), args.frames, lo, hi, weights, sigmas)
        ee = fk_batch(ik, q14)
        df = build_episode_df(ep, q14, ee, args.dt)
        out_path = os.path.join(args.out_dir, f"episode_{ep:06d}_action_fk.parquet")
        df.to_parquet(out_path, index=False)
        if (ep + 1) % 200 == 0 or ep == args.num_episodes - 1:
            print(f"  {ep + 1}/{args.num_episodes} ...")

    print(f"完成！{args.num_episodes} 个 episode 保存在 {args.out_dir}")


if __name__ == "__main__":
    main()
