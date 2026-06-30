#!/usr/bin/env python3
"""阶段 0 / Gate-0 对齐校验：torch 可微精修器 vs 生产 Pinocchio fk_correction。

验收（实验计划书 §3.2）：
  A. 同一批关节，torch FK vs Pinocchio FK 末端位姿差 ~µm；
  B. 同种子同 K 下两精修器输出关节差可忽略；
  C. torch 精修器可微（梯度有限、可回传）。

用法:
  conda activate actibot_sdk
  python ik_net_robust/check_alignment.py
  python ik_net_robust/check_alignment.py --n 1024 --data data/0602_test_for_net_action_fk
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

_this_dir = os.path.abspath(os.path.dirname(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, ".."))
_example_dir = os.path.join(_project_root, "example")
for p in [_this_dir, _project_root, _example_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import paths, data_config
from fk_utils import load_ik, compute_ee_pose, fk_correction
from diff_refiner import DiffRefiner, log3

# 通过阈值（Gate-0）
THRESH_FK_POS = 1e-5      # m，FK 位置差（~µm 级）
THRESH_FK_ORI = 1e-5      # rad，FK 姿态差
THRESH_REFINE = 1e-4      # rad，同种子同 K 精修关节差


def sample_joints(refiner, n, rng, dataset_q=None):
    """在关节限位内采样 n 组关节；若提供数据集关节则混入。"""
    lo = refiner.q_lower.cpu().numpy()
    hi = refiner.q_upper.cpu().numpy()
    # 限位可能含 inf，回退到 ±π
    lo = np.where(np.isfinite(lo), lo, -np.pi)
    hi = np.where(np.isfinite(hi), hi, np.pi)
    q = rng.uniform(lo, hi, size=(n, 14))
    if dataset_q is not None and len(dataset_q) > 0:
        k = min(len(dataset_q), n // 4)
        idx = rng.choice(len(dataset_q), size=k, replace=False)
        q[:k] = dataset_q[idx]
    return q


def load_dataset_joints(data_dir):
    """从 parquet 读取真值关节（action），用于真实分布采样。"""
    if not data_dir or not os.path.isdir(data_dir):
        return None
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*_fk.parquet")))
    if not files:
        return None
    import pandas as pd
    cols = data_config["col_action_l"] + data_config["col_action_r"]
    dfs = [pd.read_parquet(f)[cols].values.astype(np.float64) for f in files[:5]]
    return np.vstack(dfs)


def rot_angle_diff(R_a, R_b):
    """两组旋转矩阵 (N,3,3) 的角度差 (N,)，单位 rad。"""
    rel = R_a @ R_b.transpose(0, 2, 1)
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cos)


# ════════════════════════════════════════════════════════════════════
#  Test A — FK 一致性
# ════════════════════════════════════════════════════════════════════

def test_fk(ik, refiner, q_batch):
    N = q_batch.shape[0]
    # Pinocchio（逐样本）
    pin_l = np.zeros((N, 4, 4))
    pin_r = np.zeros((N, 4, 4))
    for i in range(N):
        q19 = np.zeros(19)
        q19[5:12] = q_batch[i, :7]
        q19[12:19] = q_batch[i, 7:]
        Tl, Tr = ik.get_fk_solution(q19)
        pin_l[i], pin_r[i] = Tl, Tr
    # torch（批量）
    qt = torch.tensor(q_batch, dtype=refiner.dtype)
    with torch.no_grad():
        Tl_t, Tr_t = refiner.fk_torch(qt)
    Tl_t = Tl_t.cpu().numpy()
    Tr_t = Tr_t.cpu().numpy()

    pos_diff = np.concatenate([
        np.linalg.norm(Tl_t[:, :3, 3] - pin_l[:, :3, 3], axis=1),
        np.linalg.norm(Tr_t[:, :3, 3] - pin_r[:, :3, 3], axis=1),
    ])
    ori_diff = np.concatenate([
        rot_angle_diff(Tl_t[:, :3, :3], pin_l[:, :3, :3]),
        rot_angle_diff(Tr_t[:, :3, :3], pin_r[:, :3, :3]),
    ])
    return pos_diff, ori_diff


# ════════════════════════════════════════════════════════════════════
#  Test B — 精修器一致性（同种子同 K）
# ════════════════════════════════════════════════════════════════════

def test_refine(ik, refiner, q_true, rng, K_list, perturb=0.1):
    N = q_true.shape[0]
    # 目标末端来自真值关节 FK；种子 = 真值 + 扰动（落在精修吸引域内）
    ee_target = np.stack([compute_ee_pose(ik, q_true[i]) for i in range(N)])
    seed = q_true + rng.uniform(-perturb, perturb, size=q_true.shape)

    results = {}
    ee_t = torch.tensor(ee_target, dtype=refiner.dtype)
    seed_t = torch.tensor(seed, dtype=refiner.dtype)
    for K in K_list:
        # torch（批量，固定 K 步）
        with torch.no_grad():
            q_torch = refiner.refine(seed_t, ee_t, K=K, lam=0.1, dq_clip=0.05).cpu().numpy()
        # Pinocchio（逐样本，tol=-1 强制 K 步）
        q_pin = np.zeros_like(q_true)
        for i in range(N):
            qL, qR, _, _ = fk_correction(ik, seed[i].copy(), ee_target[i],
                                         damping=0.1, max_iter=K, tol=-1.0)
            q_pin[i, :7] = qL
            q_pin[i, 7:] = qR
        results[K] = np.abs(q_torch - q_pin)
    return results


# ════════════════════════════════════════════════════════════════════
#  Test C — 可微性
# ════════════════════════════════════════════════════════════════════

def test_grad(ik, refiner, q_true, rng, K=2):
    n = min(16, q_true.shape[0])
    ee_target = np.stack([compute_ee_pose(ik, q_true[i]) for i in range(n)])
    seed = q_true[:n] + rng.uniform(-0.1, 0.1, size=(n, 14))
    seed_t = torch.tensor(seed, dtype=refiner.dtype, requires_grad=True)
    ee_t = torch.tensor(ee_target, dtype=refiner.dtype)
    q_out = refiner.refine(seed_t, ee_t, K=K)
    loss = (q_out ** 2).sum()
    loss.backward()
    g = seed_t.grad
    return g is not None and torch.isfinite(g).all().item(), (None if g is None else float(g.abs().mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512, help="采样关节数")
    parser.add_argument("--data", default=None, help="混入真实分布关节的数据集目录")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print("加载 Pinocchio 模型与 torch 精修器 ...")
    ik = load_ik()
    refiner = DiffRefiner.from_pinocchio(ik, dtype=torch.float64)
    # 缓存常数，训练时可用 DiffRefiner.from_npz 无 Pinocchio 加载
    npz_path = os.path.join(_this_dir, "refiner_consts.npz")
    refiner.save_npz(npz_path)
    print(f"  常数缓存已保存: {npz_path}")

    data_dir = args.data or paths.get("data_dir")
    dataset_q = load_dataset_joints(data_dir)
    if dataset_q is not None:
        print(f"  混入数据集真值关节: {len(dataset_q)} 帧 ({data_dir})")

    q_batch = sample_joints(refiner, args.n, rng, dataset_q)

    # ── Test A ──
    print(f"\n{'='*60}\nTest A — FK 一致性 (torch vs Pinocchio), N={args.n}\n{'='*60}")
    pos_diff, ori_diff = test_fk(ik, refiner, q_batch)
    print(f"  位置差:  mean {pos_diff.mean()*1e6:8.4f} µm   max {pos_diff.max()*1e6:8.4f} µm")
    print(f"  姿态差:  mean {ori_diff.mean()*1e6:8.4f} µrad max {ori_diff.max()*1e6:8.4f} µrad")
    passA = pos_diff.max() < THRESH_FK_POS and ori_diff.max() < THRESH_FK_ORI

    # ── Test B ──
    K_list = [1, 2, 3, 5]
    n_ref = min(args.n, 256)
    print(f"\n{'='*60}\nTest B — 精修器一致性 (同种子同 K), N={n_ref}\n{'='*60}")
    refine_res = test_refine(ik, refiner, q_batch[:n_ref], rng, K_list)
    passB = True
    print(f"  {'K':>3} | {'关节差 mean (rad)':>18} | {'关节差 max (rad)':>18}")
    print("  " + "-" * 46)
    for K in K_list:
        diff = refine_res[K]
        mx = diff.max()
        print(f"  {K:>3} | {diff.mean():>18.3e} | {mx:>18.3e}")
        passB = passB and mx < THRESH_REFINE

    # ── Test C ──
    print(f"\n{'='*60}\nTest C — 可微性\n{'='*60}")
    ok_grad, gmean = test_grad(ik, refiner, q_batch, rng)
    print(f"  种子梯度有限: {ok_grad}   |grad| mean: {gmean:.3e}")

    # ── Gate-0 汇总 ──
    print(f"\n{'='*60}\nGate-0 判定\n{'='*60}")
    def mark(ok):
        return "PASS" if ok else "FAIL"
    print(f"  A  FK 末端位姿差 ~µm  (< {THRESH_FK_POS*1e6:.0f} µm / {THRESH_FK_ORI*1e6:.0f} µrad) : {mark(passA)}")
    print(f"  B  同种子同 K 关节差可忽略 (< {THRESH_REFINE:.0e} rad)            : {mark(passB)}")
    print(f"  C  精修器可微                                            : {mark(ok_grad)}")
    gate = passA and passB and ok_grad
    print(f"\n  >>> Gate-0: {mark(gate)} <<<")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
