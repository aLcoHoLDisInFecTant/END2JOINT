#!/usr/bin/env python3
"""阶段 1 评测：用生产 Pinocchio fk_correction 扫推理步数 K，画位置精度–K 曲线。

评测协议(实验计划书 §4)：所有模型一律用生产 Pinocchio fk_correction 扫 K，
测"给定部署精修器，该种子需要几步"。每个 K 强制跑满 K 步(tol=-1)以得到
可解释的单调曲线。在 *held-out 验证集* 上评测(in-distribution，未训练)。

用法:
  python ik_net_robust/eval_k_sweep.py \
      --runs C0:ik_net_robust/results_c0 C2:ik_net_robust/results_c2 \
      --n 1500 --out ik_net_robust/results_smoke
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_this_dir = os.path.abspath(os.path.dirname(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, ".."))
_example_dir = os.path.join(_project_root, "example")
for p in [_this_dir, _project_root, _example_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import paths, data_config
from dataloader import load_episode_files, episodes_to_arrays
from model import ResidualMLP
from fk_utils import load_ik, compute_ee_pose, fk_correction

K_LIST = [0, 1, 2, 3, 5, 8, 15]
TARGET_POS_MM = 1.0   # 部署位置达标线(mean)


def build_val_arrays(val_eps):
    """构造验证集 (X_raw, ee_target, y) —— gt 模式 prev_joints，无噪声。"""
    episodes = load_episode_files(paths["data_dir"])
    X, y = episodes_to_arrays(episodes, val_eps, add_noise=False)
    ee_target = X[:, :12]
    return X, ee_target, y


def predict_seed(model, scaler, X_raw, device):
    X_n = scaler.transform_X(X_raw)
    with torch.no_grad():
        pn = model(torch.tensor(X_n, dtype=torch.float32, device=device)).cpu().numpy()
    return scaler.inverse_y(pn)


def sweep_position_error(ik, q_seed, ee_target):
    """对每个 K 用生产 fk_correction 跑满 K 步，返回 {K: (pos_mean_mm, pos_max_mm)}。"""
    N = q_seed.shape[0]
    out = {}
    for K in K_LIST:
        errs = []
        for i in range(N):
            if K == 0:
                q = q_seed[i]
            else:
                qL, qR, _, _ = fk_correction(ik, q_seed[i].copy(), ee_target[i],
                                             damping=0.1, max_iter=K, tol=-1.0)
                q = np.concatenate([qL, qR])
            ee = compute_ee_pose(ik, q)
            pos = 0.5 * (np.linalg.norm(ee[:3] - ee_target[i, :3]) +
                         np.linalg.norm(ee[6:9] - ee_target[i, 6:9]))
            errs.append(pos)
        errs = np.array(errs)
        out[K] = (float(errs.mean()) * 1000.0, float(errs.max()) * 1000.0)
    return out


def k_star(curve, target_mm=TARGET_POS_MM):
    """达标(mean<target)的最小 K；不达标返回 None。"""
    for K in K_LIST:
        if curve[K][0] < target_mm:
            return K
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="形如 C0:dir C2:dir 的条件:目录 列表")
    parser.add_argument("--n", type=int, default=1500, help="验证集子采样帧数")
    parser.add_argument("--out", default=os.path.join(_this_dir, "results_smoke"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cpu")

    runs = [r.split(":", 1) for r in args.runs]

    # 取第一个 run 的 val_eps（三条件同一划分）
    with open(os.path.join(runs[0][1], "history.json")) as f:
        val_eps = json.load(f)["info"]["val_eps"]
    X_raw, ee_target, _ = build_val_arrays(val_eps)
    rng = np.random.default_rng(args.seed)
    if args.n < len(X_raw):
        idx = rng.choice(len(X_raw), size=args.n, replace=False)
        X_raw, ee_target = X_raw[idx], ee_target[idx]
    print(f"验证集评测帧数: {len(X_raw)} (val_eps={val_eps})")

    print("加载 Pinocchio FK 引擎 ...")
    ik = load_ik()

    results = {}
    for cond, d in runs:
        ckpt = torch.load(os.path.join(d, "best_model.pt"), map_location=device, weights_only=False)
        with open(os.path.join(d, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        model = ResidualMLP().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"\n[{cond}] {d}  (epoch {ckpt.get('epoch')}, val_ref {ckpt.get('val_ref_mm'):.3f} mm)")
        q_seed = predict_seed(model, scaler, X_raw, device)
        curve = sweep_position_error(ik, q_seed, ee_target)
        results[cond] = curve
        ks = k_star(curve)
        print(f"  {'K':>3} | {'pos mean(mm)':>12} | {'pos max(mm)':>11}")
        for K in K_LIST:
            print(f"  {K:>3} | {curve[K][0]:>12.3f} | {curve[K][1]:>11.3f}")
        print(f"  K* (mean<{TARGET_POS_MM}mm) = {ks}")

    # ── 叠加曲线 ──
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"C0": "#1976D2", "C1": "#388E3C", "C2": "#E64A19"}
    for cond, curve in results.items():
        means = [curve[K][0] for K in K_LIST]
        ax.plot(K_LIST, means, "o-", label=cond, color=colors.get(cond), linewidth=1.6)
    ax.axhline(TARGET_POS_MM, color="gray", ls="--", alpha=0.6, label=f"target {TARGET_POS_MM} mm")
    ax.set_xlabel("refine steps K"); ax.set_ylabel("EE position error mean (mm)")
    ax.set_yscale("log"); ax.set_title("Position accuracy vs K (Pinocchio fk_correction)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.out, "pos_accuracy_vs_K.png")
    plt.savefig(plot_path, dpi=150); plt.close(fig)

    # ── 结论 / Gate-1 ──
    summary = {"K_list": K_LIST, "target_pos_mm": TARGET_POS_MM,
               "curves": {c: {str(K): results[c][K] for K in K_LIST} for c in results},
               "k_star": {c: k_star(results[c]) for c in results}}
    with open(os.path.join(args.out, "k_sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*56}\nGate-1 判定（C2 曲线相对 C0 是否明显左移）\n{'='*56}")
    if "C0" in results and "C2" in results:
        for K in K_LIST:
            c0, c2 = results["C0"][K][0], results["C2"][K][0]
            better = "C2<C0" if c2 < c0 else "      "
            print(f"  K={K:>2}: C0 {c0:8.3f} mm | C2 {c2:8.3f} mm  {better}")
        ks0, ks2 = k_star(results["C0"]), k_star(results["C2"])
        print(f"\n  K* : C0={ks0}  C2={ks2}")
        if ks2 is not None and (ks0 is None or ks2 < ks0):
            print("  >>> Gate-1: 方向正确（C2 用更小 K 达标 / 曲线左移）<<<")
        else:
            print("  >>> Gate-1: 未见明显左移，需回阶段0排查 unroll/对齐 <<<")
    print(f"\n曲线: {plot_path}")


if __name__ == "__main__":
    main()
