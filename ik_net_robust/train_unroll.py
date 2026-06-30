#!/usr/bin/env python3
"""阶段 1 Smoke Test 训练器：C0(关节 MSE) vs C2(端到端 unroll 可微精修)。

三条件其余变量严格一致（同网络、同数据、同优化器与预算、同输入编码），
唯一自变量是损失定义 + 有无 unroll：

  C0: 损失 = 关节 MSE（归一化空间，对录制 action），训练时无精修器。
  C2: 损失 = FK 任务损失，算在 *K_train 步可微精修后* 的关节上（refiner-aware）。
      C2a: 额外加一项对裸种子的 FK 任务损失（--seed-loss-weight > 0，双监督，更稳）。
      C2b: 只监督精修后输出（--seed-loss-weight 0，默认）。

用法:
  python ik_net_robust/train_unroll.py --condition C0 --epochs 200 --out ik_net_robust/results_c0
  python ik_net_robust/train_unroll.py --condition C2 --ktrain 2 --epochs 200 --out ik_net_robust/results_c2
"""
import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_this_dir = os.path.abspath(os.path.dirname(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, ".."))
_example_dir = os.path.join(_project_root, "example")
for p in [_this_dir, _project_root, _example_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import hp, paths
from dataloader import build_dataloaders
from model import ResidualMLP
from diff_refiner import DiffRefiner


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scaler_tensors(scaler, device, dtype):
    """sklearn StandardScaler 的 mean/scale → torch 张量（用于可微反标准化）。"""
    x_mean = torch.tensor(scaler.X.mean_, dtype=dtype, device=device)
    x_scale = torch.tensor(scaler.X.scale_, dtype=dtype, device=device)
    y_mean = torch.tensor(scaler.y.mean_, dtype=dtype, device=device)
    y_scale = torch.tensor(scaler.y.scale_, dtype=dtype, device=device)
    return x_mean, x_scale, y_mean, y_scale


@torch.no_grad()
def validate(loader, model, refiner, stats, device, k_val, w_ori):
    """验证指标：精修 k_val 步后的末端位置误差(mm) + 裸种子(K=0)位置误差(mm)。

    用 torch 精修器（与 Pinocchio µm 级一致、无 casadi、可批量），两条件共用。
    """
    model.eval()
    x_mean, x_scale, y_mean, y_scale = stats
    seed_err, ref_err = [], []
    for X_n, _ in loader:
        X_n = X_n.to(device)
        q_seed = model(X_n) * y_scale + y_mean
        ee_target = X_n[:, :12] * x_scale[:12] + x_mean[:12]
        seed_err.append(refiner.pos_error(q_seed, ee_target).cpu())
        q_ref = refiner.refine(q_seed, ee_target, K=k_val)
        ref_err.append(refiner.pos_error(q_ref, ee_target).cpu())
    seed_mm = float(torch.cat(seed_err).mean()) * 1000.0
    ref_mm = float(torch.cat(ref_err).mean()) * 1000.0
    return seed_mm, ref_mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["C0", "C2"], required=True)
    parser.add_argument("--ktrain", type=int, default=2, help="C2 训练时 unroll 步数")
    parser.add_argument("--kval", type=int, default=2, help="验证用精修步数")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--w-ori", type=float, default=0.1, help="FK 损失中姿态权重")
    parser.add_argument("--seed-loss-weight", type=float, default=0.0,
                        help="C2a：对裸种子额外加 FK 损失的权重(>0 启用双监督)")
    parser.add_argument("--out", default=None, help="输出目录(默认 results_<cond>)")
    parser.add_argument("--seed", type=int, default=hp["seed"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    out_dir = args.out or os.path.join(_this_dir, f"results_{args.condition.lower()}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"条件={args.condition}  device={device}  out={out_dir}")
    if args.condition == "C2":
        print(f"  K_train={args.ktrain}  w_ori={args.w_ori}  seed_loss_w={args.seed_loss_weight}"
              f"  ({'C2a 双监督' if args.seed_loss_weight > 0 else 'C2b 仅精修后'})")

    # ── 数据（三条件同一套划分/编码）──
    train_loader, val_loader, _, scaler, info = build_dataloaders(paths["data_dir"])
    print(f"  训练 {info['train_samples']} 样本 / {len(info['train_eps'])} eps；"
          f"验证 {info['val_samples']} / {len(info['val_eps'])} eps")
    stats = scaler_tensors(scaler, device, dtype)
    x_mean, x_scale, y_mean, y_scale = stats

    # ── 精修器（C2 训练用 + 两条件验证用）──
    refiner = DiffRefiner.from_npz(
        os.path.join(_this_dir, "refiner_consts.npz"), dtype=dtype).to(device)

    model = ResidualMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hp["lr_step"], gamma=hp["lr_gamma"])
    mse = nn.MSELoss()

    best_metric, best_epoch = float("inf"), -1
    hist = {"train_loss": [], "val_seed_mm": [], "val_ref_mm": []}
    print(f"\n{'Epoch':>6} | {'TrainLoss':>11} | {'Seed(mm)':>9} | {f'Ref@{args.kval}(mm)':>9} | Time")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for X_n, y_n in train_loader:
            X_n, y_n = X_n.to(device), y_n.to(device)
            optimizer.zero_grad()

            if args.condition == "C0":
                loss = mse(model(X_n), y_n)
            else:  # C2
                q_seed = model(X_n) * y_scale + y_mean
                ee_target = X_n[:, :12] * x_scale[:12] + x_mean[:12]
                q_ref = refiner.refine(q_seed, ee_target, K=args.ktrain)
                loss = refiner.pose_loss(q_ref, ee_target, w_ori=args.w_ori)
                if args.seed_loss_weight > 0:  # C2a 双监督
                    loss = loss + args.seed_loss_weight * refiner.pose_loss(
                        q_seed, ee_target, w_ori=args.w_ori)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item() * X_n.size(0)
            n += X_n.size(0)
        scheduler.step()
        train_loss = run_loss / max(n, 1)

        seed_mm, ref_mm = validate(val_loader, model, refiner, stats, device, args.kval, args.w_ori)
        hist["train_loss"].append(train_loss)
        hist["val_seed_mm"].append(seed_mm)
        hist["val_ref_mm"].append(ref_mm)
        print(f"{epoch:>6d} | {train_loss:>11.6f} | {seed_mm:>9.3f} | {ref_mm:>9.3f} | {time.time()-t0:.1f}s")

        # 模型选择：验证集精修 k_val 步后位置误差最低（两条件同口径）
        if ref_mm < best_metric:
            best_metric, best_epoch = ref_mm, epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "condition": args.condition, "ktrain": args.ktrain,
                        "val_ref_mm": ref_mm, "val_seed_mm": seed_mm},
                       os.path.join(out_dir, "best_model.pt"))
            with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)

    print(f"\n最佳: epoch {best_epoch}  val Ref@{args.kval} = {best_metric:.3f} mm")

    history = {"condition": args.condition, "ktrain": args.ktrain, "epochs": args.epochs,
               "w_ori": args.w_ori, "seed_loss_weight": args.seed_loss_weight,
               "best_epoch": best_epoch, "best_val_ref_mm": best_metric,
               "info": {k: info[k] for k in ("train_eps", "val_eps", "test_eps",
                                             "train_samples", "val_samples", "test_samples")},
               "history": hist}
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o))
    print(f"结果保存至 {out_dir}/")


if __name__ == "__main__":
    main()
