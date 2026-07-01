#!/usr/bin/env python3
"""训练 IK-Net 残差 MLP。

用法:
  conda activate actibot_sdk
  python ik_net/train.py

每个 epoch 结束后在验证集上计算 FK 一致性误差作为评估指标，
选择验证集 FK 位置误差最低的模型保存。
"""
import os
import sys
import time
import json
import copy
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ── 项目路径 ──
_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_example_dir = os.path.join(_project_dir, "example")
for p in [_example_dir, _project_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import hp, paths
from dataloader import build_dataloaders, episodes_to_arrays, Scaler
from model import ResidualMLP
from fk_utils import load_ik, batch_fk_error


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(loader, model, scaler, device, ik=None):
    """验证/测试集评估。

    返回: {loss, pos_err_mean, pos_err_max, ori_err_mean, ori_err_max}
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    n_batches = 0

    all_q_pred, all_y, all_X = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()
            n_batches += 1

            all_q_pred.append(pred.cpu().numpy())
            all_y.append(y_batch.cpu().numpy())
            all_X.append(X_batch.cpu().numpy())

    q_pred_norm = np.vstack(all_q_pred)
    y_norm = np.vstack(all_y)

    result = {"loss": total_loss / max(n_batches, 1)}

    # 关节角平均误差（度）
    q_gt = scaler.inverse_y(y_norm)
    q_pred = scaler.inverse_y(q_pred_norm)
    joint_mae_rad = float(np.mean(np.abs(q_pred - q_gt)))
    result["joint_mae_deg"] = joint_mae_rad * 180.0 / np.pi

    # FK 一致性（需要 ik 实例）
    if ik is not None:
        X_np = scaler.inverse_X(np.vstack(all_X))
        ee_target = X_np[:, :12]  # 前 12 维是 ee_pose_t
        pos_m, pos_max, ori_m, ori_max = batch_fk_error(ik, q_pred, ee_target)
        result.update({
            "fk_pos_err_mean": pos_m,
            "fk_pos_err_max": pos_max,
            "fk_ori_err_mean": ori_m,
            "fk_ori_err_max": ori_max,
        })

    return result


def train_epoch(loader, model, optimizer, criterion, device, scaler):
    """训练一个 epoch，返回平均 loss。"""
    model.train()
    total_loss = 0
    n = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        n += X_batch.size(0)
    return total_loss / max(n, 1)


def run_stage(name, train_loader, val_loader, model, scaler, ik, device,
              epochs, lr, patience, target_deg, lr_step, lr_gamma):
    """训练一个阶段，按 val 关节 MAE 选最优。返回 (best_state_dict, history)。

    供联合训练 (train_joint.py) 复用的通用训练循环：AdamW + StepLR，
    以「val 关节 MAE」早停/选优，达 target_deg 即停。
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    criterion = nn.MSELoss()

    best_deg, best_epoch, best_state = float("inf"), -1, None
    train_hist, val_hist, deg_hist, patience_ctr = [], [], [], 0

    print(f"\n========== Stage [{name}]  epochs={epochs} lr={lr} ==========")
    print(f"{'Epoch':>6} | {'Train':>10} | {'Val':>10} | {'Joint(°)':>8} | {'FK Pos(mm)':>10} | Time")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch(train_loader, model, optimizer, criterion, device, scaler)
        val = evaluate(val_loader, model, scaler, device, ik=ik)
        deg = val.get("joint_mae_deg", 0.0)
        fk_pos = val.get("fk_pos_err_mean", 0.0) * 1000
        train_hist.append(tr); val_hist.append(val["loss"]); deg_hist.append(deg)
        print(f"{epoch:>6d} | {tr:>10.6f} | {val['loss']:>10.6f} | {deg:>8.3f} | {fk_pos:>10.3f} | {time.time()-t0:.1f}s")

        improved = deg < best_deg
        if improved:
            best_deg, best_epoch = deg, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if deg < target_deg:
            print(f"  ✓ [{name}] 达到目标精度 {deg:.3f}° < {target_deg}°")
            break
        scheduler.step()
        if patience_ctr >= patience:
            print(f"  Early stopping [{name}] at epoch {epoch} (no improve {patience})")
            break

    print(f"  [{name}] best: epoch {best_epoch}, joint MAE = {best_deg:.3f}°")
    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"train_loss": train_hist, "val_loss": val_hist, "joint_deg": deg_hist,
               "best_epoch": best_epoch, "best_joint_deg": best_deg}
    return best_state, history


def main():
    set_seed(hp["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # ── 加载数据 ──
    print("加载数据...")
    train_loader, val_loader, test_loader, scaler, info = build_dataloaders(
        paths["data_dir"])
    print(f"  训练: {info['train_samples']} 样本 ({len(info['train_eps'])} episodes)")
    print(f"  验证: {info['val_samples']} 样本 ({len(info['val_eps'])} episodes)")
    print(f"  测试: {info['test_samples']} 样本 ({len(info['test_eps'])} episodes)")

    # ── FK 引擎 ──
    print("加载 FK 引擎...")
    ik = load_ik()

    # ── 模型 ──
    model = ResidualMLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hp["lr_step"],
                                           gamma=hp["lr_gamma"])
    criterion = nn.MSELoss()

    # ── 训练 ──
    os.makedirs(paths["results_dir"], exist_ok=True)
    best_joint_deg, best_epoch = float("inf"), -1
    train_history, val_history, deg_history, patience_counter = [], [], [], 0

    target_deg = hp["target_joint_deg"]
    print(f"\n训练 {hp['num_epochs']} epochs ... 目标: 关节角误差 < {target_deg}°")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Joint(°)':>8} | {'FK Pos(mm)':>10} | {'FK Ori(rad)':>10} | Time")
    print("-" * 80)

    for epoch in range(1, hp["num_epochs"] + 1):
        t0 = time.time()

        # ── train ──
        train_loss = train_epoch(train_loader, model, optimizer, criterion, device, scaler)
        train_history.append(train_loss)

        # ── validate ──
        val_result = evaluate(val_loader, model, scaler, device, ik=ik)
        val_history.append(val_result["loss"])
        joint_deg = val_result.get("joint_mae_deg", 0)

        fk_pos = val_result.get("fk_pos_err_mean", 0.0) * 1000
        fk_ori = val_result.get("fk_ori_err_mean", 0.0)
        elapsed = time.time() - t0

        print(f"{epoch:>6d} | {train_loss:>10.6f} | {val_result['loss']:>10.6f} | "
              f"{joint_deg:>8.3f} | {fk_pos:>10.3f} | {fk_ori:>10.4f} | {elapsed:.1f}s")

        deg_history.append(joint_deg)

        # 达到目标精度 → 提前停止
        if joint_deg < target_deg:
            print(f"\n✓ 达到目标精度: joint MAE = {joint_deg:.3f}° < {target_deg}°")
            best_joint_deg, best_epoch = joint_deg, epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_result["loss"], "joint_mae_deg": joint_deg},
                       os.path.join(paths["results_dir"], hp.get("ckpt_name", "best_model.pt")))
            with open(os.path.join(paths["results_dir"], "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            break

        scheduler.step()

        # ── 依据关节角误差保存最佳模型 ──
        if joint_deg < best_joint_deg:
            best_joint_deg, best_epoch = joint_deg, epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_result["loss"],
                "joint_mae_deg": joint_deg,
                "fk_pos_err_mean": val_result.get("fk_pos_err_mean", 0.0),
                "fk_ori_err_mean": val_result.get("fk_ori_err_mean", 0.0),
            }, os.path.join(paths["results_dir"], hp.get("ckpt_name", "best_model.pt")))
            # Scaler 单独保存（torch.load 无法反序列化非 torch 对象）
            with open(os.path.join(paths["results_dir"], "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            patience_counter = 0
        else:
            patience_counter += 1

        # ── early stopping ──
        if patience_counter >= hp["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {hp["patience"]} epochs)")
            break

    print(f"\n最佳模型: epoch {best_epoch}, joint MAE = {best_joint_deg:.3f}°")

    # ── 测试集评估 ──
    print("\n测试集评估...")
    checkpoint = torch.load(os.path.join(paths["results_dir"], hp.get("ckpt_name", "best_model.pt")), map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate(test_loader, model, scaler, device, ik=ik)
    print(f"  Joint MAE:  {test_result.get('joint_mae_deg',0):.3f}°")
    print(f"  MSE Loss:   {test_result['loss']:.6f}")
    print(f"  FK Pos Err: {test_result.get('fk_pos_err_mean', 0) * 1000:.3f} mm")
    print(f"  FK Ori Err: {test_result.get('fk_ori_err_mean', 0):.4f} rad")

    # ── 保存结果 ──
    history = {
        "train_loss": train_history,
        "val_loss": val_history,
        "joint_deg": deg_history,
        "best_epoch": best_epoch,
        "best_joint_deg": best_joint_deg,
        "test_result": test_result,
        "info": info,
    }
    with open(os.path.join(paths["results_dir"], "history.json"), "w") as f:
        # 转换 numpy 标量
        def convert(o):
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            raise TypeError
        json.dump(history, f, indent=2, default=convert)

    # ── 绘制损失曲线 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_history) + 1)

    ax1.plot(epochs, train_history, label="Train Loss", color="#2196F3", linewidth=0.8)
    ax1.plot(epochs, val_history, label="Val Loss", color="#FF5722", linewidth=0.8)
    ax1.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best (ep {best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 从 history.json 读 FK 误差（仅在 val_result 中记录了 FK Pos）
    # 重新组装 FK pos history — 从 val_history 不能直接得到，需要在循环中记录
    # 直接在训练循环中已记录了 best，这里只画 loss 曲线
    ax2.semilogy(epochs, train_history, label="Train Loss", color="#2196F3", linewidth=0.8)
    ax2.semilogy(epochs, val_history, label="Val Loss", color="#FF5722", linewidth=0.8)
    ax2.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss (log)")
    ax2.set_title("Loss (Log Scale)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(paths["results_dir"], "loss_curve.png"), dpi=150)
    plt.close(fig)
    print(f"损失曲线: {paths["results_dir"]}/loss_curve.png")

    print(f"\n结果保存至 {paths["results_dir"]}/")
    print("Done.")


if __name__ == "__main__":
    main()
