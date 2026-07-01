#!/usr/bin/env python3
"""两阶段训练 IK-Net：合成全空间预训练 → 真实任务微调。

动机（IKNET_TEST_REPORT_zh.md §7.4.7）：IKNET 仅在任务录制数据上训练，关节空间
大片区域 OOD，纯网络在未见大幅度动作上外推差。本脚本：
  Stage A  在「全空间合成 FK 数据」上预训练，得到覆盖全工作空间的粗解基座；
  Stage B  在真实任务数据（0525 + groot）上小 lr 微调，恢复 in-distribution 精度。

关键：全程冻结同一个 scaler（在 真实train ∪ 合成train 并集上拟合），否则预训练
权重的输入归一化在微调时失效。模型选择以「真实 val 关节 MAE」为准（部署指标）。

用法:
  conda activate actibot_sdk    # 或 env -u PYTHONPATH .venv/bin/python ...
  python ik_net/train_staged.py
"""
import os
import sys
import time
import json
import pickle
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_example_dir = os.path.join(_project_dir, "example")
for p in [_example_dir, _project_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import hp, paths, staged
from dataloader import build_dataloaders, fit_scaler_on_dirs
from model import ResidualMLP
from fk_utils import load_ik
from train import set_seed, train_epoch, evaluate   # 复用单阶段训练原语


def run_stage(name, train_loader, val_loader, model, scaler, ik, device,
              epochs, lr, patience, target_deg, lr_step, lr_gamma):
    """训练一个阶段，按 val 关节 MAE 选最优。返回 (best_state_dict, history)。"""
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
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    results_dir = paths["staged_results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    real_dir = paths["data_dir"]
    real_extra = paths.get("extra_data_dirs", [])
    synth_dir = paths["synthetic_data_dir"]

    # ── Step 0: 共享冻结 scaler（仅真实 train，保留任务区分辨率、与部署一致）──
    # 注：曾用「真实∪合成」并集拟合，但合成全空间使 X/y.scale 膨胀 1.5~11×，
    # 压缩真实任务区分辨率 → in-distribution 精度退化。改为仅真实 train 拟合。
    print("拟合共享 scaler（仅真实 train 划分）...")
    scaler = fit_scaler_on_dirs([(real_dir, real_extra)])
    print(f"  X scale[:3]={np.round(scaler.X.scale_[:3],3)}  y scale[:3]={np.round(scaler.y.scale_[:3],3)}")

    # ── 数据（注入冻结 scaler）──
    print("加载合成数据 ...")
    syn_train, syn_val, syn_test, _, syn_info = build_dataloaders(synth_dir, extra_dirs=[], scaler=scaler)
    print(f"  合成 train/val/test 样本: {syn_info['train_samples']}/{syn_info['val_samples']}/{syn_info['test_samples']}")
    print("加载真实数据 ...")
    real_train, real_val, real_test, _, real_info = build_dataloaders(real_dir, extra_dirs=real_extra, scaler=scaler)
    print(f"  真实 train/val/test 样本: {real_info['train_samples']}/{real_info['val_samples']}/{real_info['test_samples']}")
    print(f"  真实 test episodes (held-out): {real_info['test_eps']}")

    print("加载 FK 引擎 ...")
    ik = load_ik()

    model = ResidualMLP().to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # ── Stage A: 合成预训练 ──
    _, hist_pre = run_stage(
        "pretrain", syn_train, syn_val, model, scaler, ik, device,
        epochs=staged["pretrain_epochs"], lr=staged["pretrain_lr"],
        patience=staged["pretrain_patience"], target_deg=-1.0,  # 预训练不早停于精度
        lr_step=staged["pretrain_lr_step"], lr_gamma=staged["pretrain_lr_gamma"])

    # ── Stage B: 真实微调（从预训练权重续训）──
    _, hist_ft = run_stage(
        "finetune", real_train, real_val, model, scaler, ik, device,
        epochs=staged["finetune_epochs"], lr=staged["finetune_lr"],
        patience=staged["finetune_patience"], target_deg=hp["target_joint_deg"],
        lr_step=staged["finetune_lr_step"], lr_gamma=staged["finetune_lr_gamma"])

    # ── 保存最优（微调后）模型 + 共享 scaler ──
    ckpt_name = staged["ckpt_name"]
    torch.save({"model_state_dict": model.state_dict(),
                "best_joint_deg": hist_ft["best_joint_deg"],
                "best_epoch": hist_ft["best_epoch"]},
               os.path.join(results_dir, ckpt_name))
    with open(os.path.join(results_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # ── 真实 held-out test 评估（保护 in-distribution 的硬约束）──
    print("\n真实 held-out test 评估 ...")
    test_result = evaluate(real_test, model, scaler, device, ik=ik)
    print(f"  Joint MAE:  {test_result.get('joint_mae_deg',0):.3f}°")
    print(f"  FK Pos Err: {test_result.get('fk_pos_err_mean',0)*1000:.3f} mm")
    print(f"  FK Ori Err: {test_result.get('fk_ori_err_mean',0):.4f} rad")

    history = {"pretrain": hist_pre, "finetune": hist_ft,
               "real_test_result": test_result,
               "real_info": real_info, "synth_info": syn_info}
    with open(os.path.join(results_dir, "history.json"), "w") as f:
        def convert(o):
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.integer): return int(o)
            raise TypeError
        json.dump(history, f, indent=2, default=convert)

    # ── 损失曲线（两阶段拼接）──
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    n_pre = len(hist_pre["joint_deg"])
    ax.plot(range(1, n_pre + 1), hist_pre["joint_deg"], label="pretrain val joint(°)", color="#9C27B0", lw=0.9)
    ax.plot(range(n_pre + 1, n_pre + 1 + len(hist_ft["joint_deg"])), hist_ft["joint_deg"],
            label="finetune val joint(°)", color="#FF5722", lw=0.9)
    ax.axvline(x=n_pre + 0.5, color="gray", ls="--", alpha=0.5, label="stage switch")
    ax.set_xlabel("Epoch (concat)"); ax.set_ylabel("Val Joint MAE (°)")
    ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title("Staged training: synthetic pretrain → real finetune")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    print(f"\n结果保存至 {results_dir}/  (ckpt={ckpt_name}, scaler.pkl, history.json, loss_curve.png)")
    print("Done.")


if __name__ == "__main__":
    main()
