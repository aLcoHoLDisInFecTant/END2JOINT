#!/usr/bin/env python3
"""联合训练 IK-Net：真实任务数据 + 操作包络合成数据（数据增强）。

动机（IKNET_TEST_REPORT_zh.md §7.4.7 + 本轮诊断）：ep001 的 OOD 是右臂越过训练
边界一点点（R_sh_yaw 达 11.7σ、54% 帧出界），属「任务流形外缘」而非深内部。
均匀全空间合成对这个小 MLP 适得其反（自己 val 都 98mm）。改用「操作包络」合成
（example/make_synthetic_fullspace.py --mode envelope，径向扩展真实流形）做数据增强。

为什么联合训练而非预训练→微调：上一轮 staged 因灾难性遗忘零 OOD 收益；包络合成
离流形近、可学，全程在场即不被遗忘，又比均匀全空间稀释小得多。

关键：合成数据只进 **训练集**；val/test 用 **纯真实 held-out**（与基线同 seed=42
划分），保证 in-distribution 指标诚实可比。scaler 仅在真实 train 上拟合（部署一致）。

用法:
  conda activate actibot_sdk   # 或 env -u PYTHONPATH .venv/bin/python ...
  python ik_net/train_joint.py
"""
import os
import sys
import pickle
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_example_dir = os.path.join(_project_dir, "example")
for p in [_example_dir, _project_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import hp, paths, staged
from dataloader import (build_dataloaders, fit_scaler_on_dirs,
                        collect_train_split_arrays, IKDataset)
from model import ResidualMLP
from fk_utils import load_ik
from train import set_seed
from train_staged import run_stage   # 复用按 val 关节 MAE 选最优的训练循环


def main():
    set_seed(hp["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    results_dir = paths["joint_results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    real_dir = paths["data_dir"]
    real_extra = paths.get("extra_data_dirs", [])
    env_dir = paths["envelope_data_dir"]

    # ── scaler：仅真实 train（部署一致、保留任务区分辨率）──
    scaler = fit_scaler_on_dirs([(real_dir, real_extra)])

    # ── val/test：纯真实 held-out（与基线同 seed 划分）──
    _, real_val, real_test, _, ri = build_dataloaders(real_dir, extra_dirs=real_extra, scaler=scaler)
    print(f"真实 train/val/test 样本: {ri['train_samples']}/{ri['val_samples']}/{ri['test_samples']}")
    print(f"真实 held-out test episodes: {ri['test_eps']}")

    # ── train：真实 train + 包络合成（合成仅进训练）──
    Xr, yr = collect_train_split_arrays(real_dir, real_extra)
    Xs, ys = collect_train_split_arrays(env_dir, [])
    print(f"训练样本: 真实 {len(Xr)} + 包络合成 {len(Xs)} = {len(Xr)+len(Xs)}  (合成占比 {len(Xs)/(len(Xr)+len(Xs)):.0%})")
    X = np.vstack([Xr, Xs]); y = np.vstack([yr, ys])
    train_ds = IKDataset(scaler.transform_X(X), scaler.transform_y(y))
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, pin_memory=True)

    ik = load_ik()
    model = ResidualMLP().to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # ── 联合训练（从头，按真实 val 关节 MAE 选最优）──
    run_stage("joint", train_loader, real_val, model, scaler, ik, device,
              epochs=hp["num_epochs"], lr=hp["learning_rate"],
              patience=hp["patience"], target_deg=hp["target_joint_deg"],
              lr_step=hp["lr_step"], lr_gamma=hp["lr_gamma"])

    # ── 保存 + 真实 held-out test 评估 ──
    from train import evaluate
    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(results_dir, staged["ckpt_name"]))
    with open(os.path.join(results_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("\n真实 held-out test 评估 ...")
    tr = evaluate(real_test, model, scaler, device, ik=ik)
    print(f"  Joint MAE:  {tr.get('joint_mae_deg',0):.3f}°")
    print(f"  FK Pos Err: {tr.get('fk_pos_err_mean',0)*1000:.3f} mm")
    print(f"  FK Ori Err: {tr.get('fk_ori_err_mean',0):.4f} rad")

    with open(os.path.join(results_dir, "history.json"), "w") as f:
        json.dump({"real_test_result": tr, "real_info": ri,
                   "n_real_train": len(Xr), "n_synth_train": len(Xs)}, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating) else int(o))
    print(f"\n结果保存至 {results_dir}/")


if __name__ == "__main__":
    main()
