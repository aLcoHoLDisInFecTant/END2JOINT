"""数据加载与预处理

单文件包含全部所需数据（由 compute_fk_action.py 生成）：
  eeL/R_xyzrpy     → 输入 ee_pose (12D)
  state_*          → 输入 state_{t-1} (14D)
  L/R_sh_pitch...  → 目标 action_t (14D)
"""
import glob
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from config import hp, data_config

EE_COLS = data_config["col_eeL"] + data_config["col_eeR"]
ACTION_COLS = data_config["col_action_l"] + data_config["col_action_r"]
STATE_COLS  = data_config["col_state_l"] + data_config["col_state_r"]


def load_episode_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*_fk.parquet")))
    episodes = {}
    for fpath in files:
        base = os.path.basename(fpath)
        ep = int(base.split("_")[1])
        episodes[ep] = pd.read_parquet(fpath)
    return episodes


def split_episodes(episodes, seed=hp["seed"]):
    eps = sorted(episodes.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(eps)
    n = len(eps)
    n_train = int(n * hp["train_ratio"])
    n_val = int(n * hp["val_ratio"])
    return (sorted(eps[:n_train]), sorted(eps[n_train:n_train + n_val]),
            sorted(eps[n_train + n_val:]))


def episodes_to_arrays(episodes, ep_list):
    """构造样本（单文件，无需双源）。

    X = [ee_pose_t(12), state_{t-1}(14)]  → 26D
    y = action_t                          → 14D
    """
    all_X, all_y = [], []
    for ep in ep_list:
        df = episodes[ep]
        ee_t = df[EE_COLS].values.astype(np.float64)            # (n, 12)
        state = df[STATE_COLS].values.astype(np.float64)        # (n, 14)
        action = df[ACTION_COLS].values.astype(np.float64)      # (n, 14)
        state_prev = np.vstack([state[0:1], state[:-1]])         # (n, 14)
        X = np.hstack([ee_t, state_prev])  # (n, 26)
        all_X.append(X)
        all_y.append(action.copy())
    return np.vstack(all_X), np.vstack(all_y)


class Scaler:
    def __init__(self):
        self.X = StandardScaler()
        self.y = StandardScaler()

    def fit(self, X, y): self.X.fit(X); self.y.fit(y); return self
    def transform_X(self, X): return self.X.transform(X)
    def transform_y(self, y): return self.y.transform(y)
    def inverse_y(self, y_norm): return self.y.inverse_transform(y_norm)
    def inverse_X(self, X_norm): return self.X.inverse_transform(X_norm)


class IKDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def _load_merged_episodes(data_dir, extra_dirs):
    """加载主目录 + 额外目录，重编号避免 episode id 冲突。"""
    episodes = load_episode_files(data_dir)
    if extra_dirs:
        for edir in extra_dirs:
            ep_offset = max(episodes.keys()) + 1
            for ep_idx, df in load_episode_files(edir).items():
                episodes[ep_offset + ep_idx] = df
    return episodes


def collect_train_split_arrays(data_dir, extra_dirs=None):
    """返回某数据源（主 + 额外）train 划分的 (X, y)，用于在并集上拟合 scaler。

    划分逻辑与 build_dataloaders 完全一致（同 seed、同比例），保证「拟合 scaler
    用的 train 子集」与「实际训练用的 train 子集」一一对应、无泄漏。
    """
    episodes = _load_merged_episodes(data_dir, extra_dirs or [])
    train_eps, _, _ = split_episodes(episodes)
    return episodes_to_arrays(episodes, train_eps)


def fit_scaler_on_dirs(sources):
    """在多个数据源 train 划分的并集上拟合一个共享 Scaler。

    sources: list[(data_dir, extra_dirs)]，例如
        [(real_dir, [extra_real]), (synthetic_dir, [])]
    用于两阶段训练：scaler 必须覆盖全空间（含合成数据），并在预训练/微调
    全程冻结复用，否则预训练权重的输入归一化在微调时失效。
    """
    Xs, ys = [], []
    for data_dir, extra_dirs in sources:
        X, y = collect_train_split_arrays(data_dir, extra_dirs)
        Xs.append(X)
        ys.append(y)
    return Scaler().fit(np.vstack(Xs), np.vstack(ys))


def build_dataloaders(data_dir, extra_dirs=None, scaler=None):
    """从目录加载数据（每文件含 ee + state + action）。

    extra_dirs: 额外数据目录列表；None 时回退到 config.paths["extra_data_dirs"]
                （保持旧 train.py 行为不变）。传 [] 可显式禁用额外数据。
    scaler:     传入则跳过拟合、直接复用（两阶段共享冻结 scaler）；None 时在
                本数据源 train 划分上拟合。
    """
    if extra_dirs is None:
        from config import paths
        extra_dirs = paths.get("extra_data_dirs", [])

    episodes = _load_merged_episodes(data_dir, extra_dirs)
    if extra_dirs:
        print(f"  合并额外数据: {sum(1 for d in extra_dirs for _ in load_episode_files(d))} episodes")

    train_eps, val_eps, test_eps = split_episodes(episodes)
    X_train, y_train = episodes_to_arrays(episodes, train_eps)
    X_val, y_val = episodes_to_arrays(episodes, val_eps)
    X_test, y_test = episodes_to_arrays(episodes, test_eps)
    if scaler is None:
        scaler = Scaler().fit(X_train, y_train)

    def make_loader(X, y, shuffle):
        ds = IKDataset(scaler.transform_X(X), scaler.transform_y(y))
        return DataLoader(ds, batch_size=hp["batch_size"], shuffle=shuffle, pin_memory=True)

    info = {"train_eps": train_eps, "val_eps": val_eps, "test_eps": test_eps,
            "train_samples": len(X_train), "val_samples": len(X_val),
            "test_samples": len(X_test)}
    return (make_loader(X_train, y_train, True),
            make_loader(X_val, y_val, False),
            make_loader(X_test, y_test, False), scaler, info)
