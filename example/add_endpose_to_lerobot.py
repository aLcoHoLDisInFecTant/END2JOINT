#!/usr/bin/env python3
"""
为 LeRobot 数据集就地添加 `endpose` 列（基于 observation.state 经本地 FK 计算）。

- 复用项目本地 FK 引擎 actibot_fk.Arm_IK（Pinocchio + v3 URDF）。
- endpose 为 12 维：[eeL_x,eeL_y,eeL_z,eeL_roll,eeL_pitch,eeL_yaw,
                      eeR_x,eeR_y,eeR_z,eeR_roll,eeR_pitch,eeR_yaw]
- 直接写回每个 episode_*.parquet（新增 endpose 列）。
- 同步更新 meta/info.json、meta/stats.json、meta/episodes_stats.jsonl。

用法:
  python example/add_endpose_to_lerobot.py /path/to/lerobot_dataset_root
"""
import os, sys, glob, json, shutil, argparse
import numpy as np
import pandas as pd

_example_dir = os.path.abspath(os.path.dirname(__file__))
if _example_dir not in sys.path:
    sys.path.insert(0, _example_dir)
_project_dir = os.path.abspath(os.path.join(_example_dir, ".."))

from actibot_fk import Arm_IK
from pinocchio.rpy import matrixToRpy

URDF_PATH = os.path.join(_project_dir,
    "actibot_sdk/robot_description/v3/urdf/v3_urdf_251121-2.urdf")

ENDPOSE_NAMES = [
    "eeL_x", "eeL_y", "eeL_z", "eeL_roll", "eeL_pitch", "eeL_yaw",
    "eeR_x", "eeR_y", "eeR_z", "eeR_roll", "eeR_pitch", "eeR_yaw",
]
ENDPOSE_DIM = 12
SOURCE_COL = "observation.state"   # 用实际关节角计算 FK
ENDPOSE_COL = "endpose"


def find_parquet_files(root):
    pats = [
        os.path.join(root, "data", "chunk-*", "episode_*.parquet"),
        os.path.join(root, "data", "chunk-000", "episode_*.parquet"),
        os.path.join(root, "**", "episode_*.parquet"),
    ]
    for p in pats:
        files = sorted(glob.glob(p, recursive=True))
        if files:
            return files
    return []


def state_to_endpose(state_vec, ik):
    """16 维 state -> 19 维 q -> 12 维 endpose。"""
    s = np.asarray(state_vec, dtype=np.float64)
    q = np.zeros(19)
    q[5:12] = s[0:7]    # 左臂 7
    q[12:19] = s[7:14]  # 右臂 7
    T_l, T_r = ik.get_fk_solution(q)
    rpy_l = matrixToRpy(T_l[:3, :3])
    rpy_r = matrixToRpy(T_r[:3, :3])
    return np.concatenate([T_l[:3, 3], rpy_l, T_r[:3, 3], rpy_r]).astype(np.float32)


def compute_stats(arr):
    """arr: (N, 12) -> dict(min/max/mean/std/count)。"""
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def aggregate_global(ep_stats_list):
    """按 count 加权聚合各 episode 的 endpose 统计 -> 全局 min/max/mean/std。"""
    mins = np.array([s["min"] for s in ep_stats_list])
    maxs = np.array([s["max"] for s in ep_stats_list])
    means = np.array([s["mean"] for s in ep_stats_list])
    stds = np.array([s["std"] for s in ep_stats_list])
    counts = np.array([s["count"][0] for s in ep_stats_list], dtype=np.float64)
    total = counts.sum()
    w = counts[:, None]
    g_mean = (w * means).sum(axis=0) / total
    # E[x^2] = std^2 + mean^2
    g_ex2 = (w * (stds ** 2 + means ** 2)).sum(axis=0) / total
    g_var = np.clip(g_ex2 - g_mean ** 2, 0.0, None)
    return {
        "min": mins.min(axis=0).tolist(),
        "max": maxs.max(axis=0).tolist(),
        "mean": g_mean.tolist(),
        "std": np.sqrt(g_var).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="LeRobot 数据集根目录（含 data/ 与 meta/）")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = ap.parse_args()
    root = args.root.rstrip("/")

    files = find_parquet_files(root)
    if not files:
        print(f"错误: 未在 {root} 找到 episode_*.parquet")
        sys.exit(1)

    meta_dir = os.path.join(root, "meta")
    print(f"FK 引擎加载中 ...")
    ik = Arm_IK(URDF_PATH)
    print(f"找到 {len(files)} 个 episode 文件，开始添加 endpose 列 ...")

    ep_stats = {}  # episode_index -> stats dict
    for fpath in files:
        df = pd.read_parquet(fpath)
        if ENDPOSE_COL in df.columns:
            df = df.drop(columns=[ENDPOSE_COL])
        N = len(df)
        poses = np.zeros((N, ENDPOSE_DIM), dtype=np.float32)
        for i in range(N):
            poses[i] = state_to_endpose(df.iloc[i][SOURCE_COL], ik)
        cast = poses if args.dtype == "float32" else poses.astype(np.float64)
        df[ENDPOSE_COL] = list(cast)
        df.to_parquet(fpath, index=False)

        ep_idx = int(df.iloc[0]["episode_index"])
        ep_stats[ep_idx] = compute_stats(poses)
        print(f"  episode {ep_idx:6d}: {N:4d} 帧  -> {os.path.basename(fpath)}")

    # ---- 更新 meta/info.json ----
    info_path = os.path.join(meta_dir, "info.json")
    info = json.load(open(info_path))
    info.setdefault("features", {})[ENDPOSE_COL] = {
        "dtype": args.dtype,
        "shape": [ENDPOSE_DIM],
        "names": ENDPOSE_NAMES,
    }
    json.dump(info, open(info_path, "w"), indent=4, ensure_ascii=False)
    print(f"已更新 {info_path}")

    # ---- 更新 meta/episodes_stats.jsonl ----
    es_path = os.path.join(meta_dir, "episodes_stats.jsonl")
    if os.path.exists(es_path):
        lines = []
        with open(es_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ei = int(d["episode_index"])
                if ei in ep_stats:
                    d["stats"][ENDPOSE_COL] = ep_stats[ei]
                lines.append(json.dumps(d, ensure_ascii=False))
        with open(es_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"已更新 {es_path}")

    # ---- 更新 meta/stats.json（全局聚合）----
    stats_path = os.path.join(meta_dir, "stats.json")
    if os.path.exists(stats_path):
        stats = json.load(open(stats_path))
        stats[ENDPOSE_COL] = aggregate_global(list(ep_stats.values()))
        json.dump(stats, open(stats_path, "w"), indent=4, ensure_ascii=False)
        print(f"已更新 {stats_path}")

    print(f"\n完成! 共处理 {len(ep_stats)} 个 episode，endpose({ENDPOSE_DIM}维) 已就地写入。")


if __name__ == "__main__":
    main()
