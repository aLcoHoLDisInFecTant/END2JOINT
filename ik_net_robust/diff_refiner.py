"""Torch 可微精修器（训练用）。

对齐生产 Pinocchio `fk_correction`（见 fk_utils.py）：同一 `actibot` URDF、
锁 4 个夹爪关节、14 臂 DOF、`ee_left`/`ee_right` 帧、6D 误差
`[t_target - t, log3(R_target·Rᵀ)]`、`LOCAL_WORLD_ALIGNED` 6×7 雅可比、
阻尼 λ=0.1、`dq` 截断 ±0.05、左右臂分别迭代。

FK 不依赖 pytorch_kinematics：常数（各关节 jointPlacement、局部旋转轴、
基座与末端帧变换、关节限位）从 Pinocchio 缩减模型一次性抽取，缓存为 torch
buffer / npz，之后训练无需 Pinocchio。全流程可微，梯度可回传到种子与网络权重。
"""
import os
import numpy as np
import torch
import torch.nn as nn

# 缩减模型中左右臂的关节 id（见 actibot_fk.Arm_IK，body 0..4，左臂 6..12，右臂 13..19）
_LEFT_JOINT_IDS = list(range(6, 13))
_RIGHT_JOINT_IDS = list(range(13, 20))


# ════════════════════════════════════════════════════════════════════
#  torch 旋转工具（批量、可微）
# ════════════════════════════════════════════════════════════════════

def _rodrigues_fixed_axis(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """绕固定单位轴 axis(3,) 旋转 angle(...) 弧度，返回旋转矩阵 (...,3,3)。"""
    a = axis / torch.linalg.norm(axis)
    K = torch.zeros(3, 3, dtype=axis.dtype, device=axis.device)
    K[0, 1], K[0, 2] = -a[2], a[1]
    K[1, 0], K[1, 2] = a[2], -a[0]
    K[2, 0], K[2, 1] = -a[1], a[0]
    KK = K @ K
    s = torch.sin(angle)[..., None, None]
    c = (1.0 - torch.cos(angle))[..., None, None]
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return eye + s * K + c * KK


def log3(R: torch.Tensor) -> torch.Tensor:
    """SO(3) 对数映射，返回旋转向量 (...,3)，与 pinocchio.log3 同定义。

    用 atan2 + 下夹 sin 替代 arccos，使梯度在零旋转(R≈I)处仍有限——否则
    精修收敛时姿态误差趋零会让 arccos 的导数发散、产生 NaN。
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
    # sin θ 下夹离零，保证 sqrt / 除法 / atan2 的梯度有限；
    # θ < 1e-6 时下式触发，但此时 w≈0，数值偏差可忽略
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos * cos, min=1e-12))
    theta = torch.atan2(sin_theta, cos)
    w = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)  # = 2 sinθ · axis
    small = theta < 1e-3
    coeff = torch.where(small, 0.5 + theta ** 2 / 12.0, theta / (2.0 * sin_theta))
    return coeff[..., None] * w


def rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """(...,3) roll-pitch-yaw → 旋转矩阵 (...,3,3)，与 pinocchio.rpy.rpyToMatrix 同定义
    (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))。"""
    r, p, y = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    zero = torch.zeros_like(r)
    one = torch.ones_like(r)
    Rx = torch.stack([one, zero, zero, zero, cr, -sr, zero, sr, cr], -1).reshape(*r.shape, 3, 3)
    Ry = torch.stack([cp, zero, sp, zero, one, zero, -sp, zero, cp], -1).reshape(*r.shape, 3, 3)
    Rz = torch.stack([cy, -sy, zero, sy, cy, zero, zero, zero, one], -1).reshape(*r.shape, 3, 3)
    return Rz @ Ry @ Rx


def matrix_to_rpy(R: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 (...,3,3) → roll-pitch-yaw (...,3)，与 pinocchio.rpy.matrixToRpy 同定义。"""
    pitch = torch.arcsin(torch.clamp(-R[..., 2, 0], -1.0, 1.0))
    roll = torch.arctan2(R[..., 2, 1], R[..., 2, 2])
    yaw = torch.arctan2(R[..., 1, 0], R[..., 0, 0])
    return torch.stack([roll, pitch, yaw], dim=-1)


# ════════════════════════════════════════════════════════════════════
#  可微精修器
# ════════════════════════════════════════════════════════════════════

class DiffRefiner(nn.Module):
    """torch 可微 DLS 精修器，常数对齐 Pinocchio 缩减模型。"""

    def __init__(self, consts: dict, dtype=torch.float64):
        super().__init__()
        self.dtype = dtype

        def buf(name, arr):
            self.register_buffer(name, torch.tensor(np.asarray(arr), dtype=dtype))

        for side in ("left", "right"):
            buf(f"{side}_T_base", consts[f"{side}_T_base"])   # (4,4)
            buf(f"{side}_place", consts[f"{side}_place"])     # (7,4,4)
            buf(f"{side}_axis", consts[f"{side}_axis"])       # (7,3)
            buf(f"{side}_T_ee", consts[f"{side}_T_ee"])       # (4,4)
        buf("q_lower", consts["q_lower"])   # (14,)
        buf("q_upper", consts["q_upper"])   # (14,)

    # ── 构造 ──────────────────────────────────────────────────────────
    @classmethod
    def from_pinocchio(cls, ik, dtype=torch.float64):
        """从 actibot_fk.Arm_IK 实例抽取常数（需要 Pinocchio，仅离线/校验时用）。"""
        import pinocchio as pin
        m, d = ik.model, ik.data
        q0 = np.zeros(m.nq)
        pin.forwardKinematics(m, d, q0)
        pin.computeJointJacobians(m, d, q0)

        def local_axis(jid):
            # 数值抽取关节局部旋转轴（与关节类型 RX/RY/RZ/RevoluteUnaligned 无关，保证一致）
            J = pin.getJointJacobian(m, d, jid, pin.ReferenceFrame.LOCAL)
            ang = J[3:6, m.joints[jid].idx_v]
            return ang / np.linalg.norm(ang)

        def arm_consts(joint_ids, frame_id):
            return (
                d.oMi[m.parents[joint_ids[0]]].homogeneous.copy(),          # T_base (q_body=0)
                np.stack([m.jointPlacements[j].homogeneous.copy() for j in joint_ids]),  # (7,4,4)
                np.stack([local_axis(j) for j in joint_ids]),               # (7,3)
                m.frames[frame_id].placement.homogeneous.copy(),            # T_ee
            )

        lb, lp, la, lee = arm_consts(_LEFT_JOINT_IDS, ik.left_gripper_id)
        rb, rp, ra, ree = arm_consts(_RIGHT_JOINT_IDS, ik.right_gripper_id)

        # 关节限位：缩减模型 idx_q 顺序 = [L_7, R_7]
        lo, hi = m.lowerPositionLimit, m.upperPositionLimit
        q_lower = np.concatenate([lo[5:12], lo[12:19]])
        q_upper = np.concatenate([hi[5:12], hi[12:19]])

        consts = dict(
            left_T_base=lb, left_place=lp, left_axis=la, left_T_ee=lee,
            right_T_base=rb, right_place=rp, right_axis=ra, right_T_ee=ree,
            q_lower=q_lower, q_upper=q_upper,
        )
        return cls(consts, dtype=dtype)

    def save_npz(self, path: str):
        np.savez(
            path,
            left_T_base=self.left_T_base.cpu().numpy(), left_place=self.left_place.cpu().numpy(),
            left_axis=self.left_axis.cpu().numpy(), left_T_ee=self.left_T_ee.cpu().numpy(),
            right_T_base=self.right_T_base.cpu().numpy(), right_place=self.right_place.cpu().numpy(),
            right_axis=self.right_axis.cpu().numpy(), right_T_ee=self.right_T_ee.cpu().numpy(),
            q_lower=self.q_lower.cpu().numpy(), q_upper=self.q_upper.cpu().numpy(),
        )

    @classmethod
    def from_npz(cls, path: str, dtype=torch.float64):
        """从缓存的 npz 加载常数（无需 Pinocchio，训练时用）。"""
        z = np.load(path)
        consts = {k: z[k] for k in z.files}
        return cls(consts, dtype=dtype)

    # ── FK / 雅可比 ──────────────────────────────────────────────────
    def _fk_arm(self, q7: torch.Tensor, side: str):
        """单臂 FK。q7 (N,7) → (T_ee (N,4,4), z (N,7,3) 世界轴, p (N,7,3) 关节原点)。"""
        T_base = getattr(self, f"{side}_T_base")
        place = getattr(self, f"{side}_place")
        axis = getattr(self, f"{side}_axis")
        T_ee = getattr(self, f"{side}_T_ee")

        N = q7.shape[0]
        T = T_base.expand(N, 4, 4).clone()
        zs, ps = [], []
        for i in range(7):
            T = T @ place[i]
            z_i = T[:, :3, :3] @ (axis[i] / torch.linalg.norm(axis[i]))  # (N,3)
            p_i = T[:, :3, 3]                                            # (N,3)
            Ti = torch.eye(4, dtype=self.dtype, device=q7.device).expand(N, 4, 4).clone()
            Ti[:, :3, :3] = _rodrigues_fixed_axis(axis[i], q7[:, i])
            T = T @ Ti
            zs.append(z_i)
            ps.append(p_i)
        T_world = T @ T_ee
        return T_world, torch.stack(zs, dim=1), torch.stack(ps, dim=1)

    def _jac_arm(self, z: torch.Tensor, p: torch.Tensor, p_ee: torch.Tensor):
        """LOCAL_WORLD_ALIGNED 6×7 雅可比。z,p (N,7,3)，p_ee (N,3) → (N,6,7)。"""
        lin = torch.cross(z, p_ee[:, None, :] - p, dim=-1)  # (N,7,3)
        J = torch.cat([lin, z], dim=-1)                     # (N,7,6)
        return J.transpose(1, 2)                            # (N,6,7)

    def fk_torch(self, q14: torch.Tensor):
        """双臂 FK，返回 (T_left (N,4,4), T_right (N,4,4))。"""
        Tl, _, _ = self._fk_arm(q14[:, :7], "left")
        Tr, _, _ = self._fk_arm(q14[:, 7:], "right")
        return Tl, Tr

    def ee_pose(self, q14: torch.Tensor) -> torch.Tensor:
        """便捷接口：返回 12D [eeL_xyz, eeL_rpy, eeR_xyz, eeR_rpy]（与 compute_ee_pose 对齐）。"""
        Tl, Tr = self.fk_torch(q14)
        return torch.cat([
            Tl[:, :3, 3], matrix_to_rpy(Tl[:, :3, :3]),
            Tr[:, :3, 3], matrix_to_rpy(Tr[:, :3, :3]),
        ], dim=-1)

    # ── DLS 精修 ──────────────────────────────────────────────────────
    def _refine_arm(self, q7, t_tgt, R_tgt, side, K, lam, dq_clip):
        I6 = torch.eye(6, dtype=self.dtype, device=q7.device)
        for _ in range(K):
            T, z, p = self._fk_arm(q7, side)
            R = T[:, :3, :3]
            t = T[:, :3, 3]
            e_p = t_tgt - t
            e_r = log3(R_tgt @ R.transpose(1, 2))
            e = torch.cat([e_p, e_r], dim=-1)                       # (N,6)
            J = self._jac_arm(z, p, t)                              # (N,6,7)
            JJt = J @ J.transpose(1, 2) + (lam ** 2) * I6           # (N,6,6)
            x = torch.linalg.solve(JJt, e[..., None])               # (N,6,1)
            dq = (J.transpose(1, 2) @ x).squeeze(-1)               # (N,7)
            dq = torch.clamp(dq, -dq_clip, dq_clip)
            q7 = q7 + dq
        return q7

    # ── 任务空间损失 / 误差（可微、无 Pinocchio）────────────────────────
    def _targets(self, ee_target12: torch.Tensor):
        return (ee_target12[:, :3], rpy_to_matrix(ee_target12[:, 3:6]),
                ee_target12[:, 6:9], rpy_to_matrix(ee_target12[:, 9:12]))

    def pose_residual(self, q14: torch.Tensor, ee_target12: torch.Tensor):
        """返回左右臂 6D 误差分量 (e_pL,e_rL,e_pR,e_rR)，各 (N,3)。"""
        Tl, Tr = self.fk_torch(q14)
        tL, RL, tR, RR = self._targets(ee_target12)
        e_pL = tL - Tl[:, :3, 3]
        e_rL = log3(RL @ Tl[:, :3, :3].transpose(1, 2))
        e_pR = tR - Tr[:, :3, 3]
        e_rR = log3(RR @ Tr[:, :3, :3].transpose(1, 2))
        return e_pL, e_rL, e_pR, e_rR

    def pose_loss(self, q14: torch.Tensor, ee_target12: torch.Tensor, w_ori: float = 0.1):
        """FK 任务损失：位置平方 + w_ori·姿态平方（双臂平均）。"""
        e_pL, e_rL, e_pR, e_rR = self.pose_residual(q14, ee_target12)
        pos = (e_pL.pow(2).sum(-1) + e_pR.pow(2).sum(-1)).mean()
        ori = (e_rL.pow(2).sum(-1) + e_rR.pow(2).sum(-1)).mean()
        return pos + w_ori * ori

    def pos_error(self, q14: torch.Tensor, ee_target12: torch.Tensor) -> torch.Tensor:
        """逐样本末端位置误差（米），双臂平均，(N,)。"""
        e_pL, _, e_pR, _ = self.pose_residual(q14, ee_target12)
        return 0.5 * (e_pL.norm(dim=-1) + e_pR.norm(dim=-1))

    def refine(self, q14: torch.Tensor, ee_target12: torch.Tensor,
               K: int, lam: float = 0.1, dq_clip: float = 0.05) -> torch.Tensor:
        """可微 DLS 精修，左右臂各跑固定 K 步。

        q14: (N,14) 种子关节角 [L_7, R_7]
        ee_target12: (N,12) 目标末端 [eeL_xyzrpy, eeR_xyzrpy]
        返回: (N,14) 精修后关节角；梯度可回传到 q14。
        """
        tL, RL = ee_target12[:, :3], rpy_to_matrix(ee_target12[:, 3:6])
        tR, RR = ee_target12[:, 6:9], rpy_to_matrix(ee_target12[:, 9:12])
        qL = self._refine_arm(q14[:, :7], tL, RL, "left", K, lam, dq_clip)
        qR = self._refine_arm(q14[:, 7:], tR, RR, "right", K, lam, dq_clip)
        return torch.cat([qL, qR], dim=-1)


def load_refiner(dtype=torch.float64, npz_path=None):
    """便捷加载：优先用缓存 npz，否则从 Pinocchio 抽取。"""
    if npz_path is None:
        npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refiner_consts.npz")
    if os.path.exists(npz_path):
        return DiffRefiner.from_npz(npz_path, dtype=dtype)
    from fk_utils import load_ik
    refiner = DiffRefiner.from_pinocchio(load_ik(), dtype=dtype)
    try:
        refiner.save_npz(npz_path)
    except OSError:
        pass
    return refiner
