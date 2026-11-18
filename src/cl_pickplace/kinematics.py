"""Rotation helpers, Jacobians and the damped least squares IK step."""
from __future__ import annotations

import mujoco
import numpy as np


# ---------------------------------------------------------------- rotations
def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def wrap_half_pi(a: float) -> float:
    """Wrap to [-pi/2, pi/2): for symmetric objects whose yaw is defined mod pi."""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), float(angle))
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.asarray(a, float), np.asarray(b, float))
    return q


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, float).reshape(9))
    return q


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def axis_yaw_from_mat(mat: np.ndarray, axis: int = 0) -> float:
    """Yaw angle of a body axis projected onto the world xy plane."""
    v = mat[:, axis]
    return float(np.arctan2(v[1], v[0]))


def lookat_quat(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera orientation (MuJoCo cameras look along -z, y up) looking from pos to target."""
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:          # looking straight down: image up = world +x
        x = np.array([0.0, -1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return mat_to_quat(np.column_stack([x, y, z]))


def gripper_down_quat(yaw: float) -> np.ndarray:
    """EE orientation with hand z pointing down and hand x at world yaw `yaw`.

    Columns: x = (cos, sin, 0), z = (0, 0, -1), y = z cross x.
    """
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return mat_to_quat(np.column_stack([x, y, z]))


def closest_symmetric_yaw(target: float, current: float, period: float = np.pi) -> float:
    """Pick target + k*period closest to current (gripper is symmetric under 180 deg)."""
    return float(current + wrap_half_pi(target - current)) if period == np.pi else \
        float(current + wrap_pi(target - current))


def orientation_error(q_des: np.ndarray, mat_cur: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) that rotates the current orientation to q_des."""
    q_cur = mat_to_quat(mat_cur)
    err_local = np.zeros(3)
    mujoco.mju_subQuat(err_local, np.asarray(q_des, float), q_cur)
    return mat_cur @ err_local


def pose_error(p_des: np.ndarray, q_des: np.ndarray,
               p_cur: np.ndarray, mat_cur: np.ndarray) -> np.ndarray:
    return np.concatenate([p_des - p_cur, orientation_error(q_des, mat_cur)])


# ---------------------------------------------------------------- jacobians and IK
def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_id: int,
                  dof_idx: np.ndarray) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])


def dls_step(J: np.ndarray, v: np.ndarray, lam: float,
             q: np.ndarray | None = None, q_home: np.ndarray | None = None,
             k_null: float = 0.0) -> np.ndarray:
    """dq = J^T (J J^T + lam^2 I)^-1 v  +  (I - J_pinv J) k_null (q_home - q)."""
    JJt = J @ J.T + (lam ** 2) * np.eye(J.shape[0])
    J_dls = J.T @ np.linalg.solve(JJt, np.eye(J.shape[0]))
    dq = J_dls @ v
    if k_null > 0 and q is not None and q_home is not None:
        N = np.eye(J.shape[1]) - np.linalg.pinv(J) @ J
        dq = dq + N @ (k_null * (q_home - q))
    return dq


def clip_twist(v: np.ndarray, max_lin: float, max_ang: float) -> np.ndarray:
    v = v.copy()
    nl, na = np.linalg.norm(v[:3]), np.linalg.norm(v[3:])
    if nl > max_lin:
        v[:3] *= max_lin / nl
    if na > max_ang:
        v[3:] *= max_ang / na
    return v
