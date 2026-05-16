"""Direction Cosine Matrix (DCM) operations — ZYX aerospace 3-2-1 convention."""

import numpy as np

D2R = np.pi / 180
R2D = 180 / np.pi


def euler_to_dcm(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Build body→inertial DCM from ZYX Euler angles (degrees).

    R = Rz(yaw) · Ry(pitch) · Rx(roll)
    """
    y, p, r = yaw * D2R, pitch * D2R, roll * D2R
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,              cp*cr            ],
    ])


def dcm_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """
    Extract ZYX Euler angles (degrees) from a DCM.
    Returns (yaw, pitch, roll). Singular at pitch = ±90°.
    """
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) < 0.9999:
        yaw  = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw  = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    return yaw * R2D, pitch * R2D, roll * R2D


def determinant(R: np.ndarray) -> float:
    return float(np.linalg.det(R))


def is_orthogonal(R: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.allclose(R @ R.T, np.eye(3), atol=tol))


def skew_symmetric(omega: np.ndarray) -> np.ndarray:
    """Cross-product (skew-symmetric) matrix [ω]×."""
    p, q, r = omega
    return np.array([
        [ 0, -r,  q],
        [ r,  0, -p],
        [-q,  p,  0],
    ])


def rate_equation(R: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Poisson's kinematic equation: Ṙ = R · [ω]×  (omega in rad/s)."""
    return R @ skew_symmetric(omega)


def reorthogonalize(R: np.ndarray) -> np.ndarray:
    """Project R back onto SO(3) via SVD. Use every ~1000 integration steps."""
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt
