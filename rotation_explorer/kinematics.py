"""
Attitude kinematics: Euler angle rate equations, gimbal lock detection,
and numerical propagation helpers.
"""

import numpy as np
from .dcm import rate_equation, reorthogonalize
from .quaternion import propagate as quat_propagate, normalize as quat_normalize

D2R = np.pi / 180
R2D = 180 / np.pi


class GimbalLockError(ValueError):
    pass


def euler_rates(
    phi: float, theta: float,
    p: float, q: float, r: float,
) -> tuple[float, float, float]:
    """
    ZYX Euler angle rates from body-axis angular rates.

    Args:
        phi, theta: current roll and pitch angles (degrees)
        p, q, r:    body-axis roll, pitch, yaw rates (rad/s)

    Returns:
        (phi_dot, theta_dot, psi_dot) in deg/s

    Raises GimbalLockError when |theta| ≈ 90° (cosθ ≈ 0).
    """
    phi_r, theta_r = phi * D2R, theta * D2R
    sp, cp = np.sin(phi_r), np.cos(phi_r)
    ct = np.cos(theta_r)

    if abs(ct) < 1e-6:
        raise GimbalLockError(
            f"Gimbal lock: pitch ≈ ±90°, psi_dot diverges (cosθ = {ct:.2e})"
        )

    tt = np.tan(theta_r)
    phi_dot   = p + (q * sp + r * cp) * tt
    theta_dot = q * cp - r * sp
    psi_dot   = (q * sp + r * cp) / ct
    return phi_dot * R2D, theta_dot * R2D, psi_dot * R2D


def gimbal_lock_proximity(theta_deg: float) -> float:
    """Returns fraction 0→1 of how close pitch is to the ±90° singularity."""
    return min(1.0, abs(theta_deg) / 90.0)


def integrate_dcm(
    R: np.ndarray,
    omega: np.ndarray,
    dt: float,
    reorth_every: int = 100,
    step: int = 0,
) -> np.ndarray:
    """
    Propagate DCM one step: R_{n+1} = R_n + Ṙ·dt.
    Re-orthogonalises every `reorth_every` steps to bound numerical drift.
    """
    R_next = R + rate_equation(R, omega) * dt
    if (step + 1) % reorth_every == 0:
        R_next = reorthogonalize(R_next)
    return R_next


def integrate_quaternion(
    q: np.ndarray,
    omega: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Propagate unit quaternion one step (renormalised first-order Euler)."""
    return quat_propagate(q, omega, dt)
