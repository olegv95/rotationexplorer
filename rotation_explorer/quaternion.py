"""
Unit quaternion operations.

Convention: q = [w, x, y, z]  (scalar first, same as scipy).
"""

import numpy as np

D2R = np.pi / 180
R2D = 180 / np.pi


def normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def from_axis_angle(axis, angle_deg: float) -> np.ndarray:
    """Build unit quaternion [w, x, y, z] from axis vector and angle (degrees)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle_deg * D2R / 2
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


def from_euler_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Build unit quaternion from ZYX Euler angles (degrees)."""
    y, p, r = yaw * D2R / 2, pitch * D2R / 2, roll * D2R / 2
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    return normalize(np.array([
        cy*cp*cr + sy*sp*sr,
        cy*cp*sr - sy*sp*cr,
        cy*sp*cr + sy*cp*sr,
        sy*cp*cr - cy*sp*sr,
    ]))


def to_euler_zyx_old(q: np.ndarray) -> tuple[float, float, float]:
    """
    Extract ZYX Euler angles (degrees) from a unit quaternion.
    Matches the formula used in the original rotation_explorer.html.
    Returns (yaw, pitch, roll).
    """
    w, x, y, z = normalize(q)
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    sinp  = 2*(w*y - z*x)
    pitch = np.sign(sinp) * np.pi / 2 if abs(sinp) >= 1 else np.arcsin(sinp)
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return yaw * R2D, pitch * R2D, roll * R2D

def to_euler_zyx(q: np.ndarray) -> tuple[float, float, float]:
    """
    Extract ZYX Euler angles (degrees) from a unit quaternion.
    Returns (yaw, pitch, roll) where:
    Yaw   -> Z rotation
    Pitch -> Y rotation
    Roll  -> X rotation
    """
    # Ensure normalization to avoid domain errors in arcsin
    w, x, y, z = q / np.linalg.norm(q)
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Clamp sinp to [-1, 1] to avoid NaNs due to precision issues
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2) # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Convert to degrees (assuming R2D = 180 / np.pi)
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

def axis_angle(q: np.ndarray) -> tuple[np.ndarray | None, float]:
    """
    Decompose quaternion into (axis, angle_deg).
    Returns (None, 0.0) for the identity rotation.
    """
    w, x, y, z = normalize(q)
    
    # Flip sign if w is negative to always take the shortest arc
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
        
    theta = 2 * np.arccos(min(1.0, w))
    sa = np.sin(theta / 2)
    if sa > 1e-3:
        return np.array([x / sa, y / sa, z / sa]), theta * R2D
    return None, 0.0


def hamilton_product(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton (quaternion) product q1 ⊗ q2.  Non-commutative."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def conjugate(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def log(q: np.ndarray) -> np.ndarray:
    """Quaternion logarithm (returns pure quaternion)."""
    w, xyz = q[0], q[1:]
    v_norm = np.linalg.norm(xyz)
    if v_norm < 1e-10:
        return np.zeros(4)
    theta = np.arctan2(v_norm, w)
    return np.concatenate([[0.0], xyz / v_norm * theta])


def exp(q: np.ndarray) -> np.ndarray:
    """Quaternion exponential (expects pure quaternion input)."""
    v = q[1:]
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[np.cos(v_norm)], v / v_norm * np.sin(v_norm)])


def kinematic_matrix(q: np.ndarray) -> np.ndarray:
    """4×3 matrix Ξ(q) such that q̇ = ½ Ξ(q) · ω_body."""
    w, x, y, z = q
    return 0.5 * np.array([
        [-x, -y, -z],
        [ w, -z,  y],
        [ z,  w, -x],
        [-y,  x,  w],
    ])


def propagate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """First-order Euler integration of quaternion kinematics.  omega in rad/s."""
    qdot = kinematic_matrix(q) @ omega
    return normalize(q + qdot * dt)
