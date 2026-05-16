"""
Demonstrates every module in rotation_explorer.
Run:  python demo.py
"""

import numpy as np
from rotation_explorer import dcm, quaternion as Q, kinematics as K


def section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


# ── 1. Direction Cosine Matrix ───────────────────────────────
def demo_dcm() -> None:
    section("1. Direction Cosine Matrix  (ZYX, body->inertial)")

    yaw, pitch, roll = 30.0, 20.0, 10.0
    R = dcm.euler_to_dcm(yaw, pitch, roll)
    print(f"Euler  yaw={yaw} deg  pitch={pitch} deg  roll={roll} deg")
    print("DCM R =")
    for row in R:
        print("  ", "  ".join(f"{v:+.4f}" for v in row))
    print(f"det(R)  = {dcm.determinant(R):.6f}  (expect +1)")
    print(f"R*R^T~I : {dcm.is_orthogonal(R)}")

    yaw2, pitch2, roll2 = dcm.dcm_to_euler(R)
    print(f"Round-trip Euler: yaw={yaw2:.2f} deg  pitch={pitch2:.2f} deg  roll={roll2:.2f} deg")

    omega = np.array([0.1, 0.2, 0.3])   # body rates rad/s
    Rdot = dcm.rate_equation(R, omega)
    print(f"Rdot (first row): {Rdot[0].round(4)}")


# ── 2. Quaternions ───────────────────────────────────────────
def demo_quaternion() -> None:
    section("2. Quaternion Operations")

    q = Q.from_axis_angle([0, 1, 0], 90)   # yaw 90 deg
    print(f"Yaw 90 deg -> q = {q.round(4)}")
    yaw, pitch, roll = Q.to_euler_zyx(q)
    print(f"Back to Euler:  yaw={yaw:.1f} deg  pitch={pitch:.1f} deg  roll={roll:.1f} deg")
    axis, angle = Q.axis_angle(q)
    print(f"Axis-angle:     axis={np.round(axis,3)}  angle={angle:.1f} deg")

    print()
    q1 = Q.from_axis_angle([0, 1, 0], 45)
    q2 = Q.from_axis_angle([0, 1, 0], 45)
    q_composed = Q.hamilton_product(q1, q2)
    yaw_c, _, _ = Q.to_euler_zyx(q_composed)
    print(f"q(45) x q(45) -> yaw = {yaw_c:.1f} deg  (expect 90)")

    print()
    q = Q.from_euler_zyx(30, 20, 10)
    print(f"from_euler_zyx(30,20,10) = {q.round(4)}")

    print()
    omega = np.array([0.0, 0.5, 0.0])
    q_next = Q.propagate(q, omega, dt=0.1)
    print(f"After 0.1 s propagation with w=[0,0.5,0]: |q|={np.linalg.norm(q_next):.6f}")


# ── 3. Kinematics & Gimbal Lock ──────────────────────────────
def demo_kinematics() -> None:
    section("3. Kinematics & Gimbal Lock")

    p, q_rate, r = 0.1, 0.2, 0.3   # rad/s

    print("Euler rates at various pitch angles (p=0.1, q=0.2, r=0.3 rad/s):\n")
    print(f"  {'pitch':>7}  {'proximity':>10}  {'phi_dot':>10}  {'theta_dot':>10}  {'psi_dot':>10}")
    for theta_deg in [0, 30, 60, 80, 85, 89]:
        prox = K.gimbal_lock_proximity(theta_deg)
        try:
            fd, td, pd = K.euler_rates(0.0, theta_deg, p, q_rate, r)
            print(f"  {theta_deg:>6} deg  {prox:>10.1%}  {fd:>10.2f}  {td:>10.2f}  {pd:>10.2f}")
        except K.GimbalLockError:
            print(f"  {theta_deg:>6} deg  {prox:>10.1%}  {'GIMBAL LOCK':>32}")

    print()
    omega = np.array([p, q_rate, r])
    R = np.eye(3)
    for step in range(500):
        R = K.integrate_dcm(R, omega, dt=0.01, reorth_every=100, step=step)
    print(f"DCM after 500 steps (dt=0.01 s): det={dcm.determinant(R):.6f}  orthogonal={dcm.is_orthogonal(R)}")


if __name__ == "__main__":
    demo_dcm()
    demo_quaternion()
    demo_kinematics()
