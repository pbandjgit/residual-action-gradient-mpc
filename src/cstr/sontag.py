"""Auxiliary Sontag feedback for the LCNN-paper CSTR benchmark."""
from __future__ import annotations

import numpy as np


P_PAPER = np.array([[1060.0, 22.0], [22.0, 0.52]], dtype=float)


def saturated_sontag_np(
    sim,
    x: np.ndarray,
    input_lo: np.ndarray,
    input_hi: np.ndarray,
    P: np.ndarray = P_PAPER,
    eps: float = 1e-12,
) -> np.ndarray:
    """Universal Sontag-style feedback clipped to the admissible input box.

    The CSTR dynamics are control-affine in the shifted manipulated inputs:

        x_dot = f(x, 0) + g(x) u

    For V=x^T P x, define p=L_f V and q=L_g V. The unconstrained feedback is

        phi = - (p + sqrt(p^2 + ||q||^4)) / ||q||^2 * q

    and the final auxiliary controller is the componentwise saturated value.
    """
    x = np.asarray(x, dtype=float)
    input_lo = np.asarray(input_lo, dtype=float)
    input_hi = np.asarray(input_hi, dtype=float)
    grad_v = 2.0 * (P @ x)
    f0 = sim._dynamics(x, np.zeros(2, dtype=float))
    _, B = sim.analytic_jacobian(x, np.zeros(2, dtype=float))
    p = float(grad_v @ f0)
    q = grad_v @ B
    q_norm2 = float(q @ q)
    if q_norm2 <= eps:
        u = np.zeros(2, dtype=float)
    else:
        gain = (p + np.sqrt(p * p + q_norm2 * q_norm2)) / q_norm2
        u = -gain * q
    u = np.nan_to_num(u, nan=0.0, posinf=1e30, neginf=-1e30)
    return np.clip(u, input_lo, input_hi)


def saturated_sontag_casadi(
    ca,
    x,
    rhs_zero,
    B,
    input_lo,
    input_hi,
    P: np.ndarray = P_PAPER,
    eps: float = 1e-12,
):
    """CasADi expression for the same saturated Sontag feedback."""
    grad_v = 2.0 * ca.mtimes(P, x)
    p = ca.dot(grad_v, rhs_zero)
    q = ca.mtimes(grad_v.T, B).T
    q_norm2 = ca.dot(q, q)
    safe_norm2 = q_norm2 + eps
    gain = (p + ca.sqrt(p * p + q_norm2 * q_norm2)) / safe_norm2
    u_raw = -gain * q
    return ca.vertcat(
        ca.fmin(ca.fmax(u_raw[0], float(input_lo[0])), float(input_hi[0])),
        ca.fmin(ca.fmax(u_raw[1], float(input_lo[1])), float(input_hi[1])),
    )
