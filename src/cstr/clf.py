"""Control Lyapunov Function (CLF) design for Lyapunov-based MPC of the CSTR.

Linearizes the (open-loop-unstable) operating point, designs an LQR stabilizing
feedback K, and solves a Lyapunov equation for P so that V(x)=(x-xeq)^T P (x-xeq)
is a CLF and Phi(x)=ueq-K(x-xeq) is a known stabilizing auxiliary controller.

Used by the LMPC stability constraint: the chosen control must decrease V at
least as much as Phi (Wu/Christofides Lyapunov-based MPC).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov
from typing import Tuple


def linearize(sim, xeq: np.ndarray, ueq: np.ndarray, eps: float = 1e-6
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous-time A=df/dx, B=df/du at (xeq, ueq) via central differences."""
    def jac(fun, z):
        n = len(z)
        J = np.zeros((2, n))
        for i in range(n):
            dz = np.zeros(n); dz[i] = eps
            J[:, i] = (fun(z + dz) - fun(z - dz)) / (2 * eps)
        return J
    A = jac(lambda x: sim._dynamics(x, ueq), xeq)
    B = jac(lambda u: sim._dynamics(xeq, u), ueq)
    return A, B


def design_clf(sim, xeq: np.ndarray, ueq: np.ndarray,
               Qlqr: np.ndarray = None, Rlqr: np.ndarray = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (P, K): CLF matrix P>0 and LQR gain K stabilizing the linearization.

    Raises if the closed loop A-BK is not Hurwitz (design failed).
    """
    if Qlqr is None:
        Qlqr = np.diag([100.0, 0.1])
    if Rlqr is None:
        Rlqr = np.diag([1e-3, 1e-2])
    A, B = linearize(sim, xeq, ueq)
    P_are = solve_continuous_are(A, B, Qlqr, Rlqr)
    K = np.linalg.inv(Rlqr) @ B.T @ P_are
    Acl = A - B @ K
    if np.any(np.linalg.eigvals(Acl).real >= 0):
        raise RuntimeError(f"A-BK not Hurwitz: eig={np.linalg.eigvals(Acl)}")
    P = solve_continuous_lyapunov(Acl.T, -np.eye(2))
    return P, K


def clf_value(x: np.ndarray, P: np.ndarray, xeq: np.ndarray) -> np.ndarray:
    """V(x) = (x-xeq)^T P (x-xeq). Accepts (2,) or (N,2)."""
    d = np.atleast_2d(x) - xeq
    return np.einsum("ni,ij,nj->n", d, P, d)


def aux_controller(x: np.ndarray, K: np.ndarray, xeq: np.ndarray, ueq: np.ndarray,
                   lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Phi(x) = clip(ueq - K (x-xeq), [lo, hi])."""
    return np.clip(ueq - K @ (np.asarray(x) - xeq), lo, hi)


def save_clf(path: str, P: np.ndarray, K: np.ndarray,
             xeq: np.ndarray, ueq: np.ndarray) -> None:
    np.savez(path, P=P, K=K, xeq=xeq, ueq=ueq)


def load_clf(path: str):
    d = np.load(path)
    return d["P"], d["K"], d["xeq"], d["ueq"]
