"""Two CSTRs in series benchmark (process-control second plant).

This extends the single second-order CSTR of the LCNN paper
(`lcnn_paper_simulator.py`, Tan & Wu) to the standard Christofides/Wu
"two CSTRs in series" configuration used throughout the machine-learning /
Lyapunov-based process-control literature (Wu et al., 2019; CLBF-RNN MPC, etc.).

Same A -> B exothermic second-order kinetics and the same physical parameters as
the single-CSTR benchmark, now in two coupled tanks:

  Reactor 1: fresh feed (flow F, conc CA10, temp T0), heat input Q1.
  Reactor 2: receives reactor-1 effluent (flow F) plus a fresh feed
             (flow F, conc CA20, temp T0), heat input Q2.

Absolute state  x_abs = [CA1, T1, CA2, T2]
Absolute input  u_abs = [CA10, Q1, CA20, Q2]

ODEs (kf(T) = k0 exp(-E/(R T)),  hg = -dH/(rhoL Cp),  fv = F/V,  qc = 1/(rhoL Cp V)):

  dCA1 = fv (CA10 - CA1) - kf(T1) CA1^2
  dT1  = fv (T0   - T1) + hg kf(T1) CA1^2 + qc Q1
  dCA2 = fv (CA1 + CA20 - 2 CA2) - kf(T2) CA2^2
  dT2  = fv (T1  + T0   - 2 T2)  + hg kf(T2) CA2^2 + qc Q2

The public API mirrors `lcnn_paper_simulator.py`: shifted coordinates by
default, `_dynamics`, `analytic_jacobian` (returns (A, B)), and `step` (RK4), so
the existing auxiliary controller / experiment machinery generalizes directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import root


@dataclass
class TwoCSTRParams:
    """Same kinetics/thermo parameters as the single-CSTR LCNN benchmark."""

    F: float = 5.0          # dm^3/h, per-feed volumetric flow
    V: float = 1.0          # dm^3, per-reactor volume
    T0: float = 299.94372294372295  # K, feed temperature
    k0: float = 8.467107268646015e6  # dm^3/(mol h)
    E: float = 5.0e4        # kJ/kmol
    R: float = 8.314        # kJ/(kmol K)
    dH: float = -1.15e4     # kJ/kmol
    rhoL: float = 1000.0    # kg/m^3
    Cp: float = 0.231       # kJ/(kg K)
    # Nominal (steady-state) manipulated inputs, same operating point as single CSTR
    CA10s: float = 4.0      # mol/dm^3
    Q1s: float = 0.0        # kJ/h
    CA20s: float = 4.0      # mol/dm^3
    Q2s: float = 0.0        # kJ/h


class TwoCSTRSeriesSimulator:
    """Two second-order CSTRs in series (4 states, 4 inputs)."""

    def __init__(
        self,
        params: Optional[TwoCSTRParams] = None,
        dt_hr: float = 1e-3,
        integration_substeps: int = 100,
        shifted: bool = True,
        seed: Optional[int] = None,
    ):
        self.params = params if params is not None else TwoCSTRParams()
        self.dt_hr = float(dt_hr)
        self.integration_substeps = max(int(integration_substeps), 1)
        self.shifted = bool(shifted)
        self.rng = np.random.default_rng(seed)
        # Steady state solved once at construction.
        self._xs_abs = self._solve_steady_state()

    # ------------------------------------------------------------------ #
    # Coordinates
    # ------------------------------------------------------------------ #
    @property
    def us_abs(self) -> np.ndarray:
        p = self.params
        return np.array([p.CA10s, p.Q1s, p.CA20s, p.Q2s], dtype=float)

    @property
    def xs_abs(self) -> np.ndarray:
        return self._xs_abs.copy()

    @property
    def xeq(self) -> np.ndarray:
        return np.zeros(4, dtype=float) if self.shifted else self.xs_abs

    @property
    def ueq(self) -> np.ndarray:
        return np.zeros(4, dtype=float) if self.shifted else self.us_abs

    def to_absolute_state(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x + self.xs_abs if self.shifted else x

    def to_absolute_input(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return u + self.us_abs if self.shifted else u

    def to_model_state(self, x_abs: np.ndarray) -> np.ndarray:
        x_abs = np.asarray(x_abs, dtype=float)
        return x_abs - self.xs_abs if self.shifted else x_abs

    # ------------------------------------------------------------------ #
    # Dynamics
    # ------------------------------------------------------------------ #
    def rhs_absolute(self, x_abs: np.ndarray, u_abs: np.ndarray) -> np.ndarray:
        p = self.params
        ca1, t1 = float(x_abs[0]), max(float(x_abs[1]), 1.0)
        ca2, t2 = float(x_abs[2]), max(float(x_abs[3]), 1.0)
        ca10, q1, ca20, q2 = (float(u_abs[0]), float(u_abs[1]),
                              float(u_abs[2]), float(u_abs[3]))
        fv = p.F / p.V
        hg = -p.dH / (p.rhoL * p.Cp)
        qc = 1.0 / (p.rhoL * p.Cp * p.V)
        k1 = p.k0 * np.exp(-p.E / (p.R * t1))
        k2 = p.k0 * np.exp(-p.E / (p.R * t2))
        r1 = k1 * ca1 * ca1
        r2 = k2 * ca2 * ca2
        dca1 = fv * (ca10 - ca1) - r1
        dt1 = fv * (p.T0 - t1) + hg * r1 + qc * q1
        dca2 = fv * (ca1 + ca20 - 2.0 * ca2) - r2
        dt2 = fv * (t1 + p.T0 - 2.0 * t2) + hg * r2 + qc * q2
        return np.array([dca1, dt1, dca2, dt2], dtype=float)

    def _dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.rhs_absolute(self.to_absolute_state(x), self.to_absolute_input(u))

    def analytic_jacobian_absolute(self, x_abs: np.ndarray, u_abs: np.ndarray):
        p = self.params
        ca1, t1 = float(x_abs[0]), max(float(x_abs[1]), 1.0)
        ca2, t2 = float(x_abs[2]), max(float(x_abs[3]), 1.0)
        del u_abs
        fv = p.F / p.V
        hg = -p.dH / (p.rhoL * p.Cp)
        qc = 1.0 / (p.rhoL * p.Cp * p.V)
        k1 = p.k0 * np.exp(-p.E / (p.R * t1)); dk1 = k1 * p.E / (p.R * t1 * t1)
        k2 = p.k0 * np.exp(-p.E / (p.R * t2)); dk2 = k2 * p.E / (p.R * t2 * t2)
        A = np.array([
            [-fv - 2.0 * k1 * ca1, -dk1 * ca1 * ca1, 0.0, 0.0],
            [hg * 2.0 * k1 * ca1, -fv + hg * dk1 * ca1 * ca1, 0.0, 0.0],
            [fv, 0.0, -2.0 * fv - 2.0 * k2 * ca2, -dk2 * ca2 * ca2],
            [0.0, fv, hg * 2.0 * k2 * ca2, -2.0 * fv + hg * dk2 * ca2 * ca2],
        ], dtype=float)
        B = np.array([
            [fv, 0.0, 0.0, 0.0],
            [0.0, qc, 0.0, 0.0],
            [0.0, 0.0, fv, 0.0],
            [0.0, 0.0, 0.0, qc],
        ], dtype=float)
        return A, B

    def analytic_jacobian(self, x: np.ndarray, u: np.ndarray):
        return self.analytic_jacobian_absolute(self.to_absolute_state(x),
                                                self.to_absolute_input(u))

    # ------------------------------------------------------------------ #
    # Integration / steady state
    # ------------------------------------------------------------------ #
    def _rk4_single(self, x: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + 0.5 * h * k1, u)
        k3 = self._dynamics(x + 0.5 * h * k2, u)
        k4 = self._dynamics(x + h * k3, u)
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=float).copy()
        h = self.dt_hr / self.integration_substeps
        for _ in range(self.integration_substeps):
            y = self._rk4_single(y, u, h)
        return y.astype(np.float64)

    def _solve_steady_state(self) -> np.ndarray:
        """Solve f(x_abs, u_s)=0 near the single-CSTR operating point."""
        p = self.params
        u_s = np.array([p.CA10s, p.Q1s, p.CA20s, p.Q2s], dtype=float)
        # warm start: reactor 1 at the single-CSTR unstable SS, reactor 2 likewise
        x0 = np.array([1.95, 402.0, 1.95, 402.0], dtype=float)
        sol = root(lambda x: self.rhs_absolute(x, u_s), x0, tol=1e-12)
        if not sol.success:
            raise RuntimeError(f"two-CSTR steady-state solve failed: {sol.message}")
        return sol.x.astype(float)

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def input_dim(self) -> int:
        return 4
