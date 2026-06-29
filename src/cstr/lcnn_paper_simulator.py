"""LCNN paper CSTR benchmark simulator.

This implements the CSTR model used in Tan & Wu's LCNN-MPC paper, Section 6.1,
Eq. 54:

    dCA/dt = F/V (CA0 - CA) - k0 exp(-E/(R T)) CA^2
    dT/dt  = F/V (T0 - T) - dH/(rhoL Cp) k0 exp(-E/(R T)) CA^2
             + Q/(rhoL Cp V)

The paper states that the parameter values are the same as Wu et al. (2019b)
and gives the unstable steady state

    [CAs, Ts, CA0s, Qs] = [1.95 mol/dm^3, 402 K, 4 mol/dm^3, 0 kJ/h].

The defaults below are the standard Christofides/Wu second-order CSTR
parameters that reproduce this steady state to within roundoff/model rounding.
The public API uses shifted coordinates by default, matching the paper:

    x = [CA - CAs, T - Ts], u = [CA0 - CA0s, Q - Qs].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


@dataclass
class LCNNPaperCSTRParams:
    """Parameters for the LCNN paper's second-order CSTR benchmark."""

    F: float = 5.0          # dm^3/h
    V: float = 1.0          # dm^3
    # The commonly cited rounded values are T0=300 K and k0=8.46e6.
    # The calibrated defaults below make the paper's stated steady state
    # [CAs, Ts, CA0s, Qs] = [1.95, 402, 4, 0] exactly satisfy Eq. 54.
    T0: float = 299.94372294372295  # K
    k0: float = 8.467107268646015e6 # dm^3/(mol h)
    E: float = 5.0e4        # kJ/kmol
    R: float = 8.314        # kJ/(kmol K)
    dH: float = -1.15e4     # kJ/kmol
    rhoL: float = 1000.0    # kg/m^3
    Cp: float = 0.231       # kJ/(kg K)
    CAs: float = 1.95       # mol/dm^3
    Ts: float = 402.0       # K
    CA0s: float = 4.0       # mol/dm^3
    Qs: float = 0.0         # kJ/h


def create_lcnn_paper_cstr(
    dt_hr: float = 1e-3,
    integration_substeps: int = 100,
    shifted: bool = True,
    seed: Optional[int] = None,
) -> "LCNNPaperCSTRSimulator":
    return LCNNPaperCSTRSimulator(
        LCNNPaperCSTRParams(),
        dt_hr=dt_hr,
        integration_substeps=integration_substeps,
        shifted=shifted,
        seed=seed,
    )


class LCNNPaperCSTRSimulator:
    """Second-order CSTR benchmark from the LCNN paper."""

    def __init__(
        self,
        params: Optional[LCNNPaperCSTRParams] = None,
        dt_hr: float = 1e-3,
        integration_substeps: int = 100,
        shifted: bool = True,
        seed: Optional[int] = None,
    ):
        self.params = params if params is not None else LCNNPaperCSTRParams()
        self.dt_hr = float(dt_hr)
        self.integration_substeps = max(int(integration_substeps), 1)
        self.shifted = bool(shifted)
        self.rng = np.random.default_rng(seed)

    @property
    def xs_abs(self) -> np.ndarray:
        p = self.params
        return np.array([p.CAs, p.Ts], dtype=float)

    @property
    def us_abs(self) -> np.ndarray:
        p = self.params
        return np.array([p.CA0s, p.Qs], dtype=float)

    @property
    def xeq(self) -> np.ndarray:
        return np.zeros(2, dtype=float) if self.shifted else self.xs_abs

    @property
    def ueq(self) -> np.ndarray:
        return np.zeros(2, dtype=float) if self.shifted else self.us_abs

    def to_absolute_state(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x + self.xs_abs if self.shifted else x

    def to_absolute_input(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return u + self.us_abs if self.shifted else u

    def to_model_state(self, x_abs: np.ndarray) -> np.ndarray:
        x_abs = np.asarray(x_abs, dtype=float)
        return x_abs - self.xs_abs if self.shifted else x_abs

    def rhs_absolute(self, x_abs: np.ndarray, u_abs: np.ndarray) -> np.ndarray:
        p = self.params
        ca, temp = float(x_abs[0]), max(float(x_abs[1]), 1.0)
        ca0, q = float(u_abs[0]), float(u_abs[1])
        k = p.k0 * np.exp(-p.E / (p.R * temp))
        reaction = k * ca * ca
        dca = (p.F / p.V) * (ca0 - ca) - reaction
        dtemp = (
            (p.F / p.V) * (p.T0 - temp)
            - (p.dH / (p.rhoL * p.Cp)) * reaction
            + q / (p.rhoL * p.Cp * p.V)
        )
        return np.array([dca, dtemp], dtype=float)

    def _dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous dynamics in the configured coordinate system."""
        return self.rhs_absolute(self.to_absolute_state(x), self.to_absolute_input(u))

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

    def step_solve_ivp(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)

        def rhs(_t, y):
            return self._dynamics(np.asarray(y, dtype=float), u)

        sol = solve_ivp(
            rhs,
            (0.0, self.dt_hr),
            np.asarray(x, dtype=float),
            method="DOP853",
            rtol=1e-11,
            atol=[1e-12, 1e-9],
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        return sol.y[:, -1]

    def solve_equilibrium_abs(self, u_abs: Optional[np.ndarray] = None) -> np.ndarray:
        if u_abs is None:
            u_abs = self.us_abs
        sol = root(lambda x: self.rhs_absolute(x, u_abs), self.xs_abs)
        if not sol.success:
            raise RuntimeError(f"LCNN paper CSTR equilibrium solve failed: {sol.message}")
        return sol.x.astype(float)

    def analytic_jacobian_absolute(self, x_abs: np.ndarray, u_abs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.params
        ca, temp = float(x_abs[0]), max(float(x_abs[1]), 1.0)
        del u_abs
        k = p.k0 * np.exp(-p.E / (p.R * temp))
        dk_dtemp = k * p.E / (p.R * temp * temp)
        heat_gain = -p.dH / (p.rhoL * p.Cp)
        fv = p.F / p.V

        A = np.array([
            [-fv - 2.0 * k * ca, -dk_dtemp * ca * ca],
            [heat_gain * 2.0 * k * ca, -fv + heat_gain * dk_dtemp * ca * ca],
        ], dtype=float)
        B = np.array([
            [fv, 0.0],
            [0.0, 1.0 / (p.rhoL * p.Cp * p.V)],
        ], dtype=float)
        return A, B

    def analytic_jacobian(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.analytic_jacobian_absolute(self.to_absolute_state(x), self.to_absolute_input(u))
