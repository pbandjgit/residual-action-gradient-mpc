"""CSTR (Continuously Stirred Tank Reactor) Digital Twin Simulator.

First-order irreversible exothermic reaction A → B with Arrhenius kinetics.
Parameters from Seborg, Edgar & Mellichamp (Process Dynamics and Control).

State:  x = [C_A (mol/L), T (K)]  — directly observable
Input:  u = [F (L/min), T_c (K)]
         F:   volumetric flow rate (dilution + residence time)
         T_c: coolant temperature  (heat removal)

ODE:
  dC_A/dt = (F/V)*(C_A0 - C_A) - k(T)*C_A
  dT/dt   = (F/V)*(T0 - T) + (-dHr/rho_Cp)*k(T)*C_A - (UA/(rho_Cp*V))*(T - T_c)
  k(T) = k0 * exp(-Ea_R / T)

Sobolev motivation:
  ∂C_A_{t+1}/∂F_t ≈ dt*(C_A0 - C_A)/V  →  varies 9× over operating range
  ∂T_{t+1}/∂F_t   ≈ dt*(T0 - T)/V      →  changes sign at T=T0
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import root


@dataclass
class CSTRParams:
    """Physical parameters of the CSTR (Seborg et al.)."""
    V:       float = 100.0    # L, reactor volume
    C_A0:    float = 1.0      # mol/L, feed concentration
    T0:      float = 350.0    # K, feed temperature
    k0:      float = 7.2e10   # 1/min, pre-exponential factor
    Ea_R:    float = 8750.0   # K, activation energy / R
    dHr:     float = -5e4     # J/mol, heat of reaction (negative = exothermic)
    rho_Cp:  float = 239.0    # J/(L·K), density × specific heat
    UA:      float = 5e4      # J/(min·K), heat transfer coefficient × area
    # Input bounds
    F_min:   float = 20.0     # L/min
    F_max:   float = 200.0    # L/min
    Tc_min:  float = 250.0    # K
    Tc_max:  float = 350.0    # K
    # State bounds (safety)
    CA_min:  float = 0.001    # mol/L
    CA_max:  float = 0.999    # mol/L
    T_min:   float = 280.0    # K
    T_max:   float = 500.0    # K


def create_cstr(
    dt: float = 0.1,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
    state_clipping: bool = True,
    input_clipping: bool = True,
    integration_substeps: int = 1,
) -> "CSTRSimulator":
    return CSTRSimulator(
        CSTRParams(),
        dt=dt,
        noise_std=noise_std,
        seed=seed,
        state_clipping=state_clipping,
        input_clipping=input_clipping,
        integration_substeps=integration_substeps,
    )


class CSTRSimulator:
    """CSTR simulator using RK4 integration.

    Observable state: [C_A (mol/L), T (K)]
    Input:            [F (L/min), T_c (K)]
    """

    def __init__(
        self,
        params: Optional[CSTRParams] = None,
        dt: float = 0.1,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
        state_clipping: bool = True,
        input_clipping: bool = True,
        integration_substeps: int = 1,
    ):
        self.params = params if params is not None else CSTRParams()
        self.dt = dt
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.state_clipping = bool(state_clipping)
        self.input_clipping = bool(input_clipping)
        self.integration_substeps = max(int(integration_substeps), 1)

    # ------------------------------------------------------------------
    # ODE
    # ------------------------------------------------------------------

    def _dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time ODE: dx/dt = f(x, u)."""
        p = self.params
        C_A, T = float(x[0]), float(x[1])
        F,   T_c = float(u[0]), float(u[1])
        T = max(T, 1.0)  # guard against log(0) in exp(-Ea_R/T)
        k = p.k0 * np.exp(-p.Ea_R / T)
        dCA = (F / p.V) * (p.C_A0 - C_A) - k * C_A
        dT  = ((F / p.V) * (p.T0 - T)
               + (-p.dHr / p.rho_Cp) * k * C_A
               - (p.UA / (p.rho_Cp * p.V)) * (T - T_c))
        return np.array([dCA, dT], dtype=np.float64)

    def clip_input(self, u: np.ndarray) -> np.ndarray:
        """Project inputs to actuator bounds."""
        p = self.params
        return np.array([
            np.clip(u[0], p.F_min, p.F_max),
            np.clip(u[1], p.Tc_min, p.Tc_max),
        ], dtype=np.float64)

    def clip_state(self, x: np.ndarray) -> np.ndarray:
        """Project states to safety bounds."""
        p = self.params
        y = np.asarray(x, dtype=np.float64).copy()
        y[0] = np.clip(y[0], p.CA_min, p.CA_max)
        y[1] = np.clip(y[1], p.T_min, p.T_max)
        return y

    def in_state_bounds(self, x: np.ndarray, margin: float | np.ndarray = 0.0) -> bool:
        """Return True if x is inside the safety bounds with optional margin."""
        p = self.params
        m = np.broadcast_to(np.asarray(margin, dtype=float), (2,))
        x = np.asarray(x, dtype=float)
        return bool(
            p.CA_min + m[0] <= x[0] <= p.CA_max - m[0]
            and p.T_min + m[1] <= x[1] <= p.T_max - m[1]
        )

    def _rk4_single(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """One RK4 micro-step of the physical ODE."""
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + 0.5 * dt * k1, u)
        k3 = self._dynamics(x + 0.5 * dt * k2, u)
        k4 = self._dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _rk4_raw(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One sample interval of physical RK4 without state clipping."""
        y = np.asarray(x, dtype=np.float64).copy()
        h = self.dt / self.integration_substeps
        for _ in range(self.integration_substeps):
            y = self._rk4_single(y, u, h)
        return y

    def _rk4(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One RK4 step using this simulator's configured clipping policy."""
        x_next = self._rk4_raw(x, u)
        if self.state_clipping:
            x_next = self.clip_state(x_next)
        return x_next

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        state: np.ndarray,   # [C_A, T]
        u: np.ndarray,       # [F, T_c]
    ) -> np.ndarray:
        """One RK4 step. Returns next [C_A, T]."""
        u_eff = self.clip_input(u) if self.input_clipping else np.asarray(u, dtype=np.float64)
        x_next = self._rk4(state.astype(np.float64), u_eff)
        if self.noise_std > 0.0:
            x_next += self.rng.normal(0, self.noise_std, size=2)
            if self.state_clipping:
                x_next = self.clip_state(x_next)
        return x_next.astype(np.float32)

    def step_raw(self, state: np.ndarray, u: np.ndarray, clip_input: bool = True) -> np.ndarray:
        """One physical RK4 step without state clipping."""
        u_eff = self.clip_input(u) if clip_input else np.asarray(u, dtype=np.float64)
        return self._rk4_raw(state.astype(np.float64), u_eff).astype(np.float32)

    def simulate(
        self,
        u_sequence: np.ndarray,   # (T, 2)
        x0: np.ndarray,           # (2,) [C_A, T]
    ) -> np.ndarray:
        """Simulate trajectory. Returns (T+1, 2)."""
        n = u_sequence.shape[0]
        traj = np.zeros((n + 1, 2), dtype=np.float32)
        traj[0] = x0.astype(np.float32)
        for t in range(n):
            traj[t + 1] = self.step(traj[t], u_sequence[t])
        return traj

    def compute_jacobian_du(
        self,
        state: np.ndarray,   # [C_A, T]
        u: np.ndarray,       # [F, T_c]
        delta_F: float = 1.0,
        delta_Tc: float = 1.0,
    ) -> np.ndarray:
        """Numerical Jacobian ∂[C_A, T]_{t+1} / ∂[F, T_c]_t.

        Returns (2, 2).  Varies strongly with operating point (Sobolev signal).
        """
        p = self.params
        u0 = np.array([
            np.clip(u[0], p.F_min, p.F_max),
            np.clip(u[1], p.Tc_min, p.Tc_max),
        ], dtype=np.float64)
        deltas = np.array([delta_F, delta_Tc], dtype=np.float64)
        J = np.zeros((2, 2), dtype=np.float32)
        for j in range(2):
            u_p = u0.copy(); u_m = u0.copy()
            u_p[j] = min(u_p[j] + deltas[j], (p.F_max  if j == 0 else p.Tc_max))
            u_m[j] = max(u_m[j] - deltas[j], (p.F_min  if j == 0 else p.Tc_min))
            x_p = self._rk4(state.astype(np.float64), u_p)
            x_m = self._rk4(state.astype(np.float64), u_m)
            J[:, j] = (x_p - x_m) / (2.0 * deltas[j])
        return J

    def get_equilibrium(
        self,
        F: float,
        T_c: float,
        n_steps: int = 5000,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Approximate steady state [C_A, T] by forward simulation."""
        if x0 is None:
            x0 = np.array([0.5, 350.0], dtype=np.float64)
        x = x0.copy()
        u = np.array([F, T_c])
        for _ in range(n_steps):
            x = self._rk4(x, u)
        return x.astype(np.float32)

    def solve_equilibrium(
        self,
        F: float,
        T_c: float,
        x0: Optional[np.ndarray] = None,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """Solve f(x,u)=0 for a steady state at fixed input."""
        if x0 is None:
            x0 = np.array([0.5, 350.0], dtype=np.float64)
        u = np.array([F, T_c], dtype=np.float64)
        sol = root(lambda x: self._dynamics(np.asarray(x, dtype=float), u), x0, tol=tol)
        if not sol.success:
            raise RuntimeError(f"CSTR equilibrium solve failed: {sol.message}")
        return sol.x.astype(np.float64)

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def input_dim(self) -> int:
        return 2
