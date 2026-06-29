"""One-step state-space surrogate models for CSTR (LCNN-faithful recipe).

Rationale (see CSTR_FINDINGS_AND_PLAN.md): for the open-loop-unstable CSTR,
a one-step Markovian map F̃(x,u) -> x_next trained on direct (x,u)-pair samples
over a bounded region, rolled out over a SHORT MPC horizon, regulates the
unstable equilibrium at oracle level — whereas a multi-step GRU rollout on
trajectory data with a long horizon fails (rollout error amplifies ~L^H near
the unstable point).

Models:
  OneStepFNN   — plain MLP. Most accurate on clean data.
  OneStepLCNN  — Lipschitz-Constrained NN (Björck-orthonormal SpectralDense +
                 GroupSort), faithful to Tan & Wu (arXiv:2308.13721). Its value
                 is robustness to TRAINING-DATA NOISE; on clean data the
                 Lipschitz bound costs accuracy, so prefer OneStepFNN unless
                 studying noise robustness.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict


class GroupSort(nn.Module):
    """GroupSort activation (Anil et al. 2019). group_size=2 == MaxMin.

    Gradient-norm preserving and 1-Lipschitz; required for SpectralDense layers
    to retain expressivity under the spectral-norm constraint.
    """

    def __init__(self, group_size: int = 2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F = x.shape
        assert F % self.group_size == 0, "feature dim must divide group_size"
        x = x.view(B, F // self.group_size, self.group_size)
        # Paper convention for group_size=2 is [max(x1,x2), min(x1,x2)].
        return x.sort(dim=-1, descending=True)[0].reshape(B, F)


def bjorck_orthonormal(W: torch.Tensor, iters: int = 15) -> torch.Tensor:
    """Björck orthonormalization: drive all singular values of W toward 1.

    Faithful SpectralDense (deel-lip is TensorFlow-only). For a tall matrix this
    makes the columns orthonormal (W^T W = I); transposed for wide matrices.
    """
    transpose = W.shape[0] < W.shape[1]
    A = W.t() if transpose else W
    A = A / (torch.linalg.matrix_norm(A, 2) + 1e-9)
    for _ in range(iters):
        A = 1.5 * A - 0.5 * A @ (A.t() @ A)
    return A.t() if transpose else A


class BjorckLinear(nn.Module):
    """Linear layer whose effective weight is Björck-orthonormalized (σ -> 1),
    optionally scaled to a target Lipschitz bound."""

    def __init__(self, in_features: int, out_features: int, scale: float = 1.0,
                 iters: int = 15):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = scale
        self.iters = iters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = bjorck_orthonormal(self.weight, self.iters)
        return self.scale * (x @ W.t()) + self.bias

    def materialized_linear(self) -> nn.Linear:
        """Return an equivalent fixed ``nn.Linear`` layer for inference.

        During training, Björck orthonormalization is useful because it keeps the
        effective spectral norm constrained. During MPC evaluation, recomputing
        Björck iterations inside every L4CasADi/IPOPT call is unnecessarily
        expensive. This freezes the currently effective weight.
        """
        with torch.no_grad():
            W = self.scale * bjorck_orthonormal(self.weight, self.iters)
            layer = nn.Linear(self.weight.shape[1], self.weight.shape[0])
            layer.weight.copy_(W)
            layer.bias.copy_(self.bias)
            return layer


class MaxNormLinear(nn.Module):
    """Linear layer with a row-wise max-norm constraint.

    The LCNN paper uses SpectralDense hidden layers followed by a final dense
    linear layer with a weight constraint. PyTorch does not apply constraints
    automatically after optimizer steps, so ``apply_constraint()`` must be called
    by the training loop.
    """

    def __init__(self, in_features: int, out_features: int, max_norm: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.max_norm = float(max_norm)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias

    def apply_constraint(self) -> None:
        with torch.no_grad():
            norms = self.linear.weight.norm(p=2, dim=1, keepdim=True)
            scale = torch.clamp(self.max_norm / (norms + 1e-12), max=1.0)
            self.linear.weight.mul_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def materialized_linear(self) -> nn.Linear:
        with torch.no_grad():
            layer = nn.Linear(self.linear.weight.shape[1], self.linear.weight.shape[0])
            layer.weight.copy_(self.linear.weight)
            layer.bias.copy_(self.linear.bias)
            return layer


class OneStepFNN(nn.Module):
    """Plain MLP one-step map: (x, u) -> x_next, all normalized."""

    def __init__(self, state_dim: int = 2, input_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, xu: torch.Tensor) -> torch.Tensor:
        return self.net(xu)


class OneStepLCNN(nn.Module):
    """Faithful LCNN one-step map: SpectralDense(σ=1, Björck) + GroupSort, with a
    Lipschitz-bounded final layer. ``lipschitz_bound`` must exceed the target
    map's Lipschitz constant (cf. Remark 6 of arXiv:2308.13721)."""

    def __init__(self, state_dim: int = 2, input_dim: int = 2, hidden: int = 40,
                 lipschitz_bound: float = 3.0, n_layers: int = 2):
        super().__init__()
        assert hidden % 2 == 0, "hidden must be even for GroupSort(2)"
        self.state_dim = state_dim
        self.lipschitz_bound = lipschitz_bound
        layers = [BjorckLinear(state_dim + input_dim, hidden), GroupSort(2)]
        for _ in range(n_layers - 1):
            layers += [BjorckLinear(hidden, hidden), GroupSort(2)]
        self.body = nn.Sequential(*layers)
        self.out = MaxNormLinear(hidden, state_dim, max_norm=lipschitz_bound)

    def forward(self, xu: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(xu))

    def apply_constraints(self) -> None:
        self.out.apply_constraint()

    def lipschitz_upper_bound(self) -> float:
        with torch.no_grad():
            return float(torch.linalg.matrix_norm(self.out.weight, 2).detach())


class OneStepSmoothSpectral(nn.Module):
    """Smooth spectral one-step map: identical to ``OneStepLCNN`` (Bjorck
    SpectralDense σ=1 layers + Lipschitz-bounded MaxNorm output) but with a
    SMOOTH 1-Lipschitz activation (tanh by default) in place of GroupSort.

    This isolates a single variable for the Paper-H probe: GroupSort is
    1-Lipschitz but piecewise linear (its Jacobian jumps at sorting boundaries),
    whereas tanh is 1-Lipschitz and C-infinity. Comparing this model against
    ``OneStepLCNN`` under the same training keeps the first-order Lipschitz
    machinery fixed and changes only the smoothness of the action-gradient.

    Note: tanh is gradient-norm shrinking (|tanh'|<=1), unlike norm-preserving
    GroupSort, so this model may be less expressive under the σ=1 constraint.
    That trade-off (smoothness vs expressivity) is part of what the probe tests.
    """

    def __init__(self, state_dim: int = 2, input_dim: int = 2, hidden: int = 40,
                 lipschitz_bound: float = 3.0, n_layers: int = 2,
                 activation: str = "tanh"):
        super().__init__()
        self.state_dim = state_dim
        self.lipschitz_bound = lipschitz_bound
        acts = {"tanh": nn.Tanh, "softsign": nn.Softsign, "sigmoid": nn.Sigmoid}
        if activation not in acts:
            raise ValueError(f"unknown smooth activation: {activation}")
        act = acts[activation]
        layers = [BjorckLinear(state_dim + input_dim, hidden), act()]
        for _ in range(n_layers - 1):
            layers += [BjorckLinear(hidden, hidden), act()]
        self.body = nn.Sequential(*layers)
        self.out = MaxNormLinear(hidden, state_dim, max_norm=lipschitz_bound)

    def forward(self, xu: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(xu))

    def apply_constraints(self) -> None:
        self.out.apply_constraint()

    def lipschitz_upper_bound(self) -> float:
        with torch.no_grad():
            return float(torch.linalg.matrix_norm(self.out.weight, 2).detach())


class FrozenOneStepLCNN(nn.Module):
    """Inference-only LCNN with Björck layers materialized as fixed Linear layers."""

    def __init__(self, body: nn.Sequential, out: nn.Linear, lipschitz_bound: float):
        super().__init__()
        self.body = body
        self.out = out
        self.lipschitz_bound = lipschitz_bound

    def forward(self, xu: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(xu))

    def lipschitz_upper_bound(self) -> float:
        with torch.no_grad():
            return float(torch.linalg.matrix_norm(self.out.weight, 2).detach())


def materialize_lcnn_for_inference(model: nn.Module) -> nn.Module:
    """Freeze a trained ``OneStepLCNN`` into an equivalent faster inference model.

    Non-LCNN models are returned unchanged. The returned LCNN has no Björck
    iterations in ``forward()``, which is important for large hidden sizes and
    for L4CasADi tracing.
    """
    if not isinstance(model, OneStepLCNN):
        return model
    layers = []
    for layer in model.body:
        if isinstance(layer, BjorckLinear):
            layers.append(layer.materialized_linear())
        elif isinstance(layer, GroupSort):
            layers.append(GroupSort(layer.group_size))
        else:
            raise TypeError(f"Unsupported LCNN layer for materialization: {type(layer)!r}")
    frozen = FrozenOneStepLCNN(
        nn.Sequential(*layers),
        model.out.materialized_linear(),
        model.lipschitz_bound,
    )
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    return frozen


class SNSLinear(nn.Module):
    """Smooth-Neural-Surrogate layer (Moore et al. arXiv:2601.12169, extending
    Liu et al.). Per-row L1 weight normalization to a LEARNED layer Lipschitz
    constant c_l = exp(theta), bounding the layer's infinity-norm Lipschitz:

        W_hat_ij = min(1, c_l / sum_k|W_ik|) * W_ij,   c_l = exp(theta).

    The exponential parameterization (vs softplus) spreads the gradient evenly
    across layers and avoids optimization collapse under a tight smoothness
    budget — the failure mode of fixed spectral-norm (σ=1) LCNN here.
    """

    def __init__(self, in_features: int, out_features: int, init_c: float = 2.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.theta_c = nn.Parameter(torch.tensor(float(np.log(init_c))))

    def lipschitz(self) -> torch.Tensor:
        return torch.exp(self.theta_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.exp(self.theta_c)
        rowsum = self.weight.abs().sum(dim=1, keepdim=True)
        scale = torch.clamp(c / (rowsum + 1e-8), max=1.0)
        return x @ (scale * self.weight).t() + self.bias


class OneStepSNS(nn.Module):
    """SNS-MLP one-step map with learned layer-wise Lipschitz constants and a
    soft smoothness budget on the Lipschitz product C = prod_l c_l. Uses ReLU
    (1-Lipschitz) between SNS layers. Call ``smoothness_penalty()`` and add it
    (scaled by lambda) to the training loss."""

    def __init__(self, state_dim: int = 2, input_dim: int = 2, hidden: int = 64,
                 n_layers: int = 2, budget: float = 8.0, init_c: float = 2.0):
        super().__init__()
        self.state_dim = state_dim
        self.budget = budget
        self.layers = nn.ModuleList([SNSLinear(state_dim + input_dim, hidden, init_c)])
        for _ in range(n_layers - 1):
            self.layers.append(SNSLinear(hidden, hidden, init_c))
        self.out = SNSLinear(hidden, state_dim, init_c)

    def _layer_consts(self):
        return [l.lipschitz() for l in self.layers] + [self.out.lipschitz()]

    def lipschitz_product(self) -> torch.Tensor:
        C = self.out.lipschitz()
        for l in self.layers:
            C = C * l.lipschitz()
        return C

    def jacobian_lipschitz_bound(self) -> torch.Tensor:
        """Second-order (curvature) bound d_MLP <~ C * S, with
        S = sum_l c_l * prod_{j<l} c_j  (Moore et al. arXiv:2601.12169)."""
        cs = self._layer_consts()
        C = torch.prod(torch.stack(cs))
        S = torch.zeros((), dtype=cs[0].dtype)
        pref = torch.ones((), dtype=cs[0].dtype)
        for c in cs:
            S = S + c * pref
            pref = pref * c
        return C * S

    def smoothness_penalty(self, order: int = 1, budget: float = None) -> torch.Tensor:
        """Soft smoothness penalty (>=1). order=1 bounds the network Lipschitz C;
        order=2 bounds the Jacobian Lipschitz (curvature) C*S."""
        b = budget if budget is not None else self.budget
        quantity = self.lipschitz_product() if order == 1 else self.jacobian_lipschitz_bound()
        return torch.clamp(quantity / b, min=1.0)

    def forward(self, xu: torch.Tensor) -> torch.Tensor:
        h = xu
        for l in self.layers:
            h = torch.relu(l(h))
        return self.out(h)


def cauchy_nll(pred: torch.Tensor, target: torch.Tensor,
               dispersion: torch.Tensor) -> torch.Tensor:
    """Mean Cauchy negative log-likelihood (heavy-tailed loss).

    -log C(x; mu, Sigma) ∝ (n+1)/2 * log(1 + (e/s)·(e/s))  (isotropic-per-dim).
    ``dispersion`` is a fixed per-dim scale (from residual MAD); robust to the
    impulse-like outliers that violate Gaussian (MSE) assumptions. Gradient
    saturates ∝ e/(1+e^2), unlike MSE's unbounded ∝ e.
    """
    n = pred.shape[-1]
    e = (target - pred) / (dispersion + 1e-9)
    return ((n + 1) / 2.0) * torch.log1p((e ** 2).sum(dim=-1)).mean()


def build_onestep(cfg: Dict) -> nn.Module:
    kind = cfg.get("model", "fnn").lower()
    if kind == "fnn":
        return OneStepFNN(hidden=cfg.get("hidden", 64))
    if kind == "lcnn":
        return OneStepLCNN(hidden=cfg.get("hidden", 40),
                           lipschitz_bound=cfg.get("lipschitz_bound", 3.0),
                           n_layers=cfg.get("n_layers", 2))
    if kind == "sns":
        return OneStepSNS(hidden=cfg.get("hidden", 64),
                          n_layers=cfg.get("n_layers", 2),
                          budget=cfg.get("budget", 8.0))
    if kind in ("smooth", "smooth_spectral"):
        return OneStepSmoothSpectral(hidden=cfg.get("hidden", 40),
                                     lipschitz_bound=cfg.get("lipschitz_bound", 3.0),
                                     n_layers=cfg.get("n_layers", 2),
                                     activation=cfg.get("activation", "tanh"))
    raise ValueError(f"unknown one-step model: {kind}")
