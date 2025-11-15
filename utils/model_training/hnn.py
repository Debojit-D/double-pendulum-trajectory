#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hamiltonian Neural Network (HNN) for double-pendulum style systems.

Based on:
  Greydanus et al., "Hamiltonian Neural Networks", NeurIPS 2019.

Two modes:
  1) baseline=True
       - Plain MLP that directly regresses time-derivatives:
           f_theta : x = [q, p] -> dx = [q_dot, p_dot]
  2) baseline=False (default)
       - Learn a scalar Hamiltonian H_theta(q, p).
       - Dynamics are given by the canonical Hamiltonian flow:
           z = [q, p] in R^{2n}
           grad H = ∇_z H(z) = [∂H/∂q, ∂H/∂p]
           q_dot =  ∂H/∂p
           p_dot = -∂H/∂q
         so dx = [q_dot, p_dot] = J ∇H, with
           J = [  0  I ;
                -I  0 ]

Extensions:
  - separable=True:
        H(q, p) = T_theta(p) + V_theta(q)
    with two MLPs (one for kinetic, one for potential energy).
  - dissipative=True:
        dx = J ∇H(z) + λ * g_theta(z)
    where g_theta is an unconstrained vector field
    (useful for viscous / friction regimes).

Public API
----------
    HNN(
        n_elements,
        hidden_dims=200,
        num_layers=3,
        baseline=False,
        nonlinearity='softplus',
        separable=False,
        dissipative=False,
        dissipation_scale=1.0,
    )

        .forward(x) -> dx          # [q̇, ṗ]
        .energy(x)  -> H(x)        # scalar energy per sample (HNN mode only)

    make_optimizer(...)
    train_step(...)
    eval_step(...)

Assumptions
-----------
- Input x is shaped (B, 2*n_elements), ordered as [q (n), p (n)].
- Output dx is shaped (B, 2*n_elements), ordered as [q̇ (n), ṗ (n)].

You should handle dataset normalization and batching in your trainer.
"""

from __future__ import annotations
from typing import List, Sequence, Union, Optional, Tuple
import torch
import torch.nn as nn

# Optional: torch.func (PyTorch ≥ 2.0)
try:
    from torch.func import jacrev, vmap  # type: ignore
    _HAS_TORCH_FUNC = True
except Exception:
    _HAS_TORCH_FUNC = False

# ---------------------------------------------------------------------
# Small MLP builder
# ---------------------------------------------------------------------

_ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
}


def _as_list(hidden_dims: Union[int, Sequence[int]]) -> List[int]:
    return [int(hidden_dims)] if isinstance(hidden_dims, int) else list(hidden_dims)


class MLP(nn.Module):
    """
    Simple MLP with configurable depth/width/activation.

    Args
    ----
    in_dim      : input dimension
    hidden_dims : int or list[int] — hidden layer widths
    num_layers  : number of hidden layers if hidden_dims is an int
    out_dim     : output dimension
    nonlinearity: activation name ('softplus', 'relu', 'tanh', ...)
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Union[int, Sequence[int]] = 200,
        num_layers: int = 3,
        out_dim: int = 1,
        nonlinearity: str = "softplus",
    ):
        super().__init__()
        hidden = _as_list(hidden_dims)
        act_cls = _ACTS.get(nonlinearity.lower(), nn.Softplus)

        # If a single hidden size is given, repeat it num_layers times
        if len(hidden) == 1 and num_layers is not None and num_layers > 0:
            hidden = hidden * num_layers

        dims = [in_dim] + hidden + [out_dim]
        layers: List[nn.Module] = []
        for d_in, d_out in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(act_cls())
        layers.append(nn.Linear(dims[-2], dims[-1]))  # final linear, no activation
        self.net = nn.Sequential(*layers)

        # Kaiming init for hidden; small output init for stability
        for i, m in enumerate(self.net):
            if isinstance(m, nn.Linear):
                if i < len(self.net) - 1:
                    # hidden layers
                    nn.init.kaiming_uniform_(m.weight, a=0.01)
                    nn.init.zeros_(m.bias)
                else:
                    # output layer: very small init
                    nn.init.uniform_(m.weight, -1e-3, 1e-3)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------
# HNN core
# ---------------------------------------------------------------------

class HNN(nn.Module):
    """
    Hamiltonian Neural Network.

    Modes
    -----
    baseline = False (default):
        - Learn scalar Hamiltonian H(q, p).
        - dx = [q̇, ṗ] = [∂H/∂p, -∂H/∂q] (canonical Hamiltonian flow).
    baseline = True:
        - Plain MLP: dx = f_theta(x) with no physics structure.

    Args
    ----
    n_elements : int
        DoF per generalized coordinate (q) or momentum (p). For a
        double pendulum, n_elements = 2, x = [q1, q2, p1, p2].
    hidden_dims : int | List[int]
        Hidden sizes for internal networks.
    num_layers : int
        Depth control for MLPs (number of hidden layers if hidden_dims is int).
    baseline : bool
        If True, use direct dx regression. Otherwise, learn Hamiltonian.
    nonlinearity : str
        Activation name ('softplus', 'relu', 'tanh', ...).
    separable : bool
        If True, use H(q, p) = T(p) + V(q) with two MLPs,
        which often stabilizes training and improves interpretability.
    dissipative : bool
        If True, add an unconstrained dissipative term g_theta(x)
        to the canonical Hamiltonian flow.
    dissipation_scale : float
        Scalar λ multiplying the dissipative term.
    """

    def __init__(
        self,
        n_elements: int,
        hidden_dims: Union[int, List[int]] = 200,
        num_layers: int = 3,
        baseline: bool = False,
        nonlinearity: str = "softplus",
        separable: bool = False,
        dissipative: bool = False,
        dissipation_scale: float = 1.0,
    ):
        super().__init__()
        assert n_elements >= 1, "n_elements must be >= 1"
        self.n = int(n_elements)
        self.d = 2 * self.n
        self.baseline = bool(baseline)
        self.separable = bool(separable)
        self.dissipative = bool(dissipative)
        self.dissipation_scale = float(dissipation_scale)

        if self.baseline:
            # Direct dx regressor: R^{2n} -> R^{2n}
            self.dx_net = MLP(
                in_dim=self.d,
                hidden_dims=hidden_dims,
                num_layers=num_layers,
                out_dim=self.d,
                nonlinearity=nonlinearity,
            )
        else:
            # Hamiltonian networks
            if self.separable:
                # H(q,p) = T(p) + V(q)
                self.T_net = MLP(
                    in_dim=self.n,
                    hidden_dims=hidden_dims,
                    num_layers=num_layers,
                    out_dim=1,
                    nonlinearity=nonlinearity,
                )
                self.V_net = MLP(
                    in_dim=self.n,
                    hidden_dims=hidden_dims,
                    num_layers=num_layers,
                    out_dim=1,
                    nonlinearity=nonlinearity,
                )
            else:
                # Generic scalar H(q,p)
                self.H_net = MLP(
                    in_dim=self.d,
                    hidden_dims=hidden_dims,
                    num_layers=num_layers,
                    out_dim=1,
                    nonlinearity=nonlinearity,
                )

            # Optional dissipative field g_theta(x)
            if self.dissipative:
                self.g_net = MLP(
                    in_dim=self.d,
                    hidden_dims=hidden_dims,
                    num_layers=num_layers,
                    out_dim=self.d,
                    nonlinearity=nonlinearity,
                )
            else:
                self.g_net = None

    # --------- Internal Hamiltonian helpers ---------

    def _hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute H(x) for a batch.

        x : (B, 2n)
        returns H : (B, 1)
        """
        assert not self.baseline, "Hamiltonian is undefined for baseline=True"

        if self.separable:
            q, p = x.split(self.n, dim=-1)  # each (B, n)
            T = self.T_net(p)              # (B, 1)
            V = self.V_net(q)              # (B, 1)
            return T + V
        else:
            return self.H_net(x)           # (B, 1)

    def _H_single(self, z: torch.Tensor) -> torch.Tensor:
        """
        Scalar H(z) for a single state z in R^{2n}.

        z : (2n,)
        returns scalar tensor ()
        """
        H = self._hamiltonian(z.unsqueeze(0))  # (1, 1)
        return H.squeeze(0).squeeze(-1)

    def _grad_H(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇H(x) for a batch.

        Parameters
        ----------
        x : (B, 2n)

        Returns
        -------
        grad_H : (B, 2n)
            grad_H[b] = ∇_z H(z_b)
        """
        if self.baseline:
            raise RuntimeError("grad_H is undefined for baseline=True")

        if _HAS_TORCH_FUNC:
            # jacrev on scalar function H(z) with z in R^{2n}
            # then vmap over the batch dimension
            grad_fun = jacrev(self._H_single)         # (2n,) -> (2n,)
            grad = vmap(grad_fun)(x)                  # (B, 2n)
            return grad

        # Fallback: autograd on batch sum. Works train+eval.
        # We re-run the forward with x requiring grad.
        x_req = x.requires_grad_(True)
        H = self._hamiltonian(x_req).squeeze(-1)      # (B,)
        H_sum = H.sum()
        grad = torch.autograd.grad(
            H_sum,
            x_req,
            create_graph=self.training,  # keep graph in train for higher-order grads
            retain_graph=True,
            only_inputs=True,
        )[0]                                          # (B, 2n)
        return grad

    # --------- Public API ---------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute time-derivatives dx given state x.

        Parameters
        ----------
        x : (B, 2n) tensor
            Concatenated [q, p] per sample.

        Returns
        -------
        dx : (B, 2n) tensor
            Concatenated [q_dot, p_dot].
        """
        if self.baseline:
            return self.dx_net(x)

        # Hamiltonian flow: dx = J ∇H
        grad_H = self._grad_H(x)               # (B, 2n)
        q_grad, p_grad = grad_H.split(self.n, dim=-1)  # each (B, n)

        # Canonical equations:
        #   q̇ =  ∂H/∂p
        #   ṗ = -∂H/∂q
        q_dot = p_grad
        p_dot = -q_grad
        dx = torch.cat([q_dot, p_dot], dim=-1)  # (B, 2n)

        # Optional dissipative term
        if self.g_net is not None and self.dissipation_scale != 0.0:
            g = self.g_net(x)
            dx = dx + self.dissipation_scale * g

        return dx

    @torch.no_grad()
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return Hamiltonian H(x) as a 1D tensor of length B.

        Only valid for HNN mode (baseline=False).
        """
        if self.baseline:
            raise RuntimeError("energy() only available when baseline=False.")
        H = self._hamiltonian(x)  # (B, 1)
        return H.squeeze(-1)


# ---------------------------------------------------------------------
# Minimal training helpers
# ---------------------------------------------------------------------

def make_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """
    Convenience Adam optimizer with light weight decay default.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )


def train_step(
    model: HNN,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    x: torch.Tensor,
    dx: torch.Tensor,
    grad_clip_norm: Optional[float] = 1.0,
) -> float:
    """
    One training step with derivative supervision:

        loss = criterion(model(x), dx)

    You can augment this with:
      - energy regularizers (e.g., ||H(x_0) - H(target)||^2),
      - smoothness penalties on H or dx,
    outside this function.
    """
    model.train()
    pred = model(x)
    loss = criterion(pred, dx)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return float(loss.detach().cpu().item())


@torch.no_grad()
def eval_step(
    model: HNN,
    criterion: nn.Module,
    x: torch.Tensor,
    dx: torch.Tensor,
) -> float:
    """
    Evaluation step (no grad):

        loss = criterion(model(x), dx)
    """
    model.eval()
    pred = model(x)
    loss = criterion(pred, dx)
    return float(loss.detach().cpu().item())


# ---------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    n = 2
    B = 8

    # Random test batch: x = [q, p]
    x = torch.randn(B, 2 * n)

    print("=== HNN mode (generic H) ===")
    hnn = HNN(
        n_elements=n,
        hidden_dims=[128, 128],
        num_layers=2,
        baseline=False,
        separable=False,
        dissipative=True,
        dissipation_scale=0.1,
    )
    dx = hnn(x)
    H = hnn.energy(x)
    print("dx shape:", dx.shape, "H shape:", H.shape)

    print("\n=== HNN mode (separable H = T(p)+V(q)) ===")
    hnn_sep = HNN(
        n_elements=n,
        hidden_dims=128,
        num_layers=2,
        baseline=False,
        separable=True,
        dissipative=False,
    )
    dx_sep = hnn_sep(x)
    H_sep = hnn_sep.energy(x)
    print("dx shape:", dx_sep.shape, "H shape:", H_sep.shape)

    print("\n=== Baseline (no physics) ===")
    base = HNN(
        n_elements=n,
        hidden_dims=256,
        num_layers=3,
        baseline=True,
    )
    dx_b = base(x)
    print("Baseline dx shape:", dx_b.shape)

    # Tiny training smoke test
    opt = make_optimizer(hnn, lr=1e-3)
    loss_fn = nn.MSELoss()
    target = torch.randn_like(dx)
    tr_loss = train_step(hnn, opt, loss_fn, x, target)
    te_loss = eval_step(hnn, loss_fn, x, target)
    print(f"\ntrain={tr_loss:.3e}  eval={te_loss:.3e}")
