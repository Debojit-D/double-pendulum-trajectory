"""
Hamiltonian Neural Network (HNN) for double-pendulum style systems.

- Baseline mode: a standard MLP directly predicts time-derivatives dx.
- HNN mode: a scalar Hamiltonian network H(q, p) is learned; dynamics are
            computed via the canonical map: z = [q, p], ẋ = J ∇H, with
            J = [ 0  I; -I  0 ].

Public API:
    HNN(n_elements, hidden_dims, num_layers, baseline=False, nonlinearity='softplus')
        .forward(x) -> dx
        .energy(x)  -> H(x)   (only in HNN mode)
    make_optimizer(...)
    train_step(model, optimizer, criterion, x, dx)
    eval_step(model, criterion, x, dx)

Assumptions:
- Input x is shaped (B, 2*n_elements), ordered as [q (n), p (n)].
- Output dx is shaped (B, 2*n_elements), ordered as [q̇ (n), ṗ (n)].

Notes:
- We prefer `torch.func.jacrev` + `torch.vmap` for a clean ∇H(x).
  If unavailable, we fallback to autograd on a batch sum (works for both train/eval).
- Keep this file self-contained with minimal training utilities; the
  full pipeline remains in your trainer.
"""

from __future__ import annotations
from typing import List, Sequence, Union, Optional, Tuple
import torch
import torch.nn as nn

# Optional: torch.func (PyTorch ≥ 2.0); we gate usage dynamically
try:
    from torch.func import jacrev, vmap  # type: ignore
    _HAS_TORCH_FUNC = True
except Exception:
    _HAS_TORCH_FUNC = False

# ---------------------------------------------------------------------
# Small MLP builder (keeps your existing .utils.MLP usable if desired)
# If you already have src/models/utils.py::MLP(in_dim, hidden_dims, num_layers, out_dim, nonlinearity)
# feel free to import and swap here. This local version is safe and explicit.
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
        act = _ACTS.get(nonlinearity.lower(), nn.Softplus)

        # If num_layers includes input+output, infer the number of hidden layers
        # Otherwise, trust the provided hidden list
        if num_layers is not None and num_layers >= 2:
            # build a uniform stack if hidden dims given as int
            if len(hidden) == 1 and num_layers > 2:
                hidden = [hidden[0]] * (num_layers - 1)  # (#hidden layers)
        layers: List[nn.Module] = []
        d_prev = in_dim
        for h in hidden:
            layers += [nn.Linear(d_prev, h), act()]
            d_prev = h
        layers += [nn.Linear(d_prev, out_dim)]
        self.net = nn.Sequential(*layers)

        # Kaiming init for hidden; small output init for stability
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)
        if isinstance(self.net[-1], nn.Linear):
            nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------
# HNN core
# ---------------------------------------------------------------------

class HNN(nn.Module):
    """
    Hamiltonian Neural Network:
      - baseline=False: learn scalar H(q,p); return dx = J ∇H
      - baseline=True : direct regression dx = f_MLP(x)

    Args
    ----
    n_elements : int
        The DoF per (q or p). For double pendulum, n_elements=2.
    hidden_dims : int | List[int]
        Hidden sizes.
    num_layers : int
        Depth control (see MLP note).
    baseline : bool
        If True, use direct dx regression. If False (default), learn Hamiltonian.
    nonlinearity : str
        Activation name ('softplus', 'relu', 'tanh', ...).
    """

    def __init__(
        self,
        n_elements: int,
        hidden_dims: Union[int, List[int]] = 200,
        num_layers: int = 3,
        baseline: bool = False,
        nonlinearity: str = "softplus",
    ):
        super().__init__()
        assert n_elements >= 1, "n_elements must be >= 1"
        self.n = int(n_elements)
        self.d = 2 * self.n
        self.baseline = bool(baseline)

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
            # Scalar Hamiltonian: R^{2n} -> R
            self.H_net = MLP(
                in_dim=self.d,
                hidden_dims=hidden_dims,
                num_layers=num_layers,
                out_dim=1,
                nonlinearity=nonlinearity,
            )

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

        # HNN path: dx = J ∇H
        grad_H = self._grad_H(x)  # (B, 2n)
        q_grad, p_grad = grad_H.split(self.n, dim=-1)
        # canonical map: q̇ =  ∂H/∂p,  ṗ = -∂H/∂q
        q_dot = p_grad
        p_dot = -q_grad
        return torch.cat([q_dot, p_dot], dim=-1)

    @torch.no_grad()
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return Hamiltonian H(x). Only valid for HNN mode.
        """
        if self.baseline:
            raise RuntimeError("energy() only available when baseline=False.")
        H = self.H_net(x)
        return H.squeeze(-1)

    # --------- Internals ---------

    def _grad_H(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇H(x) for a batch. Uses torch.func when available,
        otherwise falls back to autograd over a batch-summed scalar.
        """
        if self.baseline:
            raise RuntimeError("grad_H is undefined for baseline=True")

        if _HAS_TORCH_FUNC:
            # jacrev returns Jacobian of shape (B, out_dim, in_dim) for batched x
            # with out_dim==1, squeeze to (B, in_dim)
            J = vmap(jacrev(self.H_net))(x)      # (B, 1, d)
            grad = J.squeeze(1)                  # (B, d)
            return grad

        # Fallback: autograd on batch sum. Works in both train/eval.
        # Keep graph in training for higher-order ops.
        x_req = x.requires_grad_(True)
        H = self.H_net(x_req).squeeze(-1)       # (B,)
        H_sum = H.sum()
        grad = torch.autograd.grad(
            H_sum, x_req,
            create_graph=self.training, retain_graph=True, only_inputs=True
        )[0]                                      # (B, d)
        return grad


# ---------------------------------------------------------------------
# Minimal training helpers (optional)
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
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)


def train_step(
    model: HNN,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    x: torch.Tensor,
    dx: torch.Tensor,
    grad_clip_norm: Optional[float] = 1.0,
) -> float:
    """
    One training step:
      loss = criterion(model(x), dx)

    You can augment this with energy regularizers or smoothness penalties outside.
    """
    model.train()
    pred = model(x.requires_grad_(True))
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
# Quick self-test (optional). Guarded by __main__.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    n = 2
    B = 8

    # Random test batch: x = [q, p]
    x = torch.randn(B, 2 * n, requires_grad=True)

    # HNN mode
    hnn = HNN(n_elements=n, hidden_dims=[128, 128], num_layers=3, baseline=False)
    dx = hnn(x)
    print("HNN dx shape:", dx.shape, "H shape:", hnn.energy(x).shape)

    # Baseline mode
    base = HNN(n_elements=n, hidden_dims=256, num_layers=3, baseline=True)
    dx_b = base(x)
    print("Baseline dx shape:", dx_b.shape)

    # Tiny training smoke test
    opt = make_optimizer(hnn, lr=1e-3)
    loss_fn = nn.MSELoss()
    target = torch.randn_like(dx)
    tr_loss = train_step(hnn, opt, loss_fn, x, target)
    te_loss = eval_step(hnn, loss_fn, x, target)
    print(f"train={tr_loss:.3e}  eval={te_loss:.3e}")
