#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Baseline and/or Hamiltonian Neural Network (HNN) on double-pendulum trajectories.

Data expected (per-trajectory CSV):
- Columns:
    t,q1,q2,dq1,dq2,
    tip_x,tip_y,tip_z,
    tip_x_rel,tip_z_rel,
    elbow_x,elbow_z,
    step_idx,
    ddq1,ddq2

We build CANONICAL coordinates:
    For each sample:
        q      = [q1, q2]
        dq     = [dq1, dq2]
        p      = M(q2) @ dq     (2×2 mass matrix, see _mass_matrix_canonical)
        dp/dt  = finite-diff p(t)

    X    = [q1, q2, p1, p2]
    Xdot = [dq1, dq2, dp1, dp2]

Canonical HNN expects:
    x  = [q, p]
    dx = [q_dot, p_dot]

Training:
- Concatenate all trajectories after:
    - Clamping to first T_MAX_TRAIN seconds per trajectory
    - Optional time subsampling via STRIDE
    - Angle-wrapping q1,q2 into (-pi, pi]

- Compute normalization stats from TRAIN split:
      x_n   = (x - X_mean)   / X_std
      dx_n  = (dx - dX_mean) / dX_std
- DataLoaders feed (x_n, dx_n) to the trainer.
- Inside train loop, we:
      x_phys  = x_n * X_std + X_mean
      dx_pred = HNN(x_phys)          # in physical units
      dx_pred_n = (dx_pred - dX_mean) / dX_std
  and minimize MSE(dx_pred_n, dx_n).

Outputs:
- OUT_DIR/
    models/hnn_baseline.pth      (if TRAIN_BASELINE = True)
    models/hnn_hnn.pth           (if TRAIN_HNN = True)
    hnn_run.json                 (run metadata: losses, config, shapes, stats, normalization)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm, trange
import sys

# --- Make project root importable so we can import utils.model_training.hnn ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (…/double-pendulum-trajectory)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_training.hnn import HNN  # your HNN class


# ===================== HARD-CODED CONFIG =====================

DATA_DIR = Path(
    "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal2"
)
MANIFEST = DATA_DIR / "double_pendulum_manifest_ideal.json"

OUT_DIR = DATA_DIR / "hnn_model_final"

# Use only first T_MAX_TRAIN seconds of each trajectory
T_MAX_TRAIN = 5.0  # seconds

# Downsample stride inside each trajectory (for efficiency)
STRIDE = 5  # e.g. 1 = no stride, 5 = keep every 5th sample

# Training hyperparameters
BATCH_SIZE = 1024
N_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2  # 20% for validation

GRAD_CLIP = 1.0
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 10
EARLY_STOP = 20            # <=0 disables early stopping

# HNN model hyperparameters
HIDDEN_DIMS = 256          # or [128,256,256] if your HNN supports list
NUM_LAYERS = 3
NONLINEARITY = "softplus"  # passed into HNN

# HNN variants
HNN_SEPARABLE = False
HNN_DISSIPATIVE = False
HNN_DISSIPATION_SCALE = 1.0

# Which models to train
TRAIN_BASELINE = False   # plain dx = f(x) regressor using HNN in baseline mode
TRAIN_HNN = True         # Hamiltonian model

# Runtime / output
DEVICE_STR = None        # "cuda" | "cpu" | None(auto)
SEED = 42
VERBOSE = True

SAVE_PREFIX_BASELINE = "hnn_baseline"
SAVE_PREFIX_HNN = "hnn_hnn"

# =============================================================

# Minimum required columns (ddq1,ddq2 checked later)
REQ_COLS = ["t", "q1", "q2", "dq1", "dq2"]


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _ensure_finite(
    arr: np.ndarray,
    name: str,
    ctx: str = "",
    dump_dir: Optional[Path] = None,
):
    """Raise with location if arr has NaN/Inf; optionally dump to disk."""
    if not np.isfinite(arr).all():
        r, c = np.argwhere(~np.isfinite(arr))[0]
        val = arr[r, c]
        msg = f"[NON-FINITE] {name} has {val} at (row={r}, col={c})."
        if ctx:
            msg += f" Context: {ctx}"
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            np.save(dump_dir / f"{name.replace(' ', '_')}.npy", arr)
            msg += f" Dumped to: {dump_dir/(name.replace(' ', '_') + '.npy')}"
        raise ValueError(msg)


def _read_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load one trajectory CSV into dict of arrays."""
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    for c in REQ_COLS:
        if c not in arr.dtype.names:
            raise ValueError(f"[DATA] Missing required column '{c}' in {csv_path}")
    # ddq1, ddq2 must exist (we may still want them for diagnostics / future use)
    for c in ("ddq1", "ddq2"):
        if c not in arr.dtype.names:
            raise ValueError(
                f"[DATA] Missing required column '{c}' (need MuJoCo accelerations) in {csv_path}"
            )
    return {name: arr[name] for name in arr.dtype.names}


# ---------------------------------------------------------------------
# Canonical momenta helpers
# ---------------------------------------------------------------------

def _mass_matrix_canonical(q_rel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mass matrix M(q2) for a planar 2-link double pendulum with:
        - unit lengths (l1 = l2 = 1),
        - unit point masses at link ends (m1 = m2 = 1),
        - zero link inertias at COM (I1 = I2 = 0),
        - second joint angle defined *relative* to the first (your q2),
          so abs angle of link 2 is (q1 + q2).

    Result from analytic derivation:
        M11 = 1.5 + cos(q2)
        M12 = 0.5*cos(q2) + 0.25
        M22 = 0.25

    We return components as arrays so we can build p = M(q2) dq efficiently.
    """
    c2 = np.cos(q_rel)
    M11 = 1.5 + c2
    M12 = 0.5 * c2 + 0.25
    M22 = 0.25 * np.ones_like(c2)
    return M11, M12, M22


def _q_dq_to_canonical(
    t: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert trajectory (t, q, dq) to canonical (X, Xdot):

        q      : (T, 2)  -> [q1, q2]
        dq     : (T, 2)  -> [dq1, dq2]
        M(q2)  : 2×2 mass matrix from _mass_matrix_canonical
        p      : (T, 2)  -> p = M(q2) @ dq
        dp/dt  : (T, 2)  -> finite-diff in time

    Returns
    -------
    X    : (T, 4) = [q1, q2, p1, p2]
    Xdot : (T, 4) = [dq1, dq2, dp1, dp2]
    """
    if q.shape[1] != 2 or dq.shape[1] != 2:
        raise ValueError("Expected q,dq to have shape (T,2) for double pendulum.")

    if t.shape[0] != q.shape[0]:
        raise ValueError("t and q must have same length.")

    T = q.shape[0]
    if T < 2:
        raise ValueError("Need at least 2 timesteps per trajectory to build dp/dt.")

    # Canonical momenta: p = M(q2) * dq
    q2 = q[:, 1]  # relative angle
    M11, M12, M22 = _mass_matrix_canonical(q2)

    dq1 = dq[:, 0]
    dq2 = dq[:, 1]

    p1 = M11 * dq1 + M12 * dq2
    p2 = M12 * dq1 + M22 * dq2
    p = np.stack([p1, p2], axis=1)  # (T, 2)

    # Time derivatives of p via finite-difference
    dp = np.zeros_like(p)
    dt = np.diff(t)

    # Central differences for interior points
    dt_c = (t[2:] - t[:-2])
    dp[1:-1] = (p[2:] - p[:-2]) / dt_c[:, None]

    # Forward/backward for boundaries
    dp[0] = (p[1] - p[0]) / (t[1] - t[0])
    dp[-1] = (p[-1] - p[-2]) / (t[-1] - t[-2])

    # Assemble canonical X and Xdot
    X = np.hstack([q, p])      # [q1, q2, p1, p2]
    Xdot = np.hstack([dq, dp]) # [dq1, dq2, dp1, dp2]
    return X, Xdot


# ---------------------------------------------------------------------
# Dataset build
# ---------------------------------------------------------------------

def _build_dataset(
    data_dir: Path,
    manifest: Optional[Path],
    t_max: float,
    stride: int,
    debug_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build full dataset (X, Xdot) by concatenating all trajectories in CANONICAL coordinates:

        X    = [q1, q2, p1, p2]
        Xdot = [dq1, dq2, dp1, dp2]

    where p is the canonical momentum from a planar 2-link double pendulum model
    and dp/dt is obtained by finite differencing along the trajectory.
    """
    if manifest is not None and manifest.exists():
        meta = json.loads(manifest.read_text())
        csv_paths = [data_dir / r["csv"] for r in meta.get("runs", [])]
        if not csv_paths:
            raise ValueError(f"[DATA] No runs in manifest: {manifest}")
    else:
        csv_paths = sorted(data_dir.glob("double_pendulum_traj_*.csv"))
        if not csv_paths:
            raise ValueError(f"[DATA] No CSVs found in {data_dir}")

    X_list: List[np.ndarray] = []
    Xdot_list: List[np.ndarray] = []

    for pth in tqdm(csv_paths, desc="Loading trajectories for HNN"):
        d = _read_csv(pth)
        t = d["t"].astype(float)

        # Per-run clamp: only use first t_max seconds
        mask = t <= float(t_max)
        if not np.any(mask):
            continue

        t = t[mask]
        q = np.vstack([d["q1"][mask], d["q2"][mask]]).T
        dq = np.vstack([d["dq1"][mask], d["dq2"][mask]]).T

        # ddq not directly used in canonical mapping here, but we still sanity-check it
        ddq = np.vstack([d["ddq1"][mask], d["ddq2"][mask]]).T

        # Wrap angles for stability
        q = _wrap_to_pi(q)

        _ensure_finite(q, "q", ctx=str(pth), dump_dir=debug_dir)
        _ensure_finite(dq, "dq", ctx=str(pth), dump_dir=debug_dir)
        _ensure_finite(ddq, "ddq", ctx=str(pth), dump_dir=debug_dir)

        # Convert this trajectory to canonical (q,p) and (q_dot, p_dot)
        X_traj, Xdot_traj = _q_dq_to_canonical(t, q, dq)

        _ensure_finite(X_traj, "X_traj", ctx=str(pth), dump_dir=debug_dir)
        _ensure_finite(Xdot_traj, "Xdot_traj", ctx=str(pth), dump_dir=debug_dir)

        if stride > 1:
            sl = slice(0, None, stride)
            X_traj = X_traj[sl]
            Xdot_traj = Xdot_traj[sl]

        X_list.append(X_traj)
        Xdot_list.append(Xdot_traj)

    if not X_list:
        raise RuntimeError("[DATA] No valid samples after t_max/stride filtering.")

    X_all = np.vstack(X_list)
    Xdot_all = np.vstack(Xdot_list)

    _ensure_finite(X_all, "X_all", dump_dir=debug_dir)
    _ensure_finite(Xdot_all, "Xdot_all", dump_dir=debug_dir)

    print(f"[INFO] Built dataset with shapes: X={X_all.shape}, Xdot={Xdot_all.shape}")
    return X_all, Xdot_all


def _dataset_stats(X: np.ndarray, Xdot: np.ndarray) -> Dict[str, Dict[str, float]]:
    def stats(v):
        return dict(
            min=float(np.min(v)),
            max=float(np.max(v)),
            mean=float(np.mean(v)),
            std=float(np.std(v)),
        )

    # X = [q1, q2, p1, p2]
    # Xdot = [dq1, dq2, dp1, dp2]
    S = {
        "q1": stats(X[:, 0]),
        "q2": stats(X[:, 1]),
        "p1": stats(X[:, 2]),
        "p2": stats(X[:, 3]),
        "dq1": stats(Xdot[:, 0]),
        "dq2": stats(Xdot[:, 1]),
        "dp1": stats(Xdot[:, 2]),
        "dp2": stats(Xdot[:, 3]),
    }
    return S


# ---------------------------------------------------------------------
# Training core (with normalization)
# ---------------------------------------------------------------------

def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
    sched_factor: float = 0.5,
    sched_patience: int = 10,
    early_stop_patience: int = 20,
    verbose: bool = True,
    # Normalization stats: loaders provide normalized x_n, dx_n
    x_mean: Optional[torch.Tensor] = None,
    x_std: Optional[torch.Tensor] = None,
    dx_mean: Optional[torch.Tensor] = None,
    dx_std: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    MSE loss + Adam + LR-on-plateau + grad clip + optional early stopping.

    DataLoaders are assumed to yield normalized (x_n, dx_n):
        x_n  = (x - x_mean) / x_std
        dx_n = (dx - dx_mean) / dx_std

    Inside this function we:
        x_phys    = x_n * x_std + x_mean
        dx_pred   = model(x_phys)                  # physical units
        dx_pred_n = (dx_pred - dx_mean) / dx_std
        loss      = MSE(dx_pred_n, dx_n)
    """
    model = model.to(device)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(
        opt,
        mode="min",
        factor=sched_factor,
        patience=sched_patience,
    )

    # Make sure stats are on device and broadcastable
    if x_mean is not None:
        x_mean = x_mean.to(device)
    if x_std is not None:
        x_std = x_std.to(device)
    if dx_mean is not None:
        dx_mean = dx_mean.to(device)
    if dx_std is not None:
        dx_std = dx_std.to(device)

    stats = {"train_loss": [], "val_loss": [], "learning_rate": []}
    best = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in trange(1, epochs + 1, desc=f"Train({model.__class__.__name__})", disable=not verbose):
        # ---------- train ----------
        model.train()
        tr_losses = []
        for xb_n, db_n in train_loader:
            xb_n = xb_n.to(device)
            db_n = db_n.to(device)

            xb_n = xb_n.requires_grad_(True)

            if x_mean is not None and x_std is not None and dx_mean is not None and dx_std is not None:
                # Un-normalize input to physical canonical coordinates
                xb_phys = xb_n * x_std + x_mean
                dx_pred_phys = model(xb_phys)
                # Normalize predicted derivatives
                dx_pred_n = (dx_pred_phys - dx_mean) / dx_std
                loss = crit(dx_pred_n, db_n)
            else:
                # Fallback: train directly in physical space (no normalization)
                dx_pred = model(xb_n)
                loss = crit(dx_pred, db_n)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_losses.append(loss.item())

        # ---------- eval ----------
        model.eval()
        te_losses = []
        with torch.no_grad():
            for xb_n, db_n in val_loader:
                xb_n = xb_n.to(device)
                db_n = db_n.to(device)

                if x_mean is not None and x_std is not None and dx_mean is not None and dx_std is not None:
                    xb_phys = xb_n * x_std + x_mean
                    dx_pred_phys = model(xb_phys)
                    dx_pred_n = (dx_pred_phys - dx_mean) / dx_std
                    loss = crit(dx_pred_n, db_n)
                else:
                    dx_pred = model(xb_n)
                    loss = crit(dx_pred, db_n)

                te_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else np.nan
        te_loss = float(np.mean(te_losses)) if te_losses else np.nan
        stats["train_loss"].append(tr_loss)
        stats["val_loss"].append(te_loss)
        stats["learning_rate"].append(opt.param_groups[0]["lr"])

        if verbose:
            tqdm.write(
                f"Epoch {epoch:4d} | "
                f"train {tr_loss:.3e} | val {te_loss:.3e} | "
                f"lr {opt.param_groups[0]['lr']:.2e}"
            )

        # LR scheduler + early stopping
        sched.step(te_loss)

        improved = te_loss + 1e-12 < best
        if improved:
            best = te_loss
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if early_stop_patience is not None and early_stop_patience > 0:
                if epochs_no_improve >= early_stop_patience:
                    if verbose:
                        tqdm.write(
                            f"[EarlyStop] Stopping at epoch {epoch}. "
                            f"Best val {best:.3e}"
                        )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, stats


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    debug_dir = OUT_DIR / "debug_dump"

    print(f"[INFO] Data dir: {DATA_DIR}")
    print(f"[INFO] Manifest: {MANIFEST if MANIFEST.exists() else 'None (using glob)'}")
    print(f"[INFO] Out dir:   {OUT_DIR}")
    print(f"[INFO] Using only first {T_MAX_TRAIN:.2f}s of each trajectory, stride={STRIDE}")

    # Seeds & device
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if DEVICE_STR:
        device = torch.device(DEVICE_STR)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if VERBOSE:
        print(f"[INFO] device: {device}")

    # ---------------------- Build dataset (canonical) ----------------------
    X, Xdot = _build_dataset(
        data_dir=DATA_DIR,
        manifest=MANIFEST if MANIFEST.exists() else None,
        t_max=T_MAX_TRAIN,
        stride=STRIDE,
        debug_dir=debug_dir,
    )

    stats = _dataset_stats(X, Xdot)
    print("[INFO] Dataset stats (canonical):")
    for k, v in stats.items():
        print(f"  {k:5s}: min={v['min']:+.4e} max={v['max']:+.4e} mean={v['mean']:+.4e} std={v['std']:+.4e}")

    N = X.shape[0]
    N_val = int(VAL_SPLIT * N)
    N_train = N - N_val

    # Simple split: first N_train train, last N_val val
    X_train, X_val = X[:N_train], X[N_train:]
    Xdot_train, Xdot_val = Xdot[:N_train], Xdot[N_train:]

    print(f"[INFO] Train size: {N_train}, Val size: {N_val}")

    # ---------------------- Normalization (train only) ----------------------
    eps = 1e-8
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std = np.where(X_std < eps, 1.0, X_std)

    dX_mean = Xdot_train.mean(axis=0)
    dX_std = Xdot_train.std(axis=0)
    dX_std = np.where(dX_std < eps, 1.0, dX_std)

    print("[INFO] Normalization stats (train only):")
    print("  X_mean  =", X_mean)
    print("  X_std   =", X_std)
    print("  dX_mean =", dX_mean)
    print("  dX_std  =", dX_std)

    # Normalized datasets
    X_train_n = (X_train - X_mean) / X_std
    X_val_n = (X_val - X_mean) / X_std
    Xdot_train_n = (Xdot_train - dX_mean) / dX_std
    Xdot_val_n = (Xdot_val - dX_mean) / dX_std

    # Torch versions
    X_train_t = torch.from_numpy(X_train_n).float()
    X_val_t = torch.from_numpy(X_val_n).float()
    Xdot_train_t = torch.from_numpy(Xdot_train_n).float()
    Xdot_val_t = torch.from_numpy(Xdot_val_n).float()

    train_ds = TensorDataset(X_train_t, Xdot_train_t)
    val_ds = TensorDataset(X_val_t, Xdot_val_t)

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Normalization tensors for trainer
    X_mean_t = torch.from_numpy(X_mean).float()
    X_std_t = torch.from_numpy(X_std).float()
    dX_mean_t = torch.from_numpy(dX_mean).float()
    dX_std_t = torch.from_numpy(dX_std).float()

    # ---------------------- Train models ----------------------
    stats_all: Dict[str, Dict[str, List[float]]] = {}
    saved_paths: Dict[str, str] = {}

    # We expect X = [q1, q2, p1, p2] → 2 generalized coordinates
    n_elems = 2

    # Baseline: plain MLP dx = f(x)
    if TRAIN_BASELINE:
        if VERBOSE:
            print("\n[INFO] Training BASELINE model (direct dx regressor)...")
        baseline = HNN(
            n_elements=n_elems,
            hidden_dims=HIDDEN_DIMS,
            num_layers=NUM_LAYERS,
            baseline=True,
            nonlinearity=NONLINEARITY,
        )
        baseline, stats_b = train_one_model(
            baseline,
            train_loader,
            val_loader,
            device,
            epochs=N_EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            grad_clip=GRAD_CLIP,
            sched_factor=SCHED_FACTOR,
            sched_patience=SCHED_PATIENCE,
            early_stop_patience=EARLY_STOP,
            verbose=VERBOSE,
            x_mean=X_mean_t,
            x_std=X_std_t,
            dx_mean=dX_mean_t,
            dx_std=dX_std_t,
        )
        stats_all["baseline"] = stats_b

        model_dir = (OUT_DIR / "models")
        model_dir.mkdir(parents=True, exist_ok=True)
        p = model_dir / f"{SAVE_PREFIX_BASELINE}.pth"
        torch.save(baseline.state_dict(), p)
        saved_paths["baseline"] = str(p)

    # HNN: structured Hamiltonian flow
    if TRAIN_HNN:
        if VERBOSE:
            print("\n[INFO] Training HNN model (Hamiltonian flow)...")
            print(
                f"       separable={HNN_SEPARABLE}, "
                f"dissipative={HNN_DISSIPATIVE}, "
                f"dissipation_scale={HNN_DISSIPATION_SCALE}"
            )
        hnn = HNN(
            n_elements=n_elems,
            hidden_dims=HIDDEN_DIMS,
            num_layers=NUM_LAYERS,
            baseline=False,
            nonlinearity=NONLINEARITY,
            separable=HNN_SEPARABLE,
            dissipative=HNN_DISSIPATIVE,
            dissipation_scale=HNN_DISSIPATION_SCALE,
        )
        hnn, stats_h = train_one_model(
            hnn,
            train_loader,
            val_loader,
            device,
            epochs=N_EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            grad_clip=GRAD_CLIP,
            sched_factor=SCHED_FACTOR,
            sched_patience=SCHED_PATIENCE,
            early_stop_patience=EARLY_STOP,
            verbose=VERBOSE,
            x_mean=X_mean_t,
            x_std=X_std_t,
            dx_mean=dX_mean_t,
            dx_std=dX_std_t,
        )
        stats_all["hnn"] = stats_h

        model_dir = (OUT_DIR / "models")
        model_dir.mkdir(parents=True, exist_ok=True)
        p = model_dir / f"{SAVE_PREFIX_HNN}.pth"
        torch.save(hnn.state_dict(), p)
        saved_paths["hnn"] = str(p)

    if not stats_all:
        raise RuntimeError("No models were trained. Enable TRAIN_BASELINE and/or TRAIN_HNN in the config.")

    # ---------------------- Final metrics summary ----------------------
    final_losses = {}
    for tag, s in stats_all.items():
        tr = s.get("train_loss", [])
        va = s.get("val_loss", [])
        if tr and va:
            best_val = float(min(va))
            best_train = float(min(tr))
            best_epoch = int(np.argmin(va)) + 1  # 1-based
        else:
            best_val = float("nan")
            best_train = float("nan")
            best_epoch = -1
        final_losses[tag] = {
            "best_train": best_train,
            "best_val": best_val,
            "best_epoch": best_epoch,
        }

    # ---------------------- Save manifest ----------------------
    meta = {
        "data_dir": str(DATA_DIR),
        "manifest": str(MANIFEST) if MANIFEST.exists() else None,
        "shape_X": [int(v) for v in X.shape],
        "shape_Xdot": [int(v) for v in Xdot.shape],
        "train_size": int(N_train),
        "val_size": int(N_val),
        "t_max_train": float(T_MAX_TRAIN),
        "stride": int(STRIDE),
        "val_split": float(VAL_SPLIT),
        "trained_models": list(stats_all.keys()),
        "save_paths": saved_paths,
        "dataset_stats": stats,
        "normalization": {
            "X_mean": X_mean.tolist(),
            "X_std": X_std.tolist(),
            "dX_mean": dX_mean.tolist(),
            "dX_std": dX_std.tolist(),
        },
        "config": {
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "early_stop_patience": EARLY_STOP,
            "scheduler_factor": SCHED_FACTOR,
            "scheduler_patience": SCHED_PATIENCE,
            "hidden_dims": HIDDEN_DIMS if isinstance(HIDDEN_DIMS, int) else list(HIDDEN_DIMS),
            "num_layers": NUM_LAYERS,
            "nonlinearity": NONLINEARITY,
            "device": str(device),
            "hnn_separable": HNN_SEPARABLE,
            "hnn_dissipative": HNN_DISSIPATIVE,
            "hnn_dissipation_scale": HNN_DISSIPATION_SCALE,
            "seed": SEED,
        },
        "final_losses": final_losses,
    }
    (OUT_DIR / "hnn_run.json").write_text(json.dumps(meta, indent=2))

    # ---------------------- Summary print ----------------------
    print("\n[✓] HNN training complete.")
    print(f"    Out dir: {OUT_DIR}")
    for k, p in saved_paths.items():
        print(f"    Saved {k}: {p}")
    for tag, fl in final_losses.items():
        print(
            f"    {tag}: best train={fl['best_train']:.3e}  "
            f"best val={fl['best_val']:.3e}  at epoch {fl['best_epoch']}"
        )


if __name__ == "__main__":
    main()
