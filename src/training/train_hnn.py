#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Baseline and/or Hamiltonian Neural Network (HNN) on a provided tensor.

Expected data:
    torch.Tensor with shape [N_traj, T, D]

Column mapping (configurable via flags):
    x  = data[:, :, x_cols]     -> state (e.g., [theta1, theta2, p1, p2])
    dx = data[:, :, dx_cols]    -> time-derivative (matching x order)

Outputs (under --out-dir):
    models/{prefix}_baseline.pth    (if trained)
    models/{prefix}_hnn.pth         (if trained)
    {prefix}_hnn_run.json           (run metadata: losses, config, shapes)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange, tqdm

# --- Project import: your HNN class here ---
# Must expose baseline=True (direct dx) and baseline=False (Hamiltonian) behaviors
from utils.model_training.hnn import HNN  # <- your class


# ------------------------ small helpers ------------------------

def _parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(v) for v in s.split(",") if v.strip() != ""]


def _parse_hidden_dims(s: str):
    s = (s or "").strip()
    if not s:
        return 256
    if "," in s:
        return [int(v) for v in s.split(",") if v.strip() != ""]
    return int(s)


def build_loaders_from_tensor(
    data: torch.Tensor,
    x_cols: List[int],
    dx_cols: List[int],
    batch_size: int,
    split: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    """
    data: [N_traj, T, D]  -> flatten over (traj,time) and split into train/test loaders.
    """
    assert data.dim() == 3, f"Expected [N_traj, T, D], got {tuple(data.shape)}"
    N_traj, T, D = data.shape
    flat = data.view(-1, D)
    x = flat[:, x_cols]
    dx = flat[:, dx_cols]

    split_ix = int(len(x) * split)
    train_x, test_x = x[:split_ix], x[split_ix:]
    train_dx, test_dx = dx[:split_ix], dx[split_ix:]

    train_ds = TensorDataset(train_x, train_dx)
    test_ds  = TensorDataset(test_x,  test_dx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
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
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    MSE loss + Adam + LR on plateau + grad clip + early stopping.
    """
    model = model.to(device)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(opt, mode="min", factor=sched_factor, patience=sched_patience, verbose=verbose)

    stats = {"train_loss": [], "test_loss": [], "learning_rate": []}
    best = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in trange(1, epochs + 1, desc=f"Train({model.__class__.__name__})", disable=not verbose):
        # ---------- train ----------
        model.train()
        tr_losses = []
        for xb, db in train_loader:
            xb = xb.to(device).requires_grad_(True)  # HNN path needs input grads
            db = db.to(device)
            pred = model(xb)
            loss = crit(pred, db)

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
            for xb, db in test_loader:
                xb = xb.to(device)
                db = db.to(device)
                loss = crit(model(xb), db)
                te_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else np.nan
        te_loss = float(np.mean(te_losses)) if te_losses else np.nan
        stats["train_loss"].append(tr_loss)
        stats["test_loss"].append(te_loss)
        stats["learning_rate"].append(opt.param_groups[0]["lr"])

        if verbose:
            tqdm.write(f"Epoch {epoch:4d} | train {tr_loss:.3e} | test {te_loss:.3e} | lr {opt.param_groups[0]['lr']:.2e}")

        # LR scheduler + early stopping
        sched.step(te_loss)
        if te_loss + 1e-12 < best:
            best = te_loss
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                if verbose:
                    tqdm.write(f"[EarlyStop] Stopping at epoch {epoch}. Best test {best:.3e}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, stats


# ------------------------ CLI ------------------------

def main():
    ap = argparse.ArgumentParser(description="Train Baseline/HNN on a provided tensor (no plotting).")

    # Data
    ap.add_argument("--data-tensor", type=str, required=True, help="Path to torch tensor [N_traj, T, D].")
    ap.add_argument("--x-cols", type=str, default="0,1,2,3",
                    help="Comma-separated columns for x (e.g., theta1,theta2,p1,p2).")
    ap.add_argument("--dx-cols", type=str, default="9,10,11,12",
                    help="Comma-separated columns for dx (e.g., theta1_dot,theta2_dot,p1_dot,p2_dot).")
    ap.add_argument("--split", type=float, default=0.8, help="Train/test split fraction.")

    # Model + training knobs
    ap.add_argument("--hidden-dims", type=str, default="256", help='e.g., "256" or "128,256,256"')
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--nonlinearity", type=str, default="softplus")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--early-stop", type=int, default=20)
    ap.add_argument("--scheduler-factor", type=float, default=0.5)
    ap.add_argument("--scheduler-patience", type=int, default=10)

    # Which models to train
    ap.add_argument("--train-baseline", action="store_true", default=False, help="Train the direct dx regressor.")
    ap.add_argument("--train-hnn", action="store_true", default=True, help="Train the Hamiltonian model.")

    # Runtime / output
    ap.add_argument("--device", type=str, default=None, help="cuda | cpu | None(auto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--out-dir", type=str, default="results/double_pendulum")
    ap.add_argument("--save-prefix", type=str, default="hnn")

    args = ap.parse_args()

    # Seeds & device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print(f"[INFO] device: {device}")

    # IO setup
    out_dir = Path(args.out_dir).expanduser().resolve()
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load tensor
    tensor_path = Path(args.data_tensor).expanduser().resolve()
    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor not found: {tensor_path}")
    data = torch.load(str(tensor_path))
    if not (isinstance(data, torch.Tensor) and data.dim() == 3):
        raise ValueError(f"Expected torch.Tensor [N_traj, T, D]; got {type(data)} with shape {getattr(data, 'shape', None)}")

    # Column mapping
    x_cols  = _parse_int_list(args.x_cols)
    dx_cols = _parse_int_list(args.dx_cols)
    N_traj, T, D = data.shape
    if not x_cols or not dx_cols or max(x_cols + dx_cols) >= D:
        raise ValueError(f"Bad column indices. D={D}, x_cols={x_cols}, dx_cols={dx_cols}")

    # Data loaders
    train_loader, test_loader = build_loaders_from_tensor(
        data, x_cols, dx_cols, batch_size=args.batch_size, split=float(args.split)
    )

    # Train
    stats_all: Dict[str, Dict[str, List[float]]] = {}
    saved_paths: Dict[str, str] = {}

    n_elems = len(x_cols) // 2  # expect x=[q(n), p(n)]
    hidden_dims = _parse_hidden_dims(args.hidden_dims)

    if args.train_baseline:
        baseline = HNN(
            n_elements=n_elems,
            hidden_dims=hidden_dims,
            num_layers=args.num_layers,
            baseline=True,
            nonlinearity=args.nonlinearity,
        )
        baseline, stats = train_one_model(
            baseline, train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip, sched_factor=args.scheduler_factor,
            sched_patience=args.scheduler_patience, early_stop_patience=args.early_stop,
            verbose=args.verbose
        )
        stats_all["baseline"] = stats
        p = model_dir / f"{args.save_prefix}_baseline.pth"
        torch.save(baseline.state_dict(), p)
        saved_paths["baseline"] = str(p)

    if args.train_hnn:
        hnn = HNN(
            n_elements=n_elems,
            hidden_dims=hidden_dims,
            num_layers=args.num_layers,
            baseline=False,
            nonlinearity=args.nonlinearity,
        )
        hnn, stats = train_one_model(
            hnn, train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip, sched_factor=args.scheduler_factor,
            sched_patience=args.scheduler_patience, early_stop_patience=args.early_stop,
            verbose=args.verbose
        )
        stats_all["hnn"] = stats
        p = model_dir / f"{args.save_prefix}_hnn.pth"
        torch.save(hnn.state_dict(), p)
        saved_paths["hnn"] = str(p)

    # Manifest
    meta = {
        "data_tensor": str(tensor_path),
        "shape": [int(N_traj), int(T), int(D)],
        "x_cols": x_cols,
        "dx_cols": dx_cols,
        "split": float(args.split),
        "trained_models": list(stats_all.keys()),
        "save_paths": saved_paths,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "early_stop_patience": args.early_stop,
            "scheduler_factor": args.scheduler_factor,
            "scheduler_patience": args.scheduler_patience,
            "hidden_dims": args.hidden_dims,
            "num_layers": args.num_layers,
            "nonlinearity": args.nonlinearity,
            "device": str(device),
        },
        "final_losses": {k: {
            "best_train": float(min(v.get("train_loss", [np.nan]))),
            "best_test": float(min(v.get("test_loss", [np.nan]))),
        } for k, v in stats_all.items()},
    }
    (out_dir / f"{args.save_prefix}_hnn_run.json").write_text(json.dumps(meta, indent=2))

    # Summary
    print("\n[✓] Training complete.")
    print(f"    Out dir: {out_dir}")
    for k, p in saved_paths.items():
        print(f"    Saved {k}: {p}")
    for tag, s in stats_all.items():
        tr = s.get("train_loss", [])
        te = s.get("test_loss", [])
        if tr and te:
            print(f"    {tag}: best train={min(tr):.3e}  best test={min(te):.3e}")


if __name__ == "__main__":
    main()
