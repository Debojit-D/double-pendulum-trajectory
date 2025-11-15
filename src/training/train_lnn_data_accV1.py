#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a Lagrangian Neural Network (LNN) on double-pendulum trajectories.

Data expected:
- CSV columns:
    t,q1,q2,dq1,dq2,
    tip_x,tip_y,tip_z,
    tip_x_rel,tip_z_rel,
    elbow_x,elbow_z,
    step_idx,
    ddq1,ddq2

State / derivative:
    X    = [q1, q2, dq1, dq2]
    Xdot = [dq1, dq2, ddq1, ddq2]   (ddq from MuJoCo qacc, logged as ddq1,ddq2)

Training setup (all hard-coded):
- Use only the first 5 seconds of each trajectory.
- Downsample with a fixed stride for efficiency.
- Train an MLP LNN with Adam.
- Train in *normalized* space (zero-mean, unit-variance for each dim).
- Save trained params, config, metrics, and normalization stats to out_dir.

Outputs:
- out_dir/
    lnn_params.pkl      (trained JAX params tree)
    config.json
    metrics.json
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Progress bar
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it

import sys

# ----------------- JAX & Optax -----------------
import jax
import jax.numpy as jnp

try:
    import optax
except ImportError as e:
    raise ImportError(
        "optax is required for train_lnn.py. "
        "Install with: pip install optax"
    ) from e

# --- Make project root importable so we can import utils.model_training.lnn ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (…/double-pendulum-trajectory)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_training.lnn import mlp, raw_lagrangian_eom, custom_init  # noqa: E402

# ===================== HARD-CODED CONFIG =====================

DATA_DIR = Path(
    "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal2"
)
MANIFEST = DATA_DIR / "double_pendulum_manifest_ideal.json"

OUT_DIR = DATA_DIR / "lnn_modelV1"

# Use only first 5 seconds of each trajectory
T_MAX_TRAIN = 5.0

# Downsample stride inside each trajectory (for efficiency)
STRIDE = 5  # you can change to 2 / 10, etc.

# Training hyperparameters
BATCH_SIZE = 2048
N_EPOCHS = 200
LEARNING_RATE = 3e-4
VAL_SPLIT = 0.2  # 20% for validation

# Model hyperparameters
INPUT_DIM = 4   # [q1, q2, dq1, dq2]
HIDDEN_DIM = 512
OUTPUT_DIM = 1  # scalar Lagrangian
N_HIDDEN_LAYERS = 4
RNG_SEED = 0

# =============================================================

# Minimum required columns (ddq1,ddq2 checked later)
REQ_COLS = ["t", "q1", "q2", "dq1", "dq2"]


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _ensure_finite(arr: np.ndarray, name: str, ctx: str = "", dump_dir: Optional[Path] = None):
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
    # ddq1, ddq2 must exist for LNN training with MuJoCo accelerations
    for c in ("ddq1", "ddq2"):
        if c not in arr.dtype.names:
            raise ValueError(f"[DATA] Missing required column '{c}' (need MuJoCo accelerations) in {csv_path}")
    return {name: arr[name] for name in arr.dtype.names}


def _build_dataset(
    data_dir: Path,
    manifest: Optional[Path],
    t_max: float,
    stride: int,
    debug_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build full dataset (X, Xdot) by concatenating all trajectories:

        X    = [q1, q2, dq1, dq2]
        Xdot = [dq1, dq2, ddq1, ddq2]

    ddq1, ddq2 are read directly from CSV (MuJoCo qacc logs),
    NOT estimated via finite differencing or Savitzky–Golay.
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

    for p in tqdm(csv_paths, desc="Loading trajectories for LNN"):
        d = _read_csv(p)
        t = d["t"].astype(float)

        # Per-run clamp: only use first t_max seconds
        mask = t <= float(t_max)
        if not np.any(mask):
            continue

        t = t[mask]
        q = np.vstack([d["q1"][mask], d["q2"][mask]]).T
        dq = np.vstack([d["dq1"][mask], d["dq2"][mask]]).T
        ddq = np.vstack([d["ddq1"][mask], d["ddq2"][mask]]).T

        # Wrap angles for stability
        q = _wrap_to_pi(q)

        _ensure_finite(q, "q", ctx=str(p), dump_dir=debug_dir)
        _ensure_finite(dq, "dq", ctx=str(p), dump_dir=debug_dir)
        _ensure_finite(ddq, "ddq", ctx=str(p), dump_dir=debug_dir)

        X = np.hstack([q, dq])
        Xdot = np.hstack([dq, ddq])

        _ensure_finite(X, "X", ctx=str(p), dump_dir=debug_dir)
        _ensure_finite(Xdot, "Xdot", ctx=str(p), dump_dir=debug_dir)

        if stride > 1:
            sl = slice(0, None, stride)
            X = X[sl]
            Xdot = Xdot[sl]

        X_list.append(X)
        Xdot_list.append(Xdot)

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

    S = {
        "q1": stats(X[:, 0]),
        "q2": stats(X[:, 1]),
        "dq1": stats(X[:, 2]),
        "dq2": stats(X[:, 3]),
        "ddq1": stats(Xdot[:, 2]),
        "ddq2": stats(Xdot[:, 3]),
    }
    return S


# ------------------------------ LNN Model Wrappers ------------------------------

def build_lnn_model(
    rng_key: jax.Array,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_hidden_layers: int,
):
    """
    Build LNN MLP and initialize parameters with custom_init.

    Returns:
        params, apply_fun, lagrangian_fn, predict_derivative_fn
    """
    init_fun, apply_fun = mlp(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_hidden_layers=n_hidden_layers,
    )

    # stax.init_fun signature: init_fun(rng, input_shape) -> (output_shape, params)
    _, params = init_fun(rng_key, (-1, input_dim))
    params = custom_init(params, seed=RNG_SEED)

    def lagrangian_fn(q: jnp.ndarray, q_dot: jnp.ndarray) -> jnp.ndarray:
        """Scalar L(q, q_dot) from MLP."""
        inp = jnp.concatenate([q, q_dot], axis=-1)  # (..., 4)
        L = apply_fun(params, inp)
        return jnp.squeeze(L, axis=-1)

    def predict_derivative_fn(params_, X_batch: jnp.ndarray) -> jnp.ndarray:
        """
        Compute [dq, ddq] from LNN for a batch of states X_batch.

        X_batch: (B, 4) with [q1, q2, dq1, dq2]
        returns: (B, 4) with [dq1, dq2, ddq1, ddq2]
        """

        def lag_fn(q, q_dot):
            inp = jnp.concatenate([q, q_dot], axis=-1)
            L = apply_fun(params_, inp)
            return jnp.squeeze(L, axis=-1)

        def f_state(x):
            return raw_lagrangian_eom(lag_fn, x)

        return jax.vmap(f_state)(X_batch)

    return params, apply_fun, lagrangian_fn, predict_derivative_fn


# ------------------------------ Training Logic ------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    debug_dir = OUT_DIR / "debug_dump"

    print(f"[INFO] Data dir: {DATA_DIR}")
    print(f"[INFO] Manifest: {MANIFEST if MANIFEST.exists() else 'None (using glob)'}")
    print(f"[INFO] Out dir:   {OUT_DIR}")
    print(f"[INFO] Using only first {T_MAX_TRAIN:.2f}s of each trajectory, stride={STRIDE}")

    # ---------------------- Build dataset ----------------------
    X, Xdot = _build_dataset(
        data_dir=DATA_DIR,
        manifest=MANIFEST if MANIFEST.exists() else None,
        t_max=T_MAX_TRAIN,
        stride=STRIDE,
        debug_dir=debug_dir,
    )

    stats = _dataset_stats(X, Xdot)
    print("[INFO] Dataset stats:")
    for k, v in stats.items():
        print(f"  {k:5s}: min={v['min']:+.4e} max={v['max']:+.4e} mean={v['mean']:+.4e} std={v['std']:+.4e}")

    N = X.shape[0]
    N_val = int(VAL_SPLIT * N)
    N_train = N - N_val

    X_train, X_val = X[:N_train], X[N_train:]
    Xdot_train, Xdot_val = Xdot[:N_train], Xdot[N_train:]

    print(f"[INFO] Train size: {N_train}, Val size: {N_val}")

    # ---------------------- Normalization (computed on TRAIN ONLY) ----------------------
    # Avoid divide-by-zero by clamping tiny stds
    eps = 1e-8
    X_mean_np = X_train.mean(axis=0)
    X_std_np = X_train.std(axis=0)
    X_std_np = np.where(X_std_np < eps, 1.0, X_std_np)

    Y_mean_np = Xdot_train.mean(axis=0)
    Y_std_np = Xdot_train.std(axis=0)
    Y_std_np = np.where(Y_std_np < eps, 1.0, Y_std_np)

    print("[INFO] Normalization stats (train only):")
    print("  X_mean =", X_mean_np)
    print("  X_std  =", X_std_np)
    print("  Y_mean =", Y_mean_np)
    print("  Y_std  =", Y_std_np)

    # Convert to JAX arrays
    X_mean = jnp.array(X_mean_np)
    X_std = jnp.array(X_std_np)
    Y_mean = jnp.array(Y_mean_np)
    Y_std = jnp.array(Y_std_np)

    # Convert datasets to JAX and normalized form
    X_train_j = jnp.array(X_train)
    X_val_j = jnp.array(X_val)
    Xdot_train_j = jnp.array(Xdot_train)
    Xdot_val_j = jnp.array(Xdot_val)

    X_train_n = (X_train_j - X_mean) / X_std
    X_val_n = (X_val_j - X_mean) / X_std
    Xdot_train_n = (Xdot_train_j - Y_mean) / Y_std
    Xdot_val_n = (Xdot_val_j - Y_mean) / Y_std

    # ---------------------- Build LNN model ----------------------
    rng = jax.random.PRNGKey(RNG_SEED)
    params, apply_fun, lagrangian_fn, predict_derivative_fn = build_lnn_model(
        rng_key=rng,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_hidden_layers=N_HIDDEN_LAYERS,
    )

    # Loss function in NORMALIZED space
    def loss_fn(params_, Xn_b, Yn_b):
        """
        params_ : LNN parameters
        Xn_b    : normalized states (B, 4)
        Yn_b    : normalized derivatives (B, 4)

        We:
          1) un-normalize X before passing to dynamics,
          2) run LNN to get physical derivatives,
          3) normalize predictions,
          4) compute MSE in normalized space.
        """
        # Un-normalize state
        X_b = Xn_b * X_std + X_mean
        # Predict physical derivatives
        preds = predict_derivative_fn(params_, X_b)
        # Normalize derivatives
        preds_n = (preds - Y_mean) / Y_std
        return jnp.mean((preds_n - Yn_b) ** 2)

    # Validation helper (no grad)
    @jax.jit
    def val_loss_fn(params_):
        return loss_fn(params_, X_val_n, Xdot_val_n)

    # ---------------------- Optimizer ----------------------
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params_, opt_state_, Xn_b, Yn_b):
        loss, grads = jax.value_and_grad(loss_fn)(params_, Xn_b, Yn_b)
        updates, opt_state_ = optimizer.update(grads, opt_state_)
        params_ = optax.apply_updates(params_, updates)
        return params_, opt_state_, loss

    # ---------------------- Training loop ----------------------
    num_batches = max(1, N_train // BATCH_SIZE)
    indices = np.arange(N_train)

    history = {"train_loss": [], "val_loss": []}

    print(f"[INFO] Starting LNN training for {N_EPOCHS} epochs "
          f"(batch_size={BATCH_SIZE}, num_batches={num_batches})")
    print("[INFO] Loss values are MSE in normalized space (dimensionless).")

    for epoch in range(1, N_EPOCHS + 1):
        np.random.shuffle(indices)
        epoch_losses = []

        for b in range(num_batches):
            start = b * BATCH_SIZE
            end = start + BATCH_SIZE
            batch_idx = indices[start:end]

            Xn_b = X_train_n[batch_idx]
            Yn_b = Xdot_train_n[batch_idx]

            params, opt_state, batch_loss = train_step(params, opt_state, Xn_b, Yn_b)
            epoch_losses.append(float(batch_loss))

        train_loss_epoch = float(np.mean(epoch_losses))
        val_loss_epoch = float(val_loss_fn(params))

        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss_epoch)

        print(f"[EPOCH {epoch:03d}] train_loss={train_loss_epoch:.4e}  val_loss={val_loss_epoch:.4e}")

    # ---------------------- Final metrics (OOM-safe) ----------------------
    def eval_loss_batched(params_, Xn_full, Yn_full, batch_size_eval=1024):
        """
        Compute MSE loss over a large dataset in small batches to avoid GPU OOM.
        Operates entirely in normalized space.
        """
        n = Xn_full.shape[0]
        total_loss = 0.0
        total_count = 0
        for start in range(0, n, batch_size_eval):
            end = min(start + batch_size_eval, n)
            xb_n = Xn_full[start:end]
            yb_n = Yn_full[start:end]
            batch_loss = float(loss_fn(params_, xb_n, yb_n))
            batch_size_cur = end - start
            total_loss += batch_loss * batch_size_cur
            total_count += batch_size_cur
        return total_loss / max(total_count, 1)

    # Use batched evaluation for train loss; val loss reuse existing helper.
    final_train_loss = eval_loss_batched(params, X_train_n, Xdot_train_n, batch_size_eval=1024)
    final_val_loss = float(val_loss_fn(params))

    print(f"[INFO] Final train loss (batched, normalized): {final_train_loss:.4e}")
    print(f"[INFO] Final val loss (normalized):             {final_val_loss:.4e}")

    # ---------------------- Save artifacts ----------------------
    # Save params
    with open(OUT_DIR / "lnn_params.pkl", "wb") as f:
        pickle.dump(params, f)

    # Save config (including normalization stats)
    config = dict(
        data_dir=str(DATA_DIR),
        manifest=str(MANIFEST) if MANIFEST.exists() else None,
        t_max_train=T_MAX_TRAIN,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        val_split=VAL_SPLIT,
        model=dict(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_hidden_layers=N_HIDDEN_LAYERS,
        ),
        rng_seed=RNG_SEED,
        train_size=N_train,
        val_size=N_val,
        normalization=dict(
            X_mean=X_mean_np.tolist(),
            X_std=X_std_np.tolist(),
            Y_mean=Y_mean_np.tolist(),
            Y_std=Y_std_np.tolist(),
        ),
    )
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))

    # Save metrics
    metrics = dict(
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        history=history,
        dataset_stats=stats,
    )
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n[✓] LNN training complete. Saved to: {OUT_DIR}")
    print("[✓] Files: lnn_params.pkl, config.json, metrics.json")


if __name__ == "__main__":
    main()