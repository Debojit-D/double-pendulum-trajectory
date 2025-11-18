#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for the trained Lagrangian Neural Network (LNN) on double-pendulum data.

- Loads:
    lnn_params.pkl
    config.json     (for model + normalization + data paths)

- For a chosen subset of CSVs (via manifest), it:
    * Rebuilds X = [q1, q2, dq1, dq2], Xdot = [dq1, dq2, ddq1, ddq2]
      using recorded ddq if available, otherwise Savitzky–Golay.
    * Computes one-step derivative MSE in physical units.
    * Runs a rollout using a semi-implicit symplectic Euler integrator of the LNN dynamics:
          dq_{k+1} = dq_k + dt * ddq(q_k, dq_k)
          q_{k+1}  = q_k  + dt * dq_{k+1}
      and computes RMSE on q and dq versus ground truth.
    * Computes LNN-based energy along:
          - the ground-truth trajectory
          - the LNN rollout trajectory
      and saves them for analysis.
    * Saves metrics as CSV + JSON, timeseries plots, and per-run timeseries CSVs.

Outputs (in OUT_DIR):
    eval_lnn_metrics.csv
    summary_lnn.json
    <csvstem>_lnn_timeseries.png
    <csvstem>_lnn_timeseries.csv
    <csvstem>_lnn_energy.png
"""

from __future__ import annotations
from pathlib import Path
import json
import sys
import math

import numpy as np
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# ------------------------------------------------------------------
# CONFIG: CHANGE THESE PATHS IF NEEDED
# ------------------------------------------------------------------

# Point this to the model you just trained
MODEL_DIR = Path(
    "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal2/lnn_model_Final1_5sec"
)
CONFIG_PATH = MODEL_DIR / "config.json"
PARAMS_PATH = MODEL_DIR / "lnn_params.pkl"

# We will read DATA_DIR + MANIFEST from config.json, but you can override:
DATA_DIR_OVERRIDE = None  # e.g. Path(".../SampleIdeal2") or None to use config
MANIFEST_OVERRIDE = None  # or Path(".../double_pendulum_manifest_ideal.json")

OUT_DIR = MODEL_DIR / "eval_lnn3"

# How many runs to evaluate
EVAL_MODE = "first_k"   # "manifest_all" | "first_k" | "random_k"
K         = 5
SHUFFLE_SEED = 42

# Time clamp & stride
T_MAX_EVAL   = 10.0        # seconds; keep ~ training horizon for fair comparison
DECIM_STRIDE = 1           # 1 = use all samples

# Savitzky–Golay fallback (if ddq1/ddq2 not in CSV)
SAVGOL_WINDOW   = 51
SAVGOL_POLY     = 3

ANGLE_WRAP = True
ANGLE_IDX  = (0, 1)       # q1, q2
BURNIN_SEC = 0.0          # ignore first seconds in metrics if you want

# ------------------------------------------------------------------
# Make project root importable for utils.model_training.lnn
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_training.lnn import mlp, raw_lagrangian_eom  # noqa: E402

# ------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------

def _wrap_to_pi_np(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi] (NumPy version)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _robust_savgol_window(T: int, requested: int, poly: int) -> int:
    """Same logic as in training: get an odd window <= T-1."""
    if T < 3:
        raise ValueError(f"[SavGol] Too few samples: T={T}")
    win = min(requested, T - 1)
    win = max(win, max(5, poly + 2))
    if win % 2 == 0:
        win = win - 1 if win - 1 >= 5 else win + 1
    if win % 2 == 0:
        win += 1
    if win >= T:
        win = T - 1 if (T - 1) % 2 == 1 else T - 2
    if win < 5:
        win = 5
    return win


def _estimate_ddq_savgol(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Savitzky–Golay estimate of ddq when ddq is not in the CSV."""
    dt = float(np.mean(np.diff(t)))
    if not np.all(np.diff(t) > 0):
        raise ValueError("[TIME] t must be strictly increasing for SavGol differentiation.")
    w = _robust_savgol_window(len(t), SAVGOL_WINDOW, SAVGOL_POLY)
    ddq = savgol_filter(
        q, window_length=w, polyorder=max(3, SAVGOL_POLY),
        deriv=2, delta=dt, axis=0, mode="interp"
    )
    return ddq


def _collect_files(data_dir: Path, manifest: Path) -> list[Path]:
    """Gather CSV list from manifest."""
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    meta = json.loads(manifest.read_text())
    files = [data_dir / r["csv"] for r in meta.get("runs", [])]
    if not files:
        raise RuntimeError(f"No runs listed in manifest: {manifest}")
    return files


def _choose_files(data_dir: Path, manifest: Path) -> list[Path]:
    """Pick files according to EVAL_MODE."""
    all_files = _collect_files(data_dir, manifest)
    if EVAL_MODE == "manifest_all":
        return all_files

    if EVAL_MODE == "first_k":
        return all_files[:max(0, min(K, len(all_files)))]

    if EVAL_MODE == "random_k":
        rng = np.random.default_rng(SHUFFLE_SEED)
        if len(all_files) == 0:
            return []
        idx = rng.choice(len(all_files), size=min(K, len(all_files)), replace=False)
        return [all_files[i] for i in idx]

    # default
    return all_files


def _build_X_Xdot_from_csv_for_lnn(csv_path: Path, t_max: float, stride: int):
    """
    Build X, Xdot, t for ONE CSV.

    X    = [q1, q2, dq1, dq2]
    Xdot = [dq1, dq2, ddq1, ddq2]
    """
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    cols = arr.dtype.names

    if not all(c in cols for c in ("t", "q1", "q2", "dq1", "dq2")):
        raise ValueError(f"[EVAL] Missing required cols in {csv_path}")

    t = arr["t"].astype(float)
    mask = t <= float(t_max)
    if not np.any(mask):
        raise RuntimeError(f"[EVAL] No samples within t_max={t_max} in {csv_path}")

    t = t[mask]
    q = np.vstack([arr["q1"][mask], arr["q2"][mask]]).T
    dq = np.vstack([arr["dq1"][mask], arr["dq2"][mask]]).T

    # Wrap angles like training (if ANGLE_WRAP was used there)
    q = _wrap_to_pi_np(q)

    # ddq from CSV if present, else SavGol
    if "ddq1" in cols and "ddq2" in cols:
        ddq = np.vstack([arr["ddq1"][mask], arr["ddq2"][mask]]).T
    else:
        ddq = _estimate_ddq_savgol(q, t)

    # Downsample
    if stride > 1:
        sl = slice(0, None, stride)
        t = t[sl]
        q = q[sl]
        dq = dq[sl]
        ddq = ddq[sl]

    X = np.hstack([q, dq])
    Xdot = np.hstack([dq, ddq])
    return X, Xdot, t


def _save_metrics_csv(rows: list[list[object]], out_csv: Path):
    header = "csv,rmse_q_traj,rmse_dq_traj,deriv_mse,deriv_rmse_dq,deriv_rmse_ddq,steps"
    arr = np.array(rows, dtype=object)
    np.savetxt(out_csv, arr, fmt="%s", delimiter=",", header=header, comments="")


def _save_summary_json(rows: list[list[object]], out_json: Path):
    if not rows:
        out_json.write_text(json.dumps({"count": 0}, indent=2))
        return
    vals_q   = np.array([float(r[1]) for r in rows])
    vals_dq  = np.array([float(r[2]) for r in rows])
    vals_dm  = np.array([float(r[3]) for r in rows])
    vals_dm_dq  = np.array([float(r[4]) for r in rows])
    vals_dm_ddq = np.array([float(r[5]) for r in rows])

    summary = dict(
        count=len(rows),
        rmse_q_traj_mean=float(vals_q.mean()),
        rmse_q_traj_median=float(np.median(vals_q)),
        rmse_dq_traj_mean=float(vals_dq.mean()),
        rmse_dq_traj_median=float(np.median(vals_dq)),
        deriv_mse_mean=float(vals_dm.mean()),
        deriv_mse_median=float(np.median(vals_dm)),
        deriv_rmse_dq_mean=float(vals_dm_dq.mean()),
        deriv_rmse_dq_median=float(np.median(vals_dm_dq)),
        deriv_rmse_ddq_mean=float(vals_dm_ddq.mean()),
        deriv_rmse_ddq_median=float(np.median(vals_dm_ddq)),
    )
    out_json.write_text(json.dumps(summary, indent=2))


def _save_timeseries_csv(
    t: np.ndarray,
    true_q: np.ndarray,
    true_dq: np.ndarray,
    pred_q: np.ndarray,
    pred_dq: np.ndarray,
    energy_true: np.ndarray,
    energy_pred: np.ndarray,
    out_csv: Path,
):
    """
    Save per-run timeseries:
    t, q_true, dq_true, q_LNN, dq_LNN, E_true(LNN), E_pred(LNN)
    """
    data = np.column_stack(
        [
            t,
            true_q[:, 0],
            true_q[:, 1],
            true_dq[:, 0],
            true_dq[:, 1],
            pred_q[:, 0],
            pred_q[:, 1],
            pred_dq[:, 0],
            pred_dq[:, 1],
            energy_true,
            energy_pred,
        ]
    )
    header = (
        "t,"
        "q1_true,q2_true,"
        "dq1_true,dq2_true,"
        "q1_lnn,q2_lnn,"
        "dq1_lnn,dq2_lnn,"
        "E_true_LNN,E_pred_LNN"
    )
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load config + params
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}")
    cfg = json.loads(CONFIG_PATH.read_text())

    model_cfg = cfg["model"]
    norm_cfg  = cfg.get("normalization", None)

    # Recover data_dir & manifest
    if DATA_DIR_OVERRIDE is not None:
        data_dir = DATA_DIR_OVERRIDE
    else:
        data_dir = Path(cfg["data_dir"])

    if MANIFEST_OVERRIDE is not None:
        manifest = MANIFEST_OVERRIDE
    else:
        manifest = Path(cfg["manifest"]) if cfg.get("manifest") else data_dir / "double_pendulum_manifest_ideal.json"

    # Load normalization (stored for completeness; we evaluate in physical units)
    if norm_cfg is not None:
        X_mean = np.array(norm_cfg["X_mean"], dtype=float)
        X_std  = np.array(norm_cfg["X_std"], dtype=float)
        Y_mean = np.array(norm_cfg["Y_mean"], dtype=float)
        Y_std  = np.array(norm_cfg["Y_std"], dtype=float)
    else:
        X_mean = X_std = Y_mean = Y_std = None

    # Build apply_fun with same architecture as training
    init_fun, apply_fun = mlp(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        n_hidden_layers=int(model_cfg["n_hidden_layers"]),
    )

    # Load trained params
    import pickle
    with open(PARAMS_PATH, "rb") as f:
        params = pickle.load(f)

    # Lagrangian function using loaded params
    def lag_fn(q, q_dot):
        inp = jnp.concatenate([q, q_dot], axis=-1)
        L = apply_fun(params, inp)
        return jnp.squeeze(L, axis=-1)

    # Vector field f(x) = [dq, ddq]
    def f_state(x):
        return raw_lagrangian_eom(lag_fn, x)

    f_state_vmap = jax.vmap(f_state)

    # LNN energy: E(q, dq) = sum_i dq_i * dL/d(dq_i) - L
    def _energy_single(x):
        q = x[:2]
        q_dot = x[2:]

        def L_wrt_qdot(q_dot_local):
            return lag_fn(q, q_dot_local)

        dL_dqdot = jax.grad(L_wrt_qdot)(q_dot)
        return jnp.dot(dL_dqdot, q_dot) - lag_fn(q, q_dot)

    energy_vmap = jax.jit(jax.vmap(_energy_single, in_axes=(0,)))

    # Choose subset of files
    files = _choose_files(data_dir, manifest)
    if not files:
        print("[WARN] No files selected for LNN evaluation.")
        return

    print(f"[INFO] Evaluating LNN on {len(files)} runs.")
    rows = []

    for p in files:
        X, Xdot, t = _build_X_Xdot_from_csv_for_lnn(
            p, t_max=T_MAX_EVAL, stride=max(1, DECIM_STRIDE)
        )

        dt = float(np.mean(np.diff(t)))
        rel_t = t - t[0]
        k0 = int(np.searchsorted(rel_t, BURNIN_SEC)) if BURNIN_SEC > 0 else 0

        # ------ One-step derivative evaluation (physical units) ------
        X_j = jnp.array(X)
        preds = np.array(f_state_vmap(X_j))   # (N,4)

        deriv_mse = float(np.mean((preds[k0:] - Xdot[k0:]) ** 2))
        deriv_rmse_dq   = float(np.sqrt(np.mean((preds[k0:, :2] - Xdot[k0:, :2]) ** 2)))
        deriv_rmse_ddq  = float(np.sqrt(np.mean((preds[k0:, 2:] - Xdot[k0:, 2:]) ** 2)))

        # ------ Rollout evaluation (symplectic Euler integration) ------
        N = X.shape[0]
        Y = np.zeros_like(X)
        Y[0] = X[0]

        for i in range(1, N):
            # Previous state
            x_prev = Y[i - 1]
            q_prev  = x_prev[:2]
            dq_prev = x_prev[2:]

            # LNN dynamics at previous state: f_state(x) = [dq_dt, ddq_dt]
            f_val = np.array(f_state(jnp.array(x_prev)))
            # We only need ddq_dt for symplectic Euler
            ddq_dt = f_val[2:]

            # Semi-implicit (symplectic) Euler step:
            dq_new = dq_prev + dt * ddq_dt      # update velocity first
            q_new  = q_prev  + dt * dq_new      # then position with new velocity

            Y[i, :2] = q_new
            Y[i, 2:] = dq_new

        # Angle wrapping for q error
        true_q = _wrap_to_pi_np(X[:, :2]) if ANGLE_WRAP else X[:, :2]
        pred_q = _wrap_to_pi_np(Y[:, :2]) if ANGLE_WRAP else Y[:, :2]

        true_dq = X[:, 2:]
        pred_dq = Y[:, 2:]

        rmse_q_traj = float(np.sqrt(np.mean((pred_q[k0:] - true_q[k0:]) ** 2)))
        rmse_dq_traj = float(np.sqrt(np.mean((pred_dq[k0:] - true_dq[k0:]) ** 2)))

        # ------ LNN energy along true & rollout trajectories ------
        energy_true = np.array(energy_vmap(jnp.array(X)))
        energy_pred = np.array(energy_vmap(jnp.array(Y)))

        rows.append([
            p.name,
            rmse_q_traj,
            rmse_dq_traj,
            deriv_mse,
            deriv_rmse_dq,
            deriv_rmse_ddq,
            N,
        ])

        print(
            f"[EVAL LNN] {p.name}: "
            f"traj_RMSE(q)={rmse_q_traj:.4e}, "
            f"traj_RMSE(dq)={rmse_dq_traj:.4e}, "
            f"deriv_MSE={deriv_mse:.4e}"
        )

        # ------ Timeseries CSV (for later analysis) ------
        ts_csv_path = OUT_DIR / f"{p.stem}_lnn_timeseries.csv"
        _save_timeseries_csv(
            t=t,
            true_q=true_q,
            true_dq=true_dq,
            pred_q=pred_q,
            pred_dq=pred_dq,
            energy_true=energy_true,
            energy_pred=energy_pred,
            out_csv=ts_csv_path,
        )

        # ------ State plots ------
        fig = plt.figure(figsize=(11, 8))
        ax = plt.subplot(2, 2, 1)
        ax.plot(t, true_q[:, 0], label="q1 true")
        ax.plot(t, pred_q[:, 0], "--", label="q1 LNN")
        ax.set_title("q1"); ax.legend()

        ax = plt.subplot(2, 2, 2)
        ax.plot(t, true_q[:, 1], label="q2 true")
        ax.plot(t, pred_q[:, 1], "--", label="q2 LNN")
        ax.set_title("q2"); ax.legend()

        ax = plt.subplot(2, 2, 3)
        ax.plot(t, true_dq[:, 0], label="dq1 true")
        ax.plot(t, pred_dq[:, 0], "--", label="dq1 LNN")
        ax.set_title("dq1"); ax.legend()

        ax = plt.subplot(2, 2, 4)
        ax.plot(t, true_dq[:, 1], label="dq2 true")
        ax.plot(t, pred_dq[:, 1], "--", label="dq2 LNN")
        ax.set_title("dq2"); ax.legend()

        fig.suptitle(p.name)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{p.stem}_lnn_timeseries.png", dpi=160)
        plt.close(fig)

        # ------ Energy plot ------
        fig_e = plt.figure(figsize=(7, 4))
        ax_e = fig_e.add_subplot(1, 1, 1)
        ax_e.plot(t, energy_true, label="E (LNN on true traj)")
        ax_e.plot(t, energy_pred, "--", label="E (LNN rollout)")
        ax_e.set_xlabel("t [s]")
        ax_e.set_ylabel("Energy (LNN units)")
        ax_e.set_title(f"Energy – {p.name}")
        ax_e.legend()
        fig_e.tight_layout()
        fig_e.savefig(OUT_DIR / f"{p.stem}_lnn_energy.png", dpi=160)
        plt.close(fig_e)

    # Save metrics + summary
    _save_metrics_csv(rows, OUT_DIR / "eval_lnn_metrics.csv")
    _save_summary_json(rows, OUT_DIR / "summary_lnn.json")

    print(f"[✓] Wrote LNN evaluation summary: {OUT_DIR/'eval_lnn_metrics.csv'}")
    print(f"[✓] Wrote LNN aggregate summary:  {OUT_DIR/'summary_lnn.json'}")
    print(f"[✓] Per-run timeseries CSVs & plots saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()
