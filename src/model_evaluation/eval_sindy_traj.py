#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick visualize SINDy rollout + end-effector FK (double pendulum).

Fixes:
- Reconstructs the exact feature library by PARSING saved feature_names.txt
  (supports 1, q1,q2,dq1,dq2, powers ^k, products '*', and sin()/cos() of
   q1, q2, q1-q2, q1+q2).
- Aligns coef.npy to (n_features, n_states) automatically.
- Monkey-patches model.predict_derivative to use the parsed Θ, so
  model.simulate(...) works as-is (with your guard rails).

Outputs:
- On-screen plots for q, dq and FK tip (x_rel, z_rel)
- Prints RMSE on the shown window
"""

from __future__ import annotations
from pathlib import Path
import sys, json, types
import numpy as np
import matplotlib.pyplot as plt

# ===================== HARD-CODED PATHS / OPTIONS =====================
MODEL_DIR  = Path("/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal1/sindy_model")
DATA_DIR   = Path("/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal1")
MANIFEST   = Path("/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal1/double_pendulum_manifest_ideal.json")

# Choose a specific CSV by filename; set to None to auto-pick first from manifest
CHOSEN_CSV = None  # e.g., "double_pendulum_traj_ideal_run0123.csv"

# Timeline limits / speed knobs
MAX_SECONDS = 5.0   # Compare only first N seconds (set None for full)
DECIM       = 5     # Decimate time series (1 = no decimation)

# Simulation settings (kept aligned with training eval)
SIM_METHOD   = "Radau"   # ["RK45","Radau","BDF","LSODA"]
SIM_RTOL     = 1e-4
SIM_ATOL     = 1e-6
SIM_MAX_STEP = 0.02

# Guard-rails for simulate (to avoid blow-ups)
WRAP_ANGLES_DURING_SIM = True
CLIP_Q   = np.pi    # clip |q| <= pi
CLIP_DQ  = 50.0     # clip |dq| <= 50 rad/s

# ===================== LOAD TRAINING CONFIG ONCE =====================
CFG = json.loads((MODEL_DIR / "config.json").read_text())
SAVGOL_WIN   = int(CFG["savgol_window"])
SAVGOL_ORDER = int(CFG["savgol_polyorder"])
# Even if training did not save angle_wrap/angle_idx, set sensible defaults:
ANGLE_WRAP   = bool(CFG.get("angle_wrap", True))
ANGLE_IDX    = tuple(CFG.get("angle_idx", [0, 1]))

# --- Make project root importable so we can reuse helpers ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_training.sindy import SINDyRegressor, SINDyConfig
from src.training.train_sindy import _build_X_Xdot_from_csv, _wrap_to_pi

# ---------------------------
# Utilities
# ---------------------------
def _first_manifest_file(data_dir: Path, manifest: Path) -> Path:
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    meta = json.loads(manifest.read_text())
    runs = meta.get("runs", [])
    if not runs:
        raise RuntimeError(f"No runs in manifest: {manifest}")
    return data_dir / runs[0]["csv"]

def _load_model(model_dir: Path) -> SINDyRegressor:
    coef  = np.load(model_dir / "coef.npy")
    names = (model_dir / "feature_names.txt").read_text().strip().splitlines()

    cfg = SINDyConfig(
        poly_degree=int(CFG["poly_degree"]),
        trig_harmonics=int(CFG["trig_harmonics"]),
        threshold_lambda=float(CFG["threshold_lambda"]),
        max_stlsq_iter=int(CFG["max_stlsq_iter"]),
        use_savgol_for_xdot=False,
        savgol_window=SAVGOL_WIN,
        savgol_polyorder=SAVGOL_ORDER,
        normalize_columns=bool(CFG["normalize_columns"]),
        include_bias=bool(CFG["include_bias"]),
        backend="numpy",
        device=None,
    )
    model = SINDyRegressor(cfg)
    model.feature_names_ = names
    model.n_state_ = 4

    # Align coef to shape (n_features, n_states)
    coef = np.asarray(coef)
    n_feat = len(names)
    n_state = 4
    if coef.ndim == 1:
        # try to reshape (n_feat, n_state) if flat
        if coef.size == n_feat * n_state:
            coef = coef.reshape(n_feat, n_state)
        elif coef.size == n_feat:
            coef = coef.reshape(n_feat, 1)  # degenerate single-column model
        else:
            raise ValueError(f"[coef.npy] Unexpected 1D size {coef.size}; cannot align with n_feat={n_feat}.")
    # transpose if saved as (n_state, n_feat)
    if coef.shape[0] != n_feat and coef.shape[1] == n_feat:
        coef = coef.T
    if coef.shape[0] != n_feat:
        raise ValueError(f"[coef.npy] Shape mismatch after alignment: got {coef.shape}, want ({n_feat}, {n_state}).")

    model.coef_ = coef
    return model

def _read_tip_and_angles(csv_path: Path):
    """Read t, q1, q2, tip_x_rel, tip_z_rel as float arrays."""
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    req = ["t", "q1", "q2", "tip_x_rel", "tip_z_rel"]
    for k in req:
        if k not in arr.dtype.names:
            raise ValueError(f"[DATA] Missing '{k}' in {csv_path}")
    return (
        arr["t"].astype(float),
        arr["q1"].astype(float),
        arr["q2"].astype(float),
        arr["tip_x_rel"].astype(float),
        arr["tip_z_rel"].astype(float),
    )

def _fit_link_lengths(q1: np.ndarray, q2: np.ndarray,
                      x_rel: np.ndarray, z_rel: np.ndarray) -> tuple[float, float]:
    """Fit planar L1, L2 from (q1,q2)->(x,z) by linear least-squares."""
    c1, s1   = np.cos(q1), np.sin(q1)
    c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
    A = np.empty((2 * len(q1), 2))
    A[0::2, 0] = c1;  A[0::2, 1] = c12
    A[1::2, 0] = s1;  A[1::2, 1] = s12
    b = np.empty((2 * len(q1),))
    b[0::2] = x_rel;  b[1::2] = z_rel
    L, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(L[0]), float(L[1])

def _fk_from_angles(q1: np.ndarray, q2: np.ndarray, L1: float, L2: float):
    x = L1 * np.cos(q1)       + L2 * np.cos(q1 + q2)
    z = L1 * np.sin(q1)       + L2 * np.sin(q1 + q2)
    return x, z

# --------- Feature-name parser to rebuild Θ(X) exactly like training ----------
def _build_theta_from_names(X: np.ndarray, names: list[str]) -> np.ndarray:
    """
    Construct Θ(X) by parsing feature_names.
    Supported atoms: 1, q1, q2, dq1, dq2, sin(arg), cos(arg) with arg in {q1,q2,q1-q2,q1+q2}.
    Supported ops: product '*' and powers '^k' on q*, dq* (and also on sin/cos if present).
    """
    X = np.asarray(X, dtype=float)
    assert X.ndim == 2 and X.shape[1] == 4, "X must be (N,4) with columns [q1,q2,dq1,dq2]"
    q1 = X[:, 0]; q2 = X[:, 1]; dq1 = X[:, 2]; dq2 = X[:, 3]

    # Cache common trig args
    ang_cache = {
        "q1": q1,
        "q2": q2,
        "q1-q2": q1 - q2,
        "q1+q2": q1 + q2,
    }

    def _pow(arr: np.ndarray, k: int) -> np.ndarray:
        return arr if k == 1 else (arr ** k)

    def _parse_atom(token: str) -> np.ndarray:
        """Return column vector for a single atom (with optional ^k)."""
        if token == "1":
            return np.ones_like(q1)

        # split power
        if "^" in token:
            base, kstr = token.split("^", 1)
            k = int(kstr)
        else:
            base, k = token, 1

        # base variables
        if base == "q1":
            return _pow(q1, k)
        if base == "q2":
            return _pow(q2, k)
        if base == "dq1":
            return _pow(dq1, k)
        if base == "dq2":
            return _pow(dq2, k)

        # trig functions
        if base.startswith("sin(") and base.endswith(")"):
            arg = base[4:-1].strip()
            if arg not in ang_cache:
                raise ValueError(f"Unsupported sin() argument: {arg}")
            val = np.sin(ang_cache[arg])
            return _pow(val, k)
        if base.startswith("cos(") and base.endswith(")"):
            arg = base[4:-1].strip()
            if arg not in ang_cache:
                raise ValueError(f"Unsupported cos() argument: {arg}")
            val = np.cos(ang_cache[arg])
            return _pow(val, k)

        raise ValueError(f"Unrecognized feature atom: '{token}'")

    def _parse_feature(name: str) -> np.ndarray:
        # features are products of atoms separated by '*'
        tokens = [t.strip() for t in name.split("*")]
        col = np.ones_like(q1)
        for tok in tokens:
            if tok == "":
                continue
            col = col * _parse_atom(tok)
        return col

    # Build Θ
    cols = []
    for nm in names:
        cols.append(_parse_feature(nm))
    Theta = np.stack(cols, axis=1)  # (N, n_features)
    return Theta

# ---------------------------
# Main
# ---------------------------
def main():
    model = _load_model(MODEL_DIR)

    # Monkey-patch model.predict_derivative to use our parsed Θ(X)
    def _predict_derivative_compat(self: SINDyRegressor, X: np.ndarray) -> np.ndarray:
        Theta = _build_theta_from_names(np.asarray(X, dtype=float), self.feature_names_)
        Y = Theta @ self.coef_
        if not np.isfinite(Y).all():
            raise RuntimeError("predict_derivative produced non-finite values (check inputs / library overflow).")
        return Y
    model.predict_derivative = types.MethodType(_predict_derivative_compat, model)

    # Pick CSV
    csv_path = DATA_DIR / CHOSEN_CSV if CHOSEN_CSV else _first_manifest_file(DATA_DIR, MANIFEST)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Build state/derivative using the same helper as training (and same decimation)
    Xh, Xdot_h, th = _build_X_Xdot_from_csv(
        csv_path,
        angle_wrap=ANGLE_WRAP,
        savgol_window=SAVGOL_WIN,
        savgol_polyorder=SAVGOL_ORDER,
        stride=max(1, int(DECIM)),
        debug_dump=None
    )

    # Also load raw tip & angles for FK fitting/plotting; apply same stride
    t_raw, q1_raw, q2_raw, tipx_raw, tipz_raw = _read_tip_and_angles(csv_path)
    sl = slice(0, None, max(1, int(DECIM)))
    t_sel    = t_raw[sl]
    q1_sel   = q1_raw[sl]
    q2_sel   = q2_raw[sl]
    tipx_sel = tipx_raw[sl]
    tipz_sel = tipz_raw[sl]

    # Limit to first MAX_SECONDS if requested
    if MAX_SECONDS is not None:
        m = (t_sel - t_sel[0]) <= float(MAX_SECONDS)
        th       = th[m]
        Xh       = Xh[m]
        Xdot_h   = Xdot_h[m]
        t_sel    = t_sel[m]
        q1_sel   = q1_sel[m]
        q2_sel   = q2_sel[m]
        tipx_sel = tipx_sel[m]
        tipz_sel = tipz_sel[m]

    # Sanity: make sure Θ row count matches feature count for a single point
    _ = _build_theta_from_names(Xh[0:1, :], model.feature_names_)  # raises if incompatible

    # Simulate from first state with guard-rails
    x0 = Xh[0]
    try:
        y = model.simulate(
            x0, th,
            method=SIM_METHOD, rtol=SIM_RTOL, atol=SIM_ATOL,
            max_step=SIM_MAX_STEP,
            wrap_angles=True,
            angle_idx=ANGLE_IDX,
            clip_q=CLIP_Q,
            clip_dq=CLIP_DQ,
        )
    except Exception:
        # fallback in case stiff solver complains
        y = model.simulate(
            x0, th,
            method="RK45", rtol=SIM_RTOL, atol=SIM_ATOL,
            max_step=SIM_MAX_STEP,
            wrap_angles=True,
            angle_idx=ANGLE_IDX,
            clip_q=CLIP_Q,
            clip_dq=CLIP_DQ,
        )

    # Wrap angles for display (periodic)
    if ANGLE_WRAP:
        true_q = _wrap_to_pi(Xh[:, :2])
        pred_q = _wrap_to_pi(y[:, :2])
    else:
        true_q = Xh[:, :2]
        pred_q = y[:, :2]

    # Quick RMSE over shown window
    rmse_q1  = float(np.sqrt(np.mean((pred_q[:, 0] - true_q[:, 0])**2)))
    rmse_q2  = float(np.sqrt(np.mean((pred_q[:, 1] - true_q[:, 1])**2)))
    rmse_dq1 = float(np.sqrt(np.mean((y[:, 2] - Xh[:, 2])**2)))
    rmse_dq2 = float(np.sqrt(np.mean((y[:, 3] - Xh[:, 3])**2)))
    print(f"RMSE over window ({th[-1]-th[0]:.2f}s): "
          f"q1={rmse_q1:.3e}, q2={rmse_q2:.3e}, dq1={rmse_dq1:.3e}, dq2={rmse_dq2:.3e}")

    # Fit link lengths from TRUE angles and TRUE tip (relative) on this same window
    L1, L2 = _fit_link_lengths(q1_sel, q2_sel, tipx_sel, tipz_sel)

    # FK from PREDICTED angles
    tipx_pred, tipz_pred = _fk_from_angles(pred_q[:, 0], pred_q[:, 1], L1, L2)

    # --------- PLOTS ----------
    fig = plt.figure(figsize=(12, 9))

    ax = plt.subplot(3, 2, 1)
    ax.plot(th, true_q[:, 0], label="q1 true")
    ax.plot(th, pred_q[:, 0], "--", label="q1 pred")
    ax.set_title("q1"); ax.set_xlabel("t [s]"); ax.legend()

    ax = plt.subplot(3, 2, 2)
    ax.plot(th, true_q[:, 1], label="q2 true")
    ax.plot(th, pred_q[:, 1], "--", label="q2 pred")
    ax.set_title("q2"); ax.set_xlabel("t [s]"); ax.legend()

    ax = plt.subplot(3, 2, 3)
    ax.plot(th, Xh[:, 2], label="dq1 true")
    ax.plot(th, y[:, 2], "--", label="dq1 pred")
    ax.set_title("dq1"); ax.set_xlabel("t [s]"); ax.legend()

    ax = plt.subplot(3, 2, 4)
    ax.plot(th, Xh[:, 3], label="dq2 true")
    ax.plot(th, y[:, 3], "--", label="dq2 pred")
    ax.set_title("dq2"); ax.set_xlabel("t [s]"); ax.legend()

    # End-effector (x_rel, z_rel) over time
    ax = plt.subplot(3, 2, 5)
    ax.plot(t_sel, tipx_sel, label="tip_x_rel true")
    ax.plot(th,     tipx_pred, "--", label="tip_x_rel pred (FK)")
    ax.set_title(f"End-effector x_rel (L1={L1:.3f}, L2={L2:.3f})")
    ax.set_xlabel("t [s]"); ax.legend()

    ax = plt.subplot(3, 2, 6)
    ax.plot(t_sel, tipz_sel, label="tip_z_rel true")
    ax.plot(th,     tipz_pred, "--", label="tip_z_rel pred (FK)")
    ax.set_title("End-effector z_rel")
    ax.set_xlabel("t [s]"); ax.legend()

    fig.suptitle(f"Run: {csv_path.name}  |  window: {(th[-1]-th[0]):.2f}s, "
                 f"decim={DECIM}, solver={SIM_METHOD}")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
