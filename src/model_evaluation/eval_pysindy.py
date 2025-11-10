#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a PySINDy SINDy model on a chosen double-pendulum CSV.

- Loads PySINDy model.pkl (+ metrics.json for knobs).
- Rebuilds X=[q1,q2,dq1,dq2], Xdot=[dq1,dq2,ddq1,ddq2] with Savitzky–Golay on unwrapped q.
- Simulates with guard-rails (wrap q, clip q/dq inside RHS), robust to early termination.
- Fits link lengths from GT tip and compares FK tip from predicted q.
- Prints RMSEs and shows plots.

Author: you :)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import solve_ivp
from joblib import load as joblib_load

# ===================== HARD-CODED PATHS / OPTIONS =====================
DATA_DIR  = Path("/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal1")
MANIFEST  = DATA_DIR / "double_pendulum_manifest_ideal.json"
MODEL_DIR = DATA_DIR / "pysindy_model"           # where train_pysindy.py saved artifacts
MODEL_PKL = MODEL_DIR / "model.pkl"               # PySINDy object
METRICS   = MODEL_DIR / "metrics.json"            # knobs saved at train time

# Choose a specific CSV; set None to auto-pick first from manifest
CHOSEN_CSV: Optional[str] = None  # e.g., "double_pendulum_traj_ideal_run0123.csv"

# Visualization limits
MAX_SECONDS: Optional[float] = 5.0    # compare first N seconds; None = full run
DECIM: int = 5                        # time decimation for eval/plots

# Extra sim knobs (used if metrics.json missing)
FALLBACK_SOLVERS = ["Radau", "BDF", "LSODA", "RK45"]
RTOL_DEFAULT, ATOL_DEFAULT, MAX_STEP_DEFAULT = 1e-4, 1e-6, 2e-2
ANGLE_WRAP_DEFAULT = True

# Guard-rails for RHS
CLIP_Q = np.pi      # |q| <= pi
CLIP_DQ = 50.0      # |dq| <= 50

# ===================== Data utilities =====================
REQ_COLS = ["t", "q1", "q2", "dq1", "dq2"]

def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2*np.pi) - np.pi

def _robust_savgol_window(T: int, requested: int, poly: int) -> int:
    if T < 6:
        raise ValueError(f"SavGol needs >=6 samples, got T={T}")
    win = min(requested, T - 1)
    win = max(win, max(5, poly + 2))
    if win % 2 == 0:
        win -= 1
    if win >= T:
        win = T - 1 if (T - 1) % 2 == 1 else T - 2
    if win % 2 == 0:
        win += 1
    return win

def _read_csv_cols(csv_path: Path) -> Dict[str, np.ndarray]:
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    for c in REQ_COLS:
        if c not in arr.dtype.names:
            raise ValueError(f"[DATA] Missing column '{c}' in {csv_path}")
    return {k: arr[k].astype(float) for k in arr.dtype.names}

def _ddq_from_q(q: np.ndarray, t: np.ndarray, win: int, poly: int) -> np.ndarray:
    dt = float(np.mean(np.diff(t)))
    win_rob = _robust_savgol_window(len(t), win, poly)
    return savgol_filter(q, window_length=win_rob, polyorder=poly,
                         deriv=2, delta=dt, axis=0, mode="interp")

def _build_run(csv_path: Path,
               angle_wrap: bool,
               savgol_window: int,
               savgol_polyorder: int,
               stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = _read_csv_cols(csv_path)
    t = d["t"]
    if np.any(np.diff(t) <= 0):
        i = int(np.where(np.diff(t) <= 0)[0][0])
        raise ValueError(f"[TIME] Non-monotonic or duplicate t at idx {i} in {csv_path}")

    q_raw = np.vstack([d["q1"], d["q2"]]).T
    dq    = np.vstack([d["dq1"], d["dq2"]]).T

    q_unwrap = np.unwrap(_wrap_to_pi(q_raw), axis=0) if angle_wrap else np.unwrap(q_raw, axis=0)
    ddq = _ddq_from_q(q_unwrap, t, savgol_window, max(3, savgol_polyorder))
    q_state = _wrap_to_pi(q_raw) if angle_wrap else q_raw

    X    = np.hstack([q_state, dq])
    Xdot = np.hstack([dq, ddq])
    if stride > 1:
        sl = slice(0, None, stride)
        return X[sl], Xdot[sl], t[sl]
    return X, Xdot, t

def _first_manifest_file(data_dir: Path, manifest: Path) -> Path:
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    meta = json.loads(manifest.read_text())
    runs = meta.get("runs", [])
    if not runs:
        raise RuntimeError(f"No runs in manifest: {manifest}")
    return data_dir / runs[0]["csv"]

# ===================== Tip FK helpers =====================
def _read_tip_and_angles(csv_path: Path):
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
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    z = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return x, z

# ===================== Simulation (with guard-rails) =====================
def _simulate_with_guards(model, x0: np.ndarray, t: np.ndarray,
                          rtol: float, atol: float, max_step: float,
                          solvers: List[str], angle_wrap: bool) -> Tuple[np.ndarray, str]:
    """
    Simulate using solve_ivp. Inside RHS:
      - wrap q to (-pi,pi] if angle_wrap,
      - clip |q| <= CLIP_Q, |dq| <= CLIP_DQ before calling model.predict.
    Falls back across solvers; raises if none succeed.
    Returns (y, method_used) with y shape (len(t), 4) trimmed to integrator's output.
    """
    def rhs(_t, x):
        x = x.copy()
        if angle_wrap:
            x[0:2] = _wrap_to_pi(x[0:2])
        x[0:2] = np.clip(x[0:2], -CLIP_Q, CLIP_Q)
        x[2:4] = np.clip(x[2:4], -CLIP_DQ, CLIP_DQ)
        return model.predict(x[np.newaxis, :])[0]

    for meth in solvers:
        try:
            sol = solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t,
                            method=meth, rtol=rtol, atol=atol, max_step=max_step)
            if not sol.success:
                # try next method
                continue
            return sol.y.T, meth
        except Exception:
            continue
    raise RuntimeError(f"All integrators failed: {solvers}")

# ===================== Main =====================
def main():
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"Missing PySINDy model: {MODEL_PKL}")
    model = joblib_load(MODEL_PKL)

    # Load sim + preprocessing knobs if available
    if METRICS.exists():
        M = json.loads(METRICS.read_text())
        angle_wrap = bool(M.get("angle_wrap", ANGLE_WRAP_DEFAULT))
        sav_win    = int(M.get("savgol_window", 51))
        sav_ord    = int(M.get("savgol_polyorder", 3))
        rtol       = float(M.get("rtol", RTOL_DEFAULT))
        atol       = float(M.get("atol", ATOL_DEFAULT))
        max_step   = float(M.get("max_step", MAX_STEP_DEFAULT))
        sim_stride = int(M.get("sim_stride", 1))
        solvers    = (M.get("integrators_tried") or FALLBACK_SOLVERS)
    else:
        angle_wrap = ANGLE_WRAP_DEFAULT
        sav_win, sav_ord = 51, 3
        rtol, atol, max_step = RTOL_DEFAULT, ATOL_DEFAULT, MAX_STEP_DEFAULT
        sim_stride = 1
        solvers = FALLBACK_SOLVERS

    # Pick CSV
    csv_path = DATA_DIR / CHOSEN_CSV if CHOSEN_CSV else _first_manifest_file(DATA_DIR, MANIFEST)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Build X / Xdot / t (and optionally decimate + cut window)
    Xh, Xdot_h, th = _build_run(csv_path, angle_wrap, sav_win, sav_ord, stride=max(1, DECIM))
    if MAX_SECONDS is not None:
        m = (th - th[0]) <= float(MAX_SECONDS)
        Xh, Xdot_h, th = Xh[m], Xdot_h[m], th[m]

    # Decimate for sim stability if metrics asked for it
    ss = max(1, int(sim_stride))
    th_sim = th[::ss]
    Xh_sim = Xh[::ss]
    Xdot_sim = Xdot_h[::ss]

    # Simulate with guard-rails
    x0 = Xh_sim[0]
    try:
        y, used = _simulate_with_guards(model, x0, th_sim, rtol, atol, max_step, solvers, angle_wrap)
    except Exception as e:
        # final desperate fallback: RK45 with looser tolerances
        y, used = _simulate_with_guards(model, x0, th_sim, 1e-3, 1e-6, max_step, ["RK45"], angle_wrap)

    # If integrator returned fewer points, trim GT to match
    e = min(len(y), len(Xh_sim))
    y = y[:e]
    Xh_e = Xh_sim[:e]
    Xdot_e = Xdot_sim[:e]
    th_e = th_sim[:e]

    # Periodic compare on q
    if angle_wrap:
        pred_q = _wrap_to_pi(y[:, :2])
        true_q = _wrap_to_pi(Xh_e[:, :2])
    else:
        pred_q = y[:, :2]
        true_q = Xh_e[:, :2]

    rmse_q1  = float(np.sqrt(np.mean((pred_q[:, 0] - true_q[:, 0])**2)))
    rmse_q2  = float(np.sqrt(np.mean((pred_q[:, 1] - true_q[:, 1])**2)))
    rmse_dq1 = float(np.sqrt(np.mean((y[:, 2] - Xh_e[:, 2])**2)))
    rmse_dq2 = float(np.sqrt(np.mean((y[:, 3] - Xh_e[:, 3])**2)))

    # Derivative MSE (one-step derivative fit)
    pred_dot = model.predict(Xh_e)
    deriv_mse = float(np.mean((pred_dot - Xdot_e)**2))

    print(f"[EVAL] {csv_path.name} | window={th_e[-1]-th_e[0]:.2f}s, steps={e}/{len(th_sim)}, solver={used}")
    print(f"  RMSE: q1={rmse_q1:.3e}, q2={rmse_q2:.3e}, dq1={rmse_dq1:.3e}, dq2={rmse_dq2:.3e}")
    print(f"  d/dt MSE: {deriv_mse:.3e}")

    # Fit FK link lengths on *ground-truth* (same window, same decim)
    t_raw, q1_raw, q2_raw, tipx_raw, tipz_raw = _read_tip_and_angles(csv_path)
    # match the exact indices used in th_sim (decim + window)
    base_sl = slice(0, None, max(1, DECIM))
    t_sel    = t_raw[base_sl]
    q1_sel   = q1_raw[base_sl]
    q2_sel   = q2_raw[base_sl]
    tipx_sel = tipx_raw[base_sl]
    tipz_sel = tipz_raw[base_sl]
    if MAX_SECONDS is not None:
        m2 = (t_sel - t_sel[0]) <= float(MAX_SECONDS)
        t_sel, q1_sel, q2_sel, tipx_sel, tipz_sel = t_sel[m2], q1_sel[m2], q2_sel[m2], tipx_sel[m2], tipz_sel[m2]
    # further decimate if sim_stride > 1 so lengths correspond to th_sim grid
    t_sel, q1_sel, q2_sel, tipx_sel, tipz_sel = t_sel[::ss], q1_sel[::ss], q2_sel[::ss], tipx_sel[::ss], tipz_sel[::ss]
    # and trim to e points to match y
    t_sel, q1_sel, q2_sel, tipx_sel, tipz_sel = t_sel[:e], q1_sel[:e], q2_sel[:e], tipx_sel[:e], tipz_sel[:e]

    L1, L2 = _fit_link_lengths(q1_sel, q2_sel, tipx_sel, tipz_sel)
    tipx_pred, tipz_pred = _fk_from_angles(pred_q[:, 0], pred_q[:, 1], L1, L2)

    # Tip RMSE
    rmse_tipx = float(np.sqrt(np.mean((tipx_pred - tipx_sel)**2)))
    rmse_tipz = float(np.sqrt(np.mean((tipz_pred - tipz_sel)**2)))
    print(f"  Tip RMSE: x_rel={rmse_tipx:.3e}, z_rel={rmse_tipz:.3e}  (L1={L1:.3f}, L2={L2:.3f})")

    # ===================== PLOTS =====================
    fig = plt.figure(figsize=(12, 9))

    ax = plt.subplot(3, 2, 1)
    ax.plot(th_e, true_q[:, 0], label="q1 true")
    ax.plot(th_e, pred_q[:, 0], "--", label="q1 pred")
    ax.set_title("q1"); ax.set_xlabel("t [s]"); ax.legend()

    ax = plt.subplot(3, 2, 2)
    ax.plot(th_e, true_q[:, 1], label="q2 true")
    ax.plot(th_e, pred_q[:, 1], "--", label="q2 pred")
    ax.set_title("q2"); ax.set_xlabel("t [s]"); ax.legend()

    ax = plt.subplot(3, 2, 3)
    ax.plot(th_e, Xh_e[:, 2], label="dq1 true")
    ax.plot(th_e, y[:, 2], "--", label="dq1 pred")
    ax.set_title("dq1"); ax.set_xlabel("t [s]"); ax.legend()

    ax = plt.subplot(3, 2, 4)
    ax.plot(th_e, Xh_e[:, 3], label="dq2 true")
    ax.plot(th_e, y[:, 3], "--", label="dq2 pred")
    ax.set_title("dq2"); ax.set_xlabel("t [s]"); ax.legend()

    ax = plt.subplot(3, 2, 5)
    ax.plot(t_sel, tipx_sel, label="tip_x_rel true")
    ax.plot(th_e, tipx_pred, "--", label="tip_x_rel pred (FK)")
    ax.set_title(f"End-effector x_rel (L1={L1:.3f}, L2={L2:.3f})")
    ax.set_xlabel("t [s]"); ax.legend()

    ax = plt.subplot(3, 2, 6)
    ax.plot(t_sel, tipz_sel, label="tip_z_rel true")
    ax.plot(th_e, tipz_pred, "--", label="tip_z_rel pred (FK)")
    ax.set_title("End-effector z_rel")
    ax.set_xlabel("t [s]"); ax.legend()

    fig.suptitle(f"{csv_path.name} | window {th_e[-1]-th_e[0]:.2f}s | decim={DECIM} | solver={used}")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
