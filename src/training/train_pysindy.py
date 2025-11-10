#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySINDy training for double pendulum (works with your local pysindy.SINDy class).

State  x    = [q1, q2, dq1, dq2]
Target xdot = [dq1, dq2, ddq1, ddq2]  (ddq via Savitzky–Golay on unwrapped q)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ
from joblib import dump as joblib_dump

# =========================
# CONFIG (edit if needed)
# =========================
DATA_DIR = Path("/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal1")
MANIFEST = Path("/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal1/double_pendulum_manifest_ideal.json")
OUT_DIR  = DATA_DIR / "pysindy_model"

# Preprocessing
ANGLE_WRAP       = False
SAVGOL_WINDOW    = 51
SAVGOL_POLYORDER = 3
STRIDE           = 2
HOLDOUT_RUNS     = 2  # last K CSVs as hold-out

# Library / optimizer
POLY_DEGREE      = 3
INCLUDE_INTERACT = True
INCLUDE_BIAS     = True   # set False if you want to remove constant "1" term

# STLSQ params and optional sweep
STLSQ_THRESHOLD  = 1e-5
STLSQ_MAX_ITER   = 10
STLSQ_RIDGE      = 0.0
LAMBDA_SWEEP     = "1e-5,3e-5,1e-4,3e-4,1e-3"   # "" to disable sweep

# Kinematic pinning: enforce q̇ = c * dq (for q1,q2 rows)
PIN_KINEMATICS   = True
PIN_COEFF        = (1.0, 1.0)

# Held-out rollout (stiff-friendly defaults)
SIM_STRIDE       = 1         # e.g., 5 to decimate held-out times
RTOL             = 1e-4
ATOL             = 1e-6
MAX_STEP         = 2e-2
INTEGRATOR_TRY   = ["Radau", "BDF", "LSODA"]  # fallback order

# =========================
# Utilities
# =========================
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
    if win < 5:
        win = 5
    if win >= T:
        win = T - 1 if (T - 1) % 2 == 1 else T - 2
    if win % 2 == 0:
        win += 1
    return win

def _ddq_from_q(q: np.ndarray, t: np.ndarray, win: int, poly: int) -> np.ndarray:
    dt = float(np.mean(np.diff(t)))
    win_rob = _robust_savgol_window(len(t), win, poly)
    ddq = savgol_filter(q, window_length=win_rob, polyorder=poly,
                        deriv=2, delta=dt, axis=0, mode="interp")
    return ddq

def _read_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    for c in REQ_COLS:
        if c not in arr.dtype.names:
            raise ValueError(f"[DATA] Missing column '{c}' in {csv_path}")
    return {k: arr[k] for k in arr.dtype.names}

def _build_run(csv_path: Path,
               angle_wrap: bool,
               savgol_window: int,
               savgol_polyorder: int,
               stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = _read_csv(csv_path)
    t = d["t"].astype(float)

    if np.any(np.diff(t) <= 0):
        i = int(np.where(np.diff(t) <= 0)[0][0])
        raise ValueError(f"[TIME] Non-monotonic or duplicate t at idx {i} in {csv_path}")

    q_raw = np.vstack([d["q1"], d["q2"]]).T
    dq    = np.vstack([d["dq1"], d["dq2"]]).T

    # unwrap q for stable ddq estimate
    q_for_ddq = np.unwrap(_wrap_to_pi(q_raw) if angle_wrap else q_raw, axis=0)
    ddq = _ddq_from_q(q_for_ddq, t, savgol_window, max(3, savgol_polyorder))

    q_state = _wrap_to_pi(q_raw) if angle_wrap else q_raw
    X    = np.hstack([q_state, dq])
    Xdot = np.hstack([dq, ddq])

    if stride > 1:
        sl = slice(0, None, stride)
        return X[sl], Xdot[sl], t[sl]
    return X, Xdot, t

def _load_dataset(
    data_dir: Path,
    manifest: Optional[Path],
    angle_wrap: bool,
    savgol_window: int,
    savgol_polyorder: int,
    stride: int,
    holdout_runs: int,
):
    if manifest and manifest.exists():
        meta = json.loads(manifest.read_text())
        csvs = [data_dir / r["csv"] for r in meta.get("runs", [])]
    else:
        csvs = sorted(data_dir.glob("double_pendulum_traj_*.csv"))
    if not csvs:
        raise ValueError(f"No CSVs found in {data_dir}")

    heldout_paths = csvs[-holdout_runs:] if holdout_runs > 0 else []
    train_paths   = csvs[:-holdout_runs] if holdout_runs > 0 else csvs

    Xs, Xdots, Ts = [], [], []
    for p in tqdm(train_paths, desc="Loading training CSVs"):
        X, Xdot, t = _build_run(p, angle_wrap, savgol_window, savgol_polyorder, stride)
        Xs.append(X); Xdots.append(Xdot); Ts.append(t)

    return Xs, Xdots, Ts, train_paths, heldout_paths

def _apply_kinematic_pin(model: ps.SINDy,
                         input_names: List[str],
                         pin_coeff=(1.0, 1.0)) -> None:
    """
    Enforce exact kinematics by editing optimizer.coef_:
      row 0 (d/dt q1): set to c1 at feature 'dq1'
      row 1 (d/dt q2): set to c2 at feature 'dq2'
    """
    lib_feat_names = model.get_feature_names()
    try:
        i_dq1 = lib_feat_names.index("dq1")
        i_dq2 = lib_feat_names.index("dq2")
    except ValueError:
        return

    Xi = np.array(model.optimizer.coef_, dtype=float, copy=True)
    Xi[0, :] = 0.0
    Xi[1, :] = 0.0
    Xi[0, i_dq1] = float(pin_coeff[0])
    Xi[1, i_dq2] = float(pin_coeff[1])
    model.optimizer.coef_ = Xi

def _pretty_equations(model: ps.SINDy,
                      input_names: List[str],
                      precision: int = 6) -> str:
    lib_feat_names = model.get_feature_names()
    Xi = np.array(model.optimizer.coef_, dtype=float)
    lines = []
    for k, sname in enumerate(input_names):
        terms = []
        for j, c in enumerate(Xi[k, :]):
            if c != 0.0:
                terms.append(f"{c:+.{precision}e}*{lib_feat_names[j]}")
        rhs = " + ".join(terms) if terms else "0"
        lines.append(f"d/dt {sname} = {rhs}")
    return "\n".join(lines)

def _split_train_val_lists(
    X_list: List[np.ndarray],
    Xdot_list: List[np.ndarray],
    t_list: List[np.ndarray],
    val_frac: float = 0.2,
):
    X_tr, Xd_tr, t_tr = [], [], []
    X_va, Xd_va, t_va = [], [], []
    for X, Xd, t in zip(X_list, Xdot_list, t_list, strict=True):
        n = len(t)
        k = max(1, int((1.0 - val_frac) * n))
        X_tr.append(X[:k]);     Xd_tr.append(Xd[:k]);     t_tr.append(t[:k])
        X_va.append(X[k:]);     Xd_va.append(Xd[k:]);     t_va.append(t[k:])
    return (X_tr, Xd_tr, t_tr), (X_va, Xd_va, t_va)

def _deriv_mse_on_lists(model: ps.SINDy,
                        X_val_list: List[np.ndarray],
                        Xdot_val_list: List[np.ndarray]) -> float:
    num, den = 0.0, 0
    for Xv, Xdv in zip(X_val_list, Xdot_val_list, strict=True):
        if len(Xv) == 0:
            continue
        pred = model.predict(Xv)
        diff = pred - Xdv
        num += float(np.sum(diff * diff))
        den += diff.size
    return num / max(den, 1)

# =========================
# Main
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_feature_names = ["q1", "q2", "dq1", "dq2"]

    # 1) Load dataset
    X_list, Xdot_list, t_list, train_paths, heldout_paths = _load_dataset(
        DATA_DIR, MANIFEST if MANIFEST.exists() else None,
        ANGLE_WRAP, SAVGOL_WINDOW, SAVGOL_POLYORDER, STRIDE, HOLDOUT_RUNS
    )

    # 2) Build library & model factory
    lib = PolynomialLibrary(
        degree=int(POLY_DEGREE),
        include_interaction=bool(INCLUDE_INTERACT),
        include_bias=bool(INCLUDE_BIAS),
    )
    def make_model(thr: float) -> ps.SINDy:
        opt = STLSQ(threshold=float(thr), max_iter=int(STLSQ_MAX_ITER), alpha=float(STLSQ_RIDGE))
        return ps.SINDy(feature_library=lib, optimizer=opt, discrete_time=False)

    # 3) Threshold sweep (each trajectory kept separate → strictly increasing t per traj)
    chosen_thr = float(STLSQ_THRESHOLD)
    if str(LAMBDA_SWEEP).strip():
        sweep_vals = [float(s) for s in str(LAMBDA_SWEEP).split(",")]
        (X_tr_l, Xd_tr_l, t_tr_l), (X_va_l, Xd_va_l, t_va_l) = _split_train_val_lists(
            X_list, Xdot_list, t_list, val_frac=0.2
        )
        val_mse = []
        for lam in tqdm(sweep_vals, desc="Lambda sweep (derivative MSE)"):
            m = make_model(lam)
            m.fit(X_tr_l, t=t_tr_l, x_dot=Xd_tr_l, feature_names=input_feature_names)
            val_mse.append(_deriv_mse_on_lists(m, X_va_l, Xd_va_l))
        i_best = int(np.argmin(val_mse))
        chosen_thr = float(sweep_vals[i_best])
        np.savetxt(OUT_DIR / "lambda_sweep.csv",
                   np.c_[np.array(sweep_vals), np.array(val_mse)],
                   delimiter=",", header="lambda,derivative_mse", comments="")
        print(f"[INFO] Sweep chose threshold={chosen_thr:.3e}  (derivative MSE={val_mse[i_best]:.3e})")

    # 4) Final fit on all training trajectories (lists)
    model = make_model(chosen_thr)
    model.fit(X_list, t=t_list, x_dot=Xdot_list, feature_names=input_feature_names)

    # 5) Hard pin kinematics (optional)
    if PIN_KINEMATICS:
        _apply_kinematic_pin(model, input_feature_names, PIN_COEFF)

    # 6) Pretty equations + save artifacts
    eq_str = _pretty_equations(model, input_feature_names, precision=6)
    print("\n=== Learned equations ===")
    print(eq_str)

    joblib_dump(model, OUT_DIR / "model.pkl")
    np.save(OUT_DIR / "coefficients.npy", np.array(model.optimizer.coef_, dtype=float))
    (OUT_DIR / "feature_names.txt").write_text("\n".join(model.get_feature_names()))
    (OUT_DIR / "equations.txt").write_text(eq_str + "\n")

    metrics = {
        "threshold": chosen_thr,
        "max_iter": int(STLSQ_MAX_ITER),
        "ridge_alpha": float(STLSQ_RIDGE),
        "nnz_total": int(np.sum(np.abs(model.optimizer.coef_) > 0)),
        "train_runs": [p.name for p in train_paths],
        "heldout_runs": [p.name for p in heldout_paths],
        "poly_degree": int(POLY_DEGREE),
        "include_interaction": bool(INCLUDE_INTERACT),
        "include_bias": bool(INCLUDE_BIAS),
        "pin_kinematics": bool(PIN_KINEMATICS),
        "pin_coeff": list(PIN_COEFF),
        "angle_wrap": bool(ANGLE_WRAP),
        "stride": int(STRIDE),
        "savgol_window": int(SAVGOL_WINDOW),
        "savgol_polyorder": int(SAVGOL_POLYORDER),
        "sim_stride": int(SIM_STRIDE),
        "rtol": float(RTOL),
        "atol": float(ATOL),
        "max_step": float(MAX_STEP),
        "integrators_tried": INTEGRATOR_TRY,
    }

    # 7) Held-out simulation with robust integrator fallback + early-stop handling
    if heldout_paths:
        holdout = []
        for p in tqdm(heldout_paths, desc="Evaluating held-out"):
            Xh, Xdot_h, th = _build_run(p, ANGLE_WRAP, SAVGOL_WINDOW, SAVGOL_POLYORDER, STRIDE)
            x0 = Xh[0]
            ss = max(1, int(SIM_STRIDE))
            th_dec  = th[::ss]
            Xh_dec  = Xh[::ss]
            Xdot_dec= Xdot_h[::ss]

            y = None
            used = None
            last_err = None
            for meth in INTEGRATOR_TRY:
                try:
                    y = model.simulate(
                        x0, th_dec,
                        integrator="solve_ivp",
                        integrator_kws={"method": meth, "rtol": RTOL, "atol": ATOL, "max_step": MAX_STEP},
                    )
                    used = meth
                    break
                except Exception as e:
                    last_err = e
                    continue

            if y is None:
                holdout.append({"csv": p.name, "error": str(last_err)})
                print(f"[EVAL][WARN] {p.name}: all methods failed ({last_err})")
                continue

            # === Early-termination robust metrics ===
            e = min(len(y), len(Xh_dec))  # trim GT to returned length
            y = y[:e]
            Xh_e = Xh_dec[:e]
            Xdot_e = Xdot_dec[:e]

            pred_q = _wrap_to_pi(y[:, :2])
            true_q = _wrap_to_pi(Xh_e[:, :2])
            rmse_q  = float(np.sqrt(np.mean((pred_q - true_q)**2)))
            rmse_dq = float(np.sqrt(np.mean((y[:, 2:] - Xh_e[:, 2:])**2)))

            pred_dot = model.predict(Xh_e)
            deriv_mse = float(np.mean((pred_dot - Xdot_e)**2))

            holdout.append({
                "csv": p.name, "rmse_q": rmse_q, "rmse_dq": rmse_dq,
                "derivative_mse": deriv_mse, "steps": int(e),
                "method_used": used, "requested_steps": int(len(th_dec)),
            })
            print(f"[EVAL] {p.name}: RMSE(q)={rmse_q:.4e}, RMSE(dq)={rmse_dq:.4e}, d/dt MSE={deriv_mse:.4e} (solver={used}, steps={e}/{len(th_dec)})")

        metrics["heldout_results"] = holdout

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[✓] Saved model to: {OUT_DIR}")
    print(f"[✓] nnz={metrics['nnz_total']}  lambda={metrics['threshold']:.3e}")

if __name__ == "__main__":
    main()
