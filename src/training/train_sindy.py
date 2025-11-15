#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a SINDy model on double-pendulum trajectories.

Data expected:
- CSV columns: t,q1,q2,dq1,dq2,tip_x,tip_y,tip_z,tip_x_rel,tip_z_rel,elbow_x,elbow_z,step_idx
- (Optionally) a manifest JSON listing runs -> CSV filenames.

State/derivative:
    X    = [q1, q2, dq1, dq2]
    Xdot = [dq1, dq2, ddq1, ddq2]   (ddq from Savitzky–Golay on q)

Outputs:
- out_dir/
    coef.npy
    feature_names.txt
    config.json
    metrics.json
    equations.txt
    lambda_sweep.csv (if sweep)
    selected_lambda.txt (if sweep)
    debug_dump/ (only when --debug)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

# Progress bars
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

import sys

# --- Make project root importable so we can import utils.model_training.sindy ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (…/double-pendulum-trajectory)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_training.sindy import SINDyRegressor, SINDyConfig  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402


# ===================== GLOBAL TRAINING HYPERS =====================
# Hard-coded: only use first 5 seconds of each trajectory (training + eval)
T_MAX_SECONDS = 5.0
# ================================================================


# ===================== DEFAULTS (safe & robust) =====================
DEFAULTS = {
    "data_dir": "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal1",
    "manifest": "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal1/double_pendulum_manifest_ideal.json",
    "poly_degree": 0,          # You can try 2 as well
    "trig_harmonics": 0,       # Keep 0 unless you patch class to support trig–poly cross terms
    # Leave lam=None to enable lambda-sweep below:
    "lam": None,
    "lambda_sweep": "1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2",
    "savgol_window": 51,
    "savgol_polyorder": 3,
    "angle_wrap": False,
    "stride": 2,
    "holdout_runs": 2,
    "normalize_columns": True,
    "include_bias": True,      # We will auto-fix duplicate bias if poly_degree>0
    # Backend defaults
    "backend": "torch",        # "torch" (GPU if available) or "numpy"
    "device": "cuda",          # "cuda" or "cpu" (used when backend="torch")
    "debug": False,
}
# ====================================================================


# ---------------------------
# Helpers
# ---------------------------

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
        if dump_dir:
            dump_dir.mkdir(parents=True, exist_ok=True)
            np.save(dump_dir / f"{name.replace(' ', '_')}.npy", arr)
            msg += f" Dumped to: {dump_dir/(name.replace(' ', '_') + '.npy')}"
        raise ValueError(msg)

def _read_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load one trajectory CSV into dict of arrays."""
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    # Ensure column presence
    for c in REQ_COLS:
        if c not in arr.dtype.names:
            raise ValueError(f"[DATA] Missing required column '{c}' in {csv_path}")
    data = {name: arr[name] for name in arr.dtype.names}
    return data

def _robust_savgol_window(T: int, requested: int, poly: int) -> int:
    """
    Return a robust odd window length:
    - ≤ T-1
    - ≥ max(5, poly+2)
    - odd
    """
    if T < 3:
        raise ValueError(f"[SavGol] Too few samples: T={T}")
    win = min(requested, T - 1)
    win = max(win, max(5, poly + 2))
    if win % 2 == 0:
        # prefer the nearest lower odd if possible, else +1
        win = win - 1 if win - 1 >= 5 else win + 1
    # if still even (pathological), bump to next odd
    if win % 2 == 0:
        win += 1
    if win >= T:
        win = T - 1 if (T - 1) % 2 == 1 else T - 2
    if win < 5:
        win = 5
    return win

def _estimate_ddq(dq: np.ndarray, t: np.ndarray, win: int, poly: int) -> np.ndarray:
    """Savitzky–Golay differentiation on dq to get ddq (per state)."""
    dt = float(np.mean(np.diff(t)))
    if not np.all(np.diff(t) > 0):
        raise ValueError("[TIME] t must be strictly increasing for SavGol differentiation.")
    win_robust = _robust_savgol_window(len(t), win, poly)
    ddq = savgol_filter(dq, window_length=win_robust, polyorder=poly,
                        deriv=1, delta=dt, axis=0, mode="interp")
    return ddq

def _estimate_ddq_from_q(q: np.ndarray, t: np.ndarray, win: int, poly: int) -> np.ndarray:
    """Savitzky–Golay second derivative on q to get ddq (less noisy than diff of dq)."""
    dt = float(np.mean(np.diff(t)))
    win_robust = _robust_savgol_window(len(t), win, poly)
    ddq = savgol_filter(q, window_length=win_robust, polyorder=poly,
                        deriv=2, delta=dt, axis=0, mode="interp")
    return ddq


def _build_X_Xdot_from_csv(csv_path: Path,
                           angle_wrap: bool,
                           savgol_window: int,
                           savgol_polyorder: int,
                           stride: int = 1,
                           debug_dump: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build X, Xdot, t from one CSV file with strict checks, then clamp to T_MAX_SECONDS."""
    d = _read_csv(csv_path)
    t = d["t"].astype(float)

    # Monotonic t check (within this single run)
    if np.any(np.diff(t) <= 0):
        nonmono_idx = int(np.where(np.diff(t) <= 0)[0][0])
        raise ValueError(f"[TIME] Non-monotonic or duplicate t at index {nonmono_idx} in {csv_path}")

    q = np.vstack([d["q1"], d["q2"]]).T
    dq = np.vstack([d["dq1"], d["dq2"]]).T

    if angle_wrap:
        q = _wrap_to_pi(q)

    # Check minimum length for SavGol (robust sizing inside)
    if len(t) < 6:
        raise ValueError(f"[LENGTH] {csv_path} too short for SavGol: T={len(t)} (need ≥6)")

    # Derivative: ddq from q (2nd derivative)
    ddq = _estimate_ddq_from_q(q, t, savgol_window, max(3, savgol_polyorder))

    # Sanity: no NaN/Inf in raw channels
    _ensure_finite(q,   "q",   ctx=f"file={csv_path.name}", dump_dir=debug_dump)
    _ensure_finite(dq,  "dq",  ctx=f"file={csv_path.name}", dump_dir=debug_dump)
    _ensure_finite(ddq, "ddq", ctx=f"file={csv_path.name}", dump_dir=debug_dump)

    # State and derivative
    X = np.hstack([q, dq])
    Xdot = np.hstack([dq, ddq])

    _ensure_finite(X,    "X",    ctx=f"file={csv_path.name}", dump_dir=debug_dump)
    _ensure_finite(Xdot, "Xdot", ctx=f"file={csv_path.name}", dump_dir=debug_dump)

    # ---- Hard clamp: only keep first T_MAX_SECONDS of this run ----
    if T_MAX_SECONDS is not None:
        rel_t = t - t[0]
        mask = rel_t <= T_MAX_SECONDS
        if not np.any(mask):
            raise ValueError(
                f"[T_MAX] No samples within {T_MAX_SECONDS}s in {csv_path} "
                f"(t range: {t[0]:.4f}–{t[-1]:.4f})"
            )
        t = t[mask]
        X = X[mask]
        Xdot = Xdot[mask]

        if len(t) < 6:
            raise ValueError(
                f"[T_MAX] After clamping to {T_MAX_SECONDS}s, too few samples in {csv_path} (T={len(t)})"
            )

    # Downsample (stride) after clamping
    if stride > 1:
        sl = slice(0, None, stride)
        return X[sl], Xdot[sl], t[sl]
    return X, Xdot, t

def _dataset_stats(X: np.ndarray, Xdot: np.ndarray, name: str = "train"):
    def stats(v):
        return dict(min=float(np.min(v)), max=float(np.max(v)),
                    mean=float(np.mean(v)), std=float(np.std(v)))
    S = {
        f"{name}_q1":  stats(X[:, 0]),
        f"{name}_q2":  stats(X[:, 1]),
        f"{name}_dq1": stats(X[:, 2]),
        f"{name}_dq2": stats(X[:, 3]),
        f"{name}_ddq1": stats(Xdot[:, 2]),
        f"{name}_ddq2": stats(Xdot[:, 3]),
    }
    return S

def _load_dataset(
    data_dir: Path,
    manifest: Optional[Path],
    angle_wrap: bool,
    savgol_window: int,
    savgol_polyorder: int,
    stride: int,
    holdout_runs: int,
    debug: bool,
    debug_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Path], List[Path]]:
    """Load many runs and concatenate; return train (X,Xdot,t) and a list of held-out run paths."""
    csv_paths: List[Path] = []
    if manifest and manifest.exists():
        meta = json.loads(Path(manifest).read_text())
        csv_paths = [data_dir / r["csv"] for r in meta.get("runs", [])]
        if not csv_paths:
            raise ValueError(f"[DATA] No runs in manifest: {manifest}")
    else:
        csv_paths = sorted(data_dir.glob("double_pendulum_traj_*.csv"))
        if not csv_paths:
            raise ValueError(f"[DATA] No CSVs found in {data_dir}")

    # Hold out last K runs for evaluation
    held_out = csv_paths[-holdout_runs:] if holdout_runs > 0 else []
    train_paths = csv_paths[:-holdout_runs] if holdout_runs > 0 else csv_paths

    X_list, Xdot_list, t_list = [], [], []
    for p in tqdm(train_paths, desc="Loading training CSVs"):
        try:
            X, Xdot, t = _build_X_Xdot_from_csv(
                p, angle_wrap=angle_wrap, savgol_window=savgol_window,
                savgol_polyorder=savgol_polyorder, stride=stride,
                debug_dump=(debug_dir if debug else None),
            )
        except Exception as e:
            raise RuntimeError(f"[DATA] Problem while reading/building {p}") from e

        _ensure_finite(X, "X(file)", ctx=str(p), dump_dir=(debug_dir if debug else None))
        _ensure_finite(Xdot, "Xdot(file)", ctx=str(p), dump_dir=(debug_dir if debug else None))

        X_list.append(X)
        Xdot_list.append(Xdot)
        t_list.append(t)

    X_all    = np.vstack(X_list)
    Xdot_all = np.vstack(Xdot_list)
    t_all    = np.concatenate(t_list)

    _ensure_finite(X_all, "X_all", dump_dir=(debug_dir if debug else None))
    _ensure_finite(Xdot_all, "Xdot_all", dump_dir=(debug_dir if debug else None))

    return X_all, Xdot_all, t_all, train_paths, held_out

def _nnz(Xi: np.ndarray) -> int:
    return int(np.sum(np.abs(Xi) > 0))

def _print_equations(feature_names: List[str], Xi: np.ndarray) -> str:
    """
    Pretty equations for the 4D state [q1,q2,dq1,dq2].
    Returns a string.
    """
    eqs = []
    state_names = ["q1", "q2", "dq1", "dq2"]
    for k, sname in enumerate(state_names):
        terms = []
        for i, coef in enumerate(Xi[:, k]):
            if coef != 0.0:  # exact zero after thresholding
                terms.append(f"{coef:+.4e}*{feature_names[i]}")
        rhs = " + ".join(terms) if terms else "0"
        eqs.append(f"d/dt {sname} = {rhs}")
    return "\n".join(eqs)


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Train SINDy on double-pendulum data (with rich debug)")

    # Defaults wired from DEFAULTS so you can run with no flags
    ap.add_argument("--data-dir", type=str, default=DEFAULTS["data_dir"],
                    help="Directory containing CSVs (and optional manifest)")
    ap.add_argument("--manifest", type=str, default=DEFAULTS["manifest"],
                    help="Path to manifest JSON (optional)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Where to save the trained model; defaults to data-dir/sindy_model")

    ap.add_argument("--poly-degree", type=int, default=DEFAULTS["poly_degree"])
    ap.add_argument("--trig-harmonics", type=int, default=DEFAULTS["trig_harmonics"],
                    help="Use sin/cos(k*angle) features; 0 disables (note: no trig–poly cross terms in class)")
    ap.add_argument("--lambda", dest="lam", type=float, default=DEFAULTS["lam"],
                    help="Sparsity lambda. If omitted and --lambda-sweep provided, choose best from sweep.")
    ap.add_argument("--lambda-sweep", type=str, default=DEFAULTS["lambda_sweep"],
                    help="Comma-separated lambdas for sweep, e.g. '1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2'")

    ap.add_argument("--max-stlsq-iter", type=int, default=10)

    ap.add_argument("--normalize-columns", dest="normalize_columns",
                    action="store_true", default=DEFAULTS["normalize_columns"])
    ap.add_argument("--no-normalize-columns", dest="normalize_columns",
                    action="store_false")

    ap.add_argument("--include-bias", dest="include_bias",
                    action="store_true", default=DEFAULTS["include_bias"])
    ap.add_argument("--no-bias", dest="include_bias",
                    action="store_false")

    ap.add_argument("--angle-wrap", action="store_true", default=DEFAULTS["angle_wrap"],
                    help="Wrap angles to (-pi,pi]")
    ap.add_argument("--savgol-window", type=int, default=DEFAULTS["savgol_window"])
    ap.add_argument("--savgol-polyorder", type=int, default=DEFAULTS["savgol_polyorder"])
    ap.add_argument("--stride", type=int, default=DEFAULTS["stride"],
                    help="Downsample stride (>=1)")
    ap.add_argument("--holdout-runs", type=int, default=DEFAULTS["holdout_runs"],
                    help="Number of CSV runs to hold out for eval")

    # Backend controls
    ap.add_argument("--backend", type=str, choices=["numpy", "torch"], default=DEFAULTS["backend"],
                    help="Numerical backend for STLSQ (torch uses GPU if device='cuda').")
    ap.add_argument("--device", type=str, default=DEFAULTS["device"],
                    help="Device for torch backend: 'cuda' or 'cpu'")

    # Debug
    ap.add_argument("--debug", action="store_true", default=DEFAULTS["debug"],
                    help="Enable verbose checks and dump matrices on failure")
    
    # --- Simulation / evaluation controls ---
    ap.add_argument("--sim-method", type=str, default="Radau",
                    choices=["RK45", "Radau", "BDF", "LSODA"],
                    help="IVP solver for held-out rollout.")
    ap.add_argument("--sim-rtol", type=float, default=1e-4, help="Relative tolerance for solve_ivp.")
    ap.add_argument("--sim-atol", type=float, default=1e-6, help="Absolute tolerance for solve_ivp.")
    ap.add_argument("--sim-max-step", type=float, default=0.02, help="Max integrator step (seconds).")

    # Angle wrapping + clipping guard-rails (used by SINDyRegressor.simulate)
    ap.add_argument("--sim-wrap-angles", dest="sim_wrap_angles",
                    action="store_true", default=True,
                    help="Wrap angle states to (-pi, pi] during rollout.")
    ap.add_argument("--no-sim-wrap-angles", dest="sim_wrap_angles",
                    action="store_false")
    ap.add_argument("--sim-angle-idx", type=str, default="0,1",
                    help="Comma-separated indices of angle states (e.g., '0,1').")
    ap.add_argument("--sim-clip-q", type=float, default=np.pi,
                    help="Clip |q| <= this value during rollout (None to disable).")
    ap.add_argument("--sim-clip-dq", type=float, default=50.0,
                    help="Clip |dq| <= this value during rollout (None to disable).")

    # Decimation + burn-in for RMSE
    ap.add_argument("--sim-stride", type=int, default=5,
                    help="Downsample held-out signals before sim (>=1).")
    ap.add_argument("--rmse-burnin", type=float, default=0.0,
                    help="Ignore first this many seconds when computing RMSE.")

    args = ap.parse_args()
    
    # Parse angle indices ("" → empty tuple)
    angle_idx = tuple(int(s) for s in str(args.sim_angle_idx).split(",") if s.strip())

    data_dir = Path(args.data_dir).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else None
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (data_dir / "sindy_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug_dump"

    # Safety note about trig cross terms
    if args.trig_harmonics > 0:
        print("[NOTE] trig_harmonics > 0: The current SINDy class concatenates trig(X) "
              "and poly(X) but DOES NOT multiply them. Terms like sin(q1 - q2)*dq2^2 will "
              "NOT appear unless you modify the class to include trig–poly cross terms.")

    # Warn & auto-fix duplicate-bias issue in current class:
    # If include_bias=True and poly_degree>0, the class will include TWO bias columns (one explicit,
    # one from the polynomial block). We avoid this by disabling include_bias here.
    if args.include_bias and args.poly_degree > 0:
        print("[WARN] Detected include_bias=True with poly_degree>0. "
              "The current SINDy class will add TWO bias columns (ill-conditioned). "
              "Auto-fixing by setting include_bias=False for this run.")
        args.include_bias = False

    # Load dataset (concat many runs) and keep last K for eval
    X, Xdot, t_all, train_paths, held_out_paths = _load_dataset(
        data_dir=data_dir, manifest=manifest,
        angle_wrap=args.angle_wrap,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        stride=max(1, args.stride),
        holdout_runs=max(0, args.holdout_runs),
        debug=args.debug,
        debug_dir=debug_dir,
    )

    # Print dataset stats
    ds_stats = _dataset_stats(X, Xdot, name="train")
    print("[INFO] Dataset stats:")
    for k, v in ds_stats.items():
        print(f"  {k:12s} min={v['min']:+.4e} max={v['max']:+.4e} mean={v['mean']:+.4e} std={v['std']:+.4e}")
    print(f"[INFO] Shapes: X={X.shape}, Xdot={Xdot.shape}, t_all={t_all.shape}")
    print(f"[INFO] t: first={t_all[0]:.6f} last={t_all[-1]:.6f} N={len(t_all)}")
    print(f"[INFO] Per-run clamp: using only first {T_MAX_SECONDS:.2f}s of each trajectory before concatenation.")

    # Configure SINDy
    cfg = SINDyConfig(
        poly_degree=int(args.poly_degree),
        trig_harmonics=int(args.trig_harmonics),
        threshold_lambda=1e-3 if args.lam is None else float(args.lam),
        max_stlsq_iter=int(args.max_stlsq_iter),
        use_savgol_for_xdot=False,          # we supply Xdot explicitly
        savgol_window=int(args.savgol_window),
        savgol_polyorder=int(args.savgol_polyorder),
        normalize_columns=bool(args.normalize_columns),
        include_bias=bool(args.include_bias),
        backend=str(args.backend),
        device=str(args.device) if args.backend == "torch" else None,
        # --- physics-aware library ---
        use_physics_library=True,
        angle_idx=(0, 1),
    )

    # Try to instantiate with requested backend; fallback to numpy if torch not available
    try:
        base_reg = SINDyRegressor(cfg)
    except ImportError as e:
        print(f"[WARN] {e}. Falling back to backend='numpy'.")
        cfg.backend = "numpy"
        cfg.device = None
        base_reg = SINDyRegressor(cfg)

    # If torch+cuda requested but not available, warn loudly (class will already choose cpu)
    if cfg.backend == "torch" and (cfg.device is None or str(cfg.device).lower() == "cuda"):
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                print("[WARN] device='cuda' requested but CUDA is not available. "
                      "Training will run on CPU (torch). Consider --device cpu or --backend numpy for speed.")
        except Exception:
            print("[WARN] PyTorch import failed after initial instantiation guard.")

    # =========================
    # Lambda sweep with tqdm
    # =========================
    chosen_lambda = cfg.threshold_lambda
    if args.lam is None and str(args.lambda_sweep).strip():
        sweep_vals = [float(s) for s in str(args.lambda_sweep).split(",")]
        # Split train/val temporally
        T = X.shape[0]
        T_val = max(int(0.2 * T), 1)
        X_tr, X_val = X[:-T_val], X[-T_val:]
        t_tr, t_val = t_all[:-T_val], t_all[-T_val:]
        Xdot_tr, Xdot_val = Xdot[:-T_val], Xdot[-T_val:]

        # Build libraries once (use class’s internal builder for parity)
        try:
            Theta_tr, _ = base_reg._build_library(X_tr)
            Theta_val, _ = base_reg._build_library(X_val)
        except Exception as e:
            raise RuntimeError("[LIB] Failed to build feature libraries for sweep. "
                               f"Details: {type(e).__name__}: {e}") from e

        # Column scaling from training Θ only
        if cfg.normalize_columns and Theta_tr.shape[1] > 0:
            scale = np.linalg.norm(Theta_tr, axis=0)
            scale[scale == 0] = 1.0
            Theta_tr_s = Theta_tr / scale
            Theta_val_s = Theta_val / scale
        else:
            scale = np.ones(Theta_tr.shape[1], dtype=float)
            Theta_tr_s, Theta_val_s = Theta_tr, Theta_val

        # Optional extra diagnostics
        if args.debug:
            norms = np.linalg.norm(Theta_tr, axis=0)
            topk = min(10, norms.shape[0])
            idx_sorted = np.argsort(norms)[::-1][:topk]
            print("[DEBUG] Top feature column norms (train):")
            for i in idx_sorted:
                print(f"  col[{i:4d}] norm={norms[i]:.4e}")
            zero_cols = np.where(norms < 1e-15)[0]
            if zero_cols.size:
                print(f"[DEBUG] ~zero-norm columns: {zero_cols[:20]}{' ...' if zero_cols.size>20 else ''}")
            # Dumps
            debug_dir.mkdir(parents=True, exist_ok=True)
            np.save(debug_dir / "Theta_tr.npy", Theta_tr)
            np.save(debug_dir / "Theta_val.npy", Theta_val)
            np.save(debug_dir / "Theta_tr_s.npy", Theta_tr_s)
            np.save(debug_dir / "Theta_val_s.npy", Theta_val_s)
            np.save(debug_dir / "Xdot_tr.npy", Xdot_tr)
            np.save(debug_dir / "Xdot_val.npy", Xdot_val)

        # Strict finite checks
        _ensure_finite(Theta_tr,   "Theta_tr",   dump_dir=(debug_dir if args.debug else None))
        _ensure_finite(Theta_val,  "Theta_val",  dump_dir=(debug_dir if args.debug else None))
        _ensure_finite(Theta_tr_s, "Theta_tr_s", dump_dir=(debug_dir if args.debug else None))
        _ensure_finite(Theta_val_s,"Theta_val_s",dump_dir=(debug_dir if args.debug else None))
        _ensure_finite(Xdot_tr,    "Xdot_tr",    dump_dir=(debug_dir if args.debug else None))
        _ensure_finite(Xdot_val,   "Xdot_val",   dump_dir=(debug_dir if args.debug else None))

        val_err = []
        nnz_list = []

        pbar = tqdm(sweep_vals, desc="Lambda sweep")
        for lam in pbar:
            try:
                # STLSQ on chosen backend
                if cfg.backend == "numpy":
                    Xi = base_reg._stlsq_numpy(Theta_tr_s, Xdot_tr, lam=float(lam),
                                               max_iter=cfg.max_stlsq_iter)
                else:
                    Xi = base_reg._stlsq_torch(Theta_tr_s, Xdot_tr, lam=float(lam),
                                               max_iter=cfg.max_stlsq_iter, device=cfg.device)
            except Exception as e:
                # Dump and re-raise with context
                if args.debug:
                    np.save(debug_dir / f"Xi_fail_lambda_{lam:.2e}.npy", np.array([[]]))
                    np.save(debug_dir / f"Theta_tr_s_lambda_{lam:.2e}.npy", Theta_tr_s)
                    np.save(debug_dir / f"Xdot_tr_lambda_{lam:.2e}.npy", Xdot_tr)
                raise RuntimeError(
                    f"[STLSQ] Failure during lambda={lam:.3e} on backend={cfg.backend}. "
                    f"Dumps saved to {debug_dir} (if --debug)."
                ) from e

            # Unscale coeffs and evaluate MSE on unscaled Theta
            Xi_unscaled = Xi / scale[:, None]
            pred = Theta_val @ Xi_unscaled
            mse = float(np.mean((pred - Xdot_val) ** 2))
            nz = int(np.sum(np.abs(Xi_unscaled) > 0))
            val_err.append(mse)
            nnz_list.append(nz)
            pbar.set_postfix({"λ": f"{lam:.2e}", "val_MSE": f"{mse:.3e}", "nnz": nz})

        # Select best lambda
        idx = int(np.argmin(val_err))
        chosen_lambda = float(sweep_vals[idx])
        cfg.threshold_lambda = chosen_lambda
        print(f"[INFO] Sweep selected lambda={chosen_lambda:.3e} (val MSE={val_err[idx]:.3e})")

        # Save sweep curve
        sweep_arr = np.c_[np.array(sweep_vals), np.array(val_err), np.array(nnz_list)]
        np.savetxt(out_dir / "lambda_sweep.csv",
                   sweep_arr, delimiter=",", header="lambda,val_error,nnz", comments="")
        (out_dir / "selected_lambda.txt").write_text(f"{chosen_lambda:.6e}\n")

    # =========================
    # Final fit on all training
    # =========================
    print("[INFO] Fitting final SINDy model...")
    try:
        sindy = SINDyRegressor(cfg).fit(X, t_all, Xdot=Xdot)
    except Exception as e:
        if args.debug:
            # Dump entire training set for post-mortem
            debug_dir.mkdir(parents=True, exist_ok=True)
            np.save(debug_dir / "X_all.npy", X)
            np.save(debug_dir / "Xdot_all.npy", Xdot)
            np.save(debug_dir / "t_all.npy", t_all)
        raise RuntimeError(
            f"[FIT] Final fit failed on backend={cfg.backend}. "
            f"Dumped arrays to {debug_dir} (if --debug). Details: {type(e).__name__}: {e}"
        ) from e

    # Save model
    np.save(out_dir / "coef.npy", sindy.coef_)
    (out_dir / "feature_names.txt").write_text("\n".join(sindy.feature_names_))

    # Pretty print
    eq_str = _print_equations(sindy.feature_names_, sindy.coef_)
    print("\n=== Learned equations ===")
    print(eq_str)
    (out_dir / "equations.txt").write_text(eq_str + "\n")

    # --------------------------
    # Evaluate on held-out runs
    # --------------------------
    metrics = {
        "chosen_lambda": chosen_lambda,
        "nnz_total": _nnz(sindy.coef_),
        "train_rows": int(X.shape[0]),
        "train_runs": len(train_paths),
        "heldout_runs": len(held_out_paths),
        "holdout_results": [],
        "backend": cfg.backend,
        "device": cfg.device if cfg.backend == "torch" else None,
        "dataset_stats": _dataset_stats(X, Xdot, name="train"),
        "t_max_seconds": T_MAX_SECONDS,
    }

    if held_out_paths:
        for p in tqdm(held_out_paths, desc="Evaluating held-out runs"):
            Xh, Xdot_h, th = _build_X_Xdot_from_csv(
                p, angle_wrap=args.angle_wrap,
                savgol_window=args.savgol_window,
                savgol_polyorder=args.savgol_polyorder,
                stride=max(1, args.stride),
                debug_dump=(debug_dir if args.debug else None),
            )
            # Simulate from first state using learned f
            x0 = Xh[0]

            # Decimate for numerical robustness and speed
            sim_stride = max(1, int(args.sim_stride))
            th_sim = th[::sim_stride]
            Xh_sim = Xh[::sim_stride]
            Xdot_h_sim = Xdot_h[::sim_stride]

            # Burn-in index (seconds from start of th_sim)
            if float(args.rmse_burnin) > 0:
                rel_t = th_sim - th_sim[0]
                k0 = int(np.searchsorted(rel_t, float(args.rmse_burnin)))
            else:
                k0 = 0

            # Build kwargs for SINDy.simulate (uses the new guard-rail signature)
            sim_kwargs = dict(
                method=args.sim_method,
                rtol=float(args.sim_rtol),
                atol=float(args.sim_atol),
                max_step=float(args.sim_max_step),
                wrap_angles=bool(args.sim_wrap_angles),
                angle_idx=angle_idx,
                clip_q=(None if args.sim_clip_q is None else float(args.sim_clip_q)),
                clip_dq=(None if args.sim_clip_dq is None else float(args.sim_clip_dq)),
            )

            # Try chosen method, then fall back if needed
            try_methods = [args.sim_method]
            # sensible fallback order
            for m in ["Radau", "BDF", "LSODA", "RK45"]:
                if m not in try_methods:
                    try_methods.append(m)

            y_sim = None
            last_err = None
            for m in try_methods:
                try:
                    sim_kwargs["method"] = m
                    y_sim = sindy.simulate(x0, th_sim, **sim_kwargs)
                    print(f"[SIM] {p.name}: succeeded with method={m}, "
                          f"rtol={sim_kwargs['rtol']}, atol={sim_kwargs['atol']}, max_step={sim_kwargs['max_step']}")
                    break
                except Exception as e:
                    last_err = e
                    print(f"[SIM][WARN] {p.name}: method={m} failed: {e}")

            if y_sim is None:
                if args.debug:
                    np.save(debug_dir / f"heldout_X_{p.name}.npy", Xh)
                    np.save(debug_dir / f"heldout_t_{p.name}.npy", th)
                raise RuntimeError(f"[SIM] simulate() failed on held-out file {p.name}: {last_err}")

            # angle wrap for error (match periodicity): wrap both true & pred for q
            true_q = _wrap_to_pi(Xh_sim[:, :2])
            pred_q = _wrap_to_pi(y_sim[:, :2])

            # RMSE on states (with burn-in)
            rmse_q  = float(np.sqrt(np.mean((pred_q[k0:] - true_q[k0:])**2)))
            rmse_dq = float(np.sqrt(np.mean((y_sim[k0:, 2:] - Xh_sim[k0:, 2:])**2)))

            # Derivative MSE on held-out (one-step derivative fit quality)
            try:
                f_pred = sindy.predict_derivative(Xh_sim)
                deriv_mse = float(np.mean((f_pred[k0:] - Xdot_h_sim[k0:]) ** 2))
            except Exception as e:
                deriv_mse = float("nan")
                if args.debug:
                    print(f"[WARN] predict_derivative failed on {p.name}: {e}")

            metrics["holdout_results"].append({
                "csv": str(p.name),
                "rmse_q": rmse_q,
                "rmse_dq": rmse_dq,
                "derivative_mse": deriv_mse,
                "steps": int(len(th_sim)),
                "rmse_burnin_index": int(k0),
                "sim_method_used": sim_kwargs["method"],
                "feature_count": len(sindy.feature_names_),
                "sim": {
                    "method_requested": args.sim_method,
                    "rtol": float(args.sim_rtol),
                    "atol": float(args.sim_atol),
                    "max_step": float(args.sim_max_step),
                    "wrap_angles": bool(args.sim_wrap_angles),
                    "angle_idx": list(angle_idx),
                    "clip_q": (None if args.sim_clip_q is None else float(args.sim_clip_q)),
                    "clip_dq": (None if args.sim_clip_dq is None else float(args.sim_clip_dq)),
                    "stride": int(args.sim_stride),
                    "rmse_burnin": float(args.rmse_burnin),
                },
            })
            print(f"[EVAL] {p.name}: RMSE(q)={rmse_q:.4e}, RMSE(dq)={rmse_dq:.4e}, d/dt MSE={deriv_mse:.4e}")


    # Save config + metrics
    config_to_save = dict(
        poly_degree=cfg.poly_degree,
        trig_harmonics=cfg.trig_harmonics,
        threshold_lambda=cfg.threshold_lambda,
        max_stlsq_iter=cfg.max_stlsq_iter,
        normalize_columns=cfg.normalize_columns,
        include_bias=cfg.include_bias,
        angle_wrap=args.angle_wrap,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        stride=args.stride,
        data_dir=str(data_dir),
        manifest=str(manifest) if manifest else None,
        train_runs=[p.name for p in train_paths],
        heldout_runs=[p.name for p in held_out_paths],
        backend=cfg.backend,
        device=cfg.device if cfg.backend == "torch" else None,
        t_max_seconds=T_MAX_SECONDS,
    )
    (out_dir / "config.json").write_text(json.dumps(config_to_save, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n[✓] Saved model to: {out_dir}")
    print(f"[✓] nnz={metrics['nnz_total']}  lambda={chosen_lambda:.3e}  backend={cfg.backend}"
          + (f" device={cfg.device}" if cfg.backend == "torch" else ""))
    if metrics["heldout_runs"]:
        print(f"[✓] Held-out results saved in metrics.json")

if __name__ == "__main__":
    main()
