# utils/model_training/sindy.py
# -*- coding: utf-8 -*-
"""
SINDy (Sparse Identification of Nonlinear Dynamics) with optional GPU acceleration.

CPU backend:  NumPy + SciPy (original behavior)
GPU backend:  PyTorch (torch.linalg.lstsq) for STLSQ on CUDA

- Library Θ(X) is built with NumPy (feature parity and simplicity).
- If backend="torch", Θ and Xdot are moved to torch tensors and STLSQ runs on GPU.
- Coefficients Ξ are returned to NumPy so predict/simulate keep working the same.

Robustness:
- On GPU, defaults to float32 to avoid fragile cuSOLVER/MAGMA code paths with float64.
- Enforces contiguity, checks finiteness (NaN/Inf), and falls back to stabilized
  normal-equations solves if torch.linalg.lstsq fails.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

try:
    from scipy.signal import savgol_filter
    from scipy.integrate import solve_ivp
except Exception as e:
    raise ImportError(
        "This module requires scipy (signal, integrate). Install with: pip install scipy"
    ) from e

# Optional torch import (only needed if backend="torch")
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


# ----------------------------
# Utility: feature constructors (NumPy)
# ----------------------------

def _poly_features(X: np.ndarray, degree: int, include_bias: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Dense polynomial features up to 'degree' (with interactions).
    If include_bias=True, a bias column is the first column. Otherwise no bias column is added.
    """
    n_samples, n_state = X.shape
    feats = []
    names = []
    if include_bias:
        feats.append(np.ones((n_samples, 1), dtype=X.dtype))
        names.append("1")

    def _multi_indices(n_vars: int, deg: int):
        if n_vars == 1:
            yield (deg,)
            return
        if deg == 0:
            yield (0,) * n_vars
            return
        from itertools import combinations
        for bars in combinations(range(deg + n_vars - 1), n_vars - 1):
            exps = []
            last = -1
            for b in bars + (deg + n_vars - 1,):
                exps.append(b - last - 1)
                last = b
            yield tuple(exps)

    for d in range(1, degree + 1):
        for exps in _multi_indices(n_state, d):
            mon = np.ones(n_samples, dtype=X.dtype)
            name_parts = []
            for j, e in enumerate(exps):
                if e > 0:
                    mon *= X[:, j] ** e
                    name_parts.append(f"x{j+1}^{e}" if e > 1 else f"x{j+1}")
            feats.append(mon[:, None])
            names.append("*".join(name_parts) if name_parts else "1")
    return np.hstack(feats), names


def _trig_features(X: np.ndarray, harmonics: int = 1) -> Tuple[np.ndarray, List[str]]:
    """
    Trig library: [sin(k*x_i), cos(k*x_i)] for k=1..harmonics
    """
    n_samples, n_state = X.shape
    feats = []
    names = []
    for k in range(1, harmonics + 1):
        feats.append(np.sin(k * X))
        feats.append(np.cos(k * X))
        names.extend([f"sin{k}(x{i+1})" for i in range(n_state)])
        names.extend([f"cos{k}(x{i+1})" for i in range(n_state)])
    if feats:
        return np.hstack(feats), names
    return np.zeros((n_samples, 0), dtype=X.dtype), []


# ----------------------------
# Core SINDy Regressor
# ----------------------------

@dataclass
class SINDyConfig:
    poly_degree: int = 3
    trig_harmonics: int = 0
    threshold_lambda: float = 1e-3
    max_stlsq_iter: int = 10
    use_savgol_for_xdot: bool = True
    savgol_window: int = 31
    savgol_polyorder: int = 3
    normalize_columns: bool = True
    include_bias: bool = True
    # Backend:
    backend: str = "numpy"         # "numpy" (default) or "torch"
    device: Optional[str] = None   # e.g., "cuda", "cpu" (only for backend="torch")
    # Debugging / verbosity:
    debug: bool = False
    # --- NEW ---
    use_physics_library: bool = False          # turn on to use physics-aware basis
    angle_idx: Tuple[int, ...] = (0, 1)        # which state indices are angles


class SINDyRegressor:
    """
    SINDy with optional GPU acceleration for the STLSQ solve (via PyTorch).
    """

    def __init__(self, config: Optional[SINDyConfig] = None):
        self.cfg = config or SINDyConfig()
        self.coef_: Optional[np.ndarray] = None      # Ξ (p x n)
        self.feature_names_: List[str] = []
        self.column_scale_: Optional[np.ndarray] = None
        self.n_state_: Optional[int] = None

        if self.cfg.backend not in ("numpy", "torch"):
            raise ValueError("backend must be 'numpy' or 'torch'")

        if self.cfg.backend == "torch" and not _TORCH_AVAILABLE:
            raise ImportError("backend='torch' requested but PyTorch is not installed.")

        if self.cfg.backend == "torch" and self.cfg.device is None:
            # choose sensible default
            self.cfg.device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    # ---------- Public API ----------

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        Xdot: Optional[np.ndarray] = None,
    ) -> "SINDyRegressor":
        """
        Fit SINDy model from state time series.

        X:    (T, n)
        t:    (T,)
        Xdot: (T, n) if provided; else estimated if cfg.use_savgol_for_xdot is True
        """
        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=float)
        assert X.ndim == 2 and t.ndim == 1 and X.shape[0] == t.shape[0], "Shape mismatch X vs t"
        self.n_state_ = X.shape[1]

        if Xdot is None:
            if not self.cfg.use_savgol_for_xdot:
                raise ValueError("Xdot is None and derivative estimation disabled.")
            Xdot = self._estimate_derivative_savgol(X, t)
        else:
            Xdot = np.asarray(Xdot, dtype=float)

        # Basic finiteness checks
        if self.cfg.debug:
            print(f"[SINDy.fit] X shape={X.shape}, Xdot shape={Xdot.shape}, t shape={t.shape}")
            self._debug_check_np("X", X)
            self._debug_check_np("Xdot", Xdot)

        # Build Θ(X) in NumPy
        Theta, names = self._build_library(X)
        self.feature_names_ = names

        if self.cfg.debug:
            print(f"[SINDy.fit] Built library: Theta shape={Theta.shape} with {len(names)} features")

        # Column normalization (NumPy)
        if self.cfg.normalize_columns and Theta.shape[1] > 0:
            scale = np.linalg.norm(Theta, axis=0)
            scale[scale == 0] = 1.0
            Theta_scaled = Theta / scale
            self.column_scale_ = scale
            if self.cfg.debug:
                print("[SINDy.fit] Column normalization enabled. Example scales:",
                      f"min={scale.min():.3e}, max={scale.max():.3e}")
        else:
            Theta_scaled = Theta
            self.column_scale_ = np.ones(Theta.shape[1], dtype=Theta.dtype)
            if self.cfg.debug:
                print("[SINDy.fit] Column normalization disabled.")

        # Solve for sparse coefficients Ξ with STLSQ
        if self.cfg.backend == "numpy":
            if self.cfg.debug:
                print("[SINDy.fit] Using NumPy backend for STLSQ.")
            Xi = self._stlsq_numpy(Theta_scaled, Xdot, lam=self.cfg.threshold_lambda,
                                   max_iter=self.cfg.max_stlsq_iter)
        else:
            if self.cfg.debug:
                print(f"[SINDy.fit] Using Torch backend on device={self.cfg.device} for STLSQ.")
            Xi = self._stlsq_torch(Theta_scaled, Xdot, lam=self.cfg.threshold_lambda,
                                   max_iter=self.cfg.max_stlsq_iter, device=self.cfg.device)

        # Undo scaling on coefficients so prediction uses raw Θ(X)
        Xi_unscaled = Xi / self.column_scale_[:, None]
        self.coef_ = Xi_unscaled
        if self.cfg.debug:
            nnz = int(np.sum(np.abs(self.coef_) > 0))
            print(f"[SINDy.fit] Fit complete. nnz={nnz}")
        return self

    def predict_derivative(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        Theta, _ = self._build_library(np.asarray(X, dtype=float))
        Y = Theta @ self.coef_
        if not np.isfinite(Y).all():
            raise RuntimeError("predict_derivative produced non-finite values (check inputs / library overflow).")
        return Y

    def simulate(
        self,
        x0: np.ndarray,
        t_span: np.ndarray,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        method: str = "RK45",
        max_step: float = np.inf,                 # NEW
        wrap_angles: bool = False,                # NEW
        angle_idx: Tuple[int, ...] = (0, 1),      # NEW: which states are angles
        clip_q: Optional[float] = None,           # NEW: clip |q| <= clip_q
        clip_dq: Optional[float] = None,          # NEW: clip |dq| <= clip_dq
    ) -> np.ndarray:
        """
        Roll out the learned ODE using SciPy solve_ivp (CPU).
        Optional guard rails help avoid numerical blow-ups during long rollouts.
        """
        self._check_fitted()
        x0 = np.asarray(x0, dtype=float).ravel()
        assert x0.size == self.n_state_, "x0 dimension mismatch"
        t0, tf = float(t_span[0]), float(t_span[-1])

        def _wrap_pi(a: float) -> float:
            return (a + np.pi) % (2*np.pi) - np.pi

        def _guard_state(x: np.ndarray) -> np.ndarray:
            z = np.array(x, dtype=float, copy=True)
            if wrap_angles and angle_idx:
                for i in angle_idx:
                    z[i] = _wrap_pi(z[i])
            if clip_q is not None and z.size >= 2:
                z[:2] = np.clip(z[:2], -clip_q, clip_q)
            if clip_dq is not None and z.size >= 4:
                z[2:] = np.clip(z[2:], -clip_dq, clip_dq)
            return z

        def rhs(t, x):
            xs = _guard_state(x)
            fx = self.predict_derivative(xs[None, :]).ravel()
            if not np.isfinite(fx).all():
                raise RuntimeError("SINDy RHS produced non-finite derivative (overflow/NaN).")
            return fx

        sol = solve_ivp(
            rhs, (t0, tf), x0, t_eval=t_span,
            rtol=rtol, atol=atol, method=method, max_step=max_step
        )
        if not sol.success:
            raise RuntimeError(f"SINDy simulate() integrator failed: {sol.message}")
        return sol.y.T


    def sweep_lambda(
        self,
        X: np.ndarray,
        t: np.ndarray,
        Xdot: Optional[np.ndarray],
        lambdas: List[float],
        val_split: float = 0.2,
    ) -> Dict[str, np.ndarray]:
        """
        Sweep λ and compute validation error (one-step derivative MSE) and nnz.
        Done on the configured backend (torch if chosen).
        """
        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=float)
        if Xdot is None and self.cfg.use_savgol_for_xdot:
            Xdot = self._estimate_derivative_savgol(X, t)
        elif Xdot is None:
            raise ValueError("Xdot is None and derivative estimation disabled.")
        else:
            Xdot = np.asarray(Xdot, dtype=float)

        T = X.shape[0]
        T_val = max(int(val_split * T), 1)
        X_tr, X_val = X[:-T_val], X[-T_val:]
        t_tr, t_val = t[:-T_val], t[-T_val:]
        Xdot_tr, Xdot_val = Xdot[:-T_val], Xdot[-T_val:]

        Theta_tr, _ = self._build_library(X_tr)
        Theta_val, _ = self._build_library(X_val)

        # Column scaling from training only
        if self.cfg.normalize_columns and Theta_tr.shape[1] > 0:
            scale = np.linalg.norm(Theta_tr, axis=0)
            scale[scale == 0] = 1.0
            Theta_tr_s = Theta_tr / scale
            Theta_val_s = Theta_val / scale
        else:
            scale = np.ones(Theta_tr.shape[1], dtype=float)
            Theta_tr_s, Theta_val_s = Theta_tr, Theta_val

        if self.cfg.debug:
            print(f"[SINDy.sweep_lambda] Theta_tr_s={Theta_tr_s.shape}, Theta_val={Theta_val.shape}, "
                  f"lambdas={lambdas}")

        val_err = []
        nnz_list = []
        for lam in lambdas:
            if self.cfg.backend == "numpy":
                Xi = self._stlsq_numpy(Theta_tr_s, Xdot_tr, lam=float(lam),
                                       max_iter=self.cfg.max_stlsq_iter)
            else:
                Xi = self._stlsq_torch(Theta_tr_s, Xdot_tr, lam=float(lam),
                                       max_iter=self.cfg.max_stlsq_iter, device=self.cfg.device)
            Xi_unscaled = Xi / scale[:, None]
            pred = Theta_val @ Xi_unscaled
            mse = float(np.mean((pred - Xdot_val) ** 2))
            val_err.append(mse)
            nnz_list.append(int(np.sum(np.abs(Xi_unscaled) > 0)))
            if self.cfg.debug:
                print(f"[SINDy.sweep_lambda] λ={lam:.3e}  val_MSE={mse:.3e}  nnz={nnz_list[-1]}")

        return {"lambda": np.array(lambdas), "val_error": np.array(val_err), "nnz": np.array(nnz_list)}

    # ---------- Internals ----------

    def _estimate_derivative_savgol(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        dt = float(np.mean(np.diff(t)))
        win = self.cfg.savgol_window
        if win >= X.shape[0]:
            win = max(5, X.shape[0] - (1 - X.shape[0] % 2))
        if win % 2 == 0:
            win += 1
        # smooth (unused externally but common in SINDy examples)
        _ = savgol_filter(X, window_length=win, polyorder=self.cfg.savgol_polyorder, axis=0, mode="interp")
        Xdot = savgol_filter(X, window_length=win, polyorder=self.cfg.savgol_polyorder,
                             deriv=1, delta=dt, axis=0, mode="interp")
        return Xdot

    def _build_library(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        # NEW: prefer physics-aware library if requested
        if getattr(self.cfg, "use_physics_library", False):
            Theta, names = self._build_physics_library(X)
            if self.cfg.debug:
                self._debug_check_np("Theta (physics)", Theta)
            return Theta, names

        # Original polynomial/trig path (unchanged)
        poly_deg = max(self.cfg.poly_degree, 0)
        Theta_list = []
        names: List[str] = []

        if self.cfg.include_bias:
            Theta_list.append(np.ones((X.shape[0], 1), dtype=X.dtype))
            names.append("1")

        if poly_deg > 0:
            Theta_poly, names_poly = _poly_features(X, poly_deg, include_bias=False)
            Theta_list.append(Theta_poly)
            names.extend(names_poly)

        if self.cfg.trig_harmonics > 0:
            Theta_trig, names_trig = _trig_features(X, harmonics=self.cfg.trig_harmonics)
            Theta_list.append(Theta_trig)
            names.extend(names_trig)

        if not Theta_list:
            return np.zeros((X.shape[0], 0), dtype=X.dtype), []
        Theta = np.hstack(Theta_list)
        if self.cfg.debug:
            self._debug_check_np("Theta", Theta)
        return Theta, names

    
    
    def _build_physics_library(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Compact physics-aware Θ(X):
          Features: 1, {dq1,dq2,dq1^2,dq2^2,dq1*dq2},
                    {sin(q1),cos(q1),sin(q2),cos(q2),sin(q1-q2),cos(q1-q2)},
                    (each of those 6 trig terms) × (the 5 velocity monomials above).
        Assumes state ordering [q1,q2,dq1,dq2] and angle_idx=(0,1).
        """
        X = np.asarray(X, dtype=float)
        nT, nS = X.shape
        ai = tuple(self.cfg.angle_idx)
        if len(ai) != 2:
            raise ValueError("angle_idx must contain exactly two indices for (q1,q2).")

        # Map indices
        q1, q2 = X[:, ai[0]], X[:, ai[1]]

        # Heuristic: velocities are the other two cols (works for [q1,q2,dq1,dq2])
        vel_idx = [i for i in range(nS) if i not in ai]
        if len(vel_idx) != 2:
            raise ValueError("Expected 2 velocity columns besides angle_idx.")
        dq1, dq2 = X[:, vel_idx[0]], X[:, vel_idx[1]]

        # Trig building blocks
        s1, c1  = np.sin(q1), np.cos(q1)
        s2, c2  = np.sin(q2), np.cos(q2)
        s12     = np.sin(q1 - q2)
        c12     = np.cos(q1 - q2)

        feats, names = [], []
        def add(col, nm): feats.append(col[:, None]); names.append(nm)

        # bias
        add(np.ones_like(q1), "1")

        # velocity monomials
        add(dq1, "dq1"); add(dq2, "dq2")
        add(dq1*dq2, "dq1*dq2")
        add(dq1**2, "dq1^2"); add(dq2**2, "dq2^2")

        # pure trig
        trig_list = [("sin(q1)", s1), ("cos(q1)", c1),
                     ("sin(q2)", s2), ("cos(q2)", c2),
                     ("sin(q1-q2)", s12), ("cos(q1-q2)", c12)]
        for nm, val in trig_list:
            add(val, nm)

        # trig × velocity monomials
        vel_basis = [("dq1", dq1), ("dq2", dq2),
                     ("dq1^2", dq1**2), ("dq2^2", dq2**2), ("dq1*dq2", dq1*dq2)]
        for tnm, tval in trig_list:
            for vnm, vval in vel_basis:
                add(tval * vval, f"{tnm}*{vnm}")

        Theta = np.hstack(feats)
        return Theta, names



    # ---- STLSQ backends ----

    def _stlsq_numpy(self, Theta: np.ndarray, Xdot: np.ndarray, lam: float, max_iter: int = 10) -> np.ndarray:
        if self.cfg.debug:
            print(f"[STLSQ/NumPy] Theta={Theta.shape} Xdot={Xdot.shape} lam={lam:.3e} iters={max_iter}")
        self._debug_check_np("Theta", Theta)
        self._debug_check_np("Xdot", Xdot)

        T, p = Theta.shape
        n = Xdot.shape[1]
        Xi, *_ = np.linalg.lstsq(Theta, Xdot, rcond=None)

        for it in range(max_iter):
            small = np.abs(Xi) < lam
            Xi[small] = 0.0
            for k in range(n):
                support = ~small[:, k]
                if np.any(support):
                    Xi_k, *_ = np.linalg.lstsq(Theta[:, support], Xdot[:, k], rcond=None)
                    Xi[support, k] = Xi_k
                else:
                    Xi[:, k] = 0.0
            if self.cfg.debug:
                nnz = int(np.sum(np.abs(Xi) > 0))
                print(f"[STLSQ/NumPy] iter={it+1}/{max_iter} nnz={nnz}")
        return Xi

    def _stlsq_torch(
        self,
        Theta_np: np.ndarray,
        Xdot_np: np.ndarray,
        lam: float,
        max_iter: int = 10,
        device: Optional[str] = "cuda"
    ) -> np.ndarray:
        """
        Robust Torch STLSQ on device ("cuda" or "cpu"). Returns Xi as NumPy array.
        - float32 by default on GPU; float64 on CPU.
        - Ensures contiguity, checks finiteness.
        - Falls back to stabilized normal-equations solve if torch.linalg.lstsq fails.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not available.")

        if self.cfg.debug:
            print(f"[STLSQ/Torch] device={device} lam={lam:.3e} iters={max_iter} "
                  f"Theta_np={Theta_np.shape} Xdot_np={Xdot_np.shape}")

        # Prefer cuSOLVER if available (PyTorch >= 2.4). Ignore errors if not.
        try:
            import torch.backends.cuda as tbc
            if device and str(device).startswith("cuda"):
                try:
                    tbc.preferred_linalg_library("cusolver")
                except Exception:
                    pass
        except Exception:
            pass

        # Choose dtype: float32 on GPU is safer; float64 on CPU is fine.
        use_fp64 = (device == "cpu")
        dtype = torch.float64 if use_fp64 else torch.float32

        # Basic NumPy checks before moving to torch (quick fail if NaN/Inf):
        self._debug_check_np("Theta_np (pre-torch)", Theta_np)
        self._debug_check_np("Xdot_np (pre-torch)", Xdot_np)

        # to torch + contiguity
        Theta = torch.as_tensor(Theta_np, device=device, dtype=dtype).contiguous()
        Xdot  = torch.as_tensor(Xdot_np,  device=device, dtype=dtype).contiguous()

        # Finiteness checks on device
        self._debug_check_torch("Theta (torch)", Theta)
        self._debug_check_torch("Xdot (torch)", Xdot)

        T, p = Theta.shape
        n = Xdot.shape[1]

        def robust_ls(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            """Try lstsq; on failure fall back to normal equations with tiny ridge."""
            if A.numel() == 0 or B.numel() == 0:
                return torch.zeros((A.shape[1], B.shape[1]), device=A.device, dtype=A.dtype)

            try:
                sol = torch.linalg.lstsq(A, B).solution
                if not torch.isfinite(sol).all():
                    raise RuntimeError("lstsq produced non-finite solution")
                return sol
            except Exception as ex:
                if self.cfg.debug:
                    print(f"[STLSQ/Torch] lstsq failed ({type(ex).__name__}): {ex}. "
                          f"Falling back to normal equations.")
                At = A.transpose(0, 1).contiguous()
                AtA = At @ A
                AtB = At @ B
                diag_mean = torch.mean(torch.diag(AtA))
                # tiny ridge scaled with diag mean for numerical stability
                eps = (1e-8 if A.dtype == torch.float32 else 1e-12) * (diag_mean.abs() + 1.0)
                AtA_reg = AtA + eps * torch.eye(AtA.shape[0], device=A.device, dtype=A.dtype)
                sol = torch.linalg.solve(AtA_reg, AtB)
                if not torch.isfinite(sol).all():
                    raise RuntimeError("normal-equations solve produced non-finite solution")
                return sol

        # Initial LS estimate
        Xi = robust_ls(Theta, Xdot)  # (p, n)
        if self.cfg.debug:
            nnz0 = int(torch.sum(torch.abs(Xi) > 0).item())
            print(f"[STLSQ/Torch] initial nnz={nnz0}")

        for it in range(max_iter):
            small = torch.abs(Xi) < lam
            Xi = Xi.clone()
            Xi[small] = 0.0

            # Refit each column on its active support
            for k in range(n):
                support = ~small[:, k]
                if torch.any(support):
                    A = Theta[:, support].contiguous()
                    b = Xdot[:, k:k+1].contiguous()
                    Xi_k = robust_ls(A, b).view(-1)
                    Xi[support, k] = Xi_k
                else:
                    Xi[:, k] = 0.0

            if self.cfg.debug:
                nnz = int(torch.sum(torch.abs(Xi) > 0).item())
                print(f"[STLSQ/Torch] iter={it+1}/{max_iter} nnz={nnz}")

        # back to numpy
        Xi_np = Xi.detach().cpu().numpy()
        self._debug_check_np("Xi (torch->np)", Xi_np)
        return Xi_np

    def _check_fitted(self):
        if self.coef_ is None:
            raise RuntimeError("SINDyRegressor is not fitted yet. Call fit() first.")

    # ---------- Debug helpers ----------

    def _debug_check_np(self, name: str, arr: np.ndarray):
        """Print stats and first bad index if NaN/Inf (when debug=True)."""
        if not self.cfg.debug:
            return
        if arr.size == 0:
            print(f"[debug] {name}: EMPTY")
            return
        finite = np.isfinite(arr)
        if not finite.all():
            idx = np.argwhere(~finite)
            r, c = idx[0]
            print(f"[debug] {name}: non-finite at ({r},{c}) -> {arr[r, c]}")
        print(f"[debug] {name}: shape={arr.shape}  "
              f"min={np.nanmin(arr):.3e} max={np.nanmax(arr):.3e} "
              f"mean={np.nanmean(arr):.3e} std={np.nanstd(arr):.3e}")

    def _debug_check_torch(self, name: str, ten: "torch.Tensor"):
        """Print stats and first bad index if NaN/Inf (when debug=True)."""
        if not self.cfg.debug:
            return
        if ten.numel() == 0:
            print(f"[debug] {name}: EMPTY")
            return
        finite = torch.isfinite(ten)
        if not torch.all(finite):
            bad = (~finite).nonzero()
            r, c = bad[0].tolist() if bad.dim() == 2 and bad.shape[1] == 2 else (int(bad[0]), 0)
            val = ten.flatten()[0] if bad.numel() == 0 else ten[r, c]
            print(f"[debug] {name}: non-finite at ({r},{c}) -> {val}")
        print(f"[debug] {name}: shape={tuple(ten.shape)}  "
              f"min={ten.min().item():.3e} max={ten.max().item():.3e} "
              f"mean={ten.mean().item():.3e} std={ten.std().item():.3e}")
