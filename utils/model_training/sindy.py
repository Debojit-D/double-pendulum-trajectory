# utils/model_training/sindy.py
# -*- coding: utf-8 -*-
"""
SINDy (Sparse Identification of Nonlinear Dynamics)

Implements the core algorithm from:
Brunton, Proctor, Kutz, "Discovering governing equations from data:
Sparse identification of nonlinear dynamical systems (SINDy)."
arXiv:1509.03580

Key ideas implemented:
- Build a nonlinear feature library Θ(X) (polynomials + optional trig).  [Sec. 3; Eq. (5)]
- Solve  Xdot = Θ(X) Ξ  with sparse columns Ξ via STLSQ.            [Sec. 3.1; Code 1]
- Estimate derivatives if only states are measured (Savitzky–Golay).  [Sec. 3.1]
- Choose λ via accuracy–complexity tradeoff (Pareto elbow).          [Sec. 3.2]

Design choices mirror the paper’s recommendations; see in-line notes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    from scipy.signal import savgol_filter
    from scipy.integrate import solve_ivp
except Exception as e:
    raise ImportError(
        "This module requires scipy (signal, integrate). "
        "Install with: pip install scipy"
    ) from e


# ----------------------------
# Utility: feature constructors
# ----------------------------

def _poly_features(X: np.ndarray, degree: int) -> Tuple[np.ndarray, List[str]]:
    """
    Build dense polynomial features up to a given degree (including interactions).
    Θ_poly(X) = [1, x_i, x_i x_j, ..., monomials up to 'degree']  (order does not matter)
    This corresponds to Θ(X) terms used in the paper’s examples (e.g., Lorenz up to 5th). [Eq. (23)]
    """
    n_samples, n_state = X.shape
    # Start with bias 1
    feats = [np.ones((n_samples, 1), dtype=X.dtype)]
    names = ["1"]

    # Generate all monomials up to 'degree'
    # Efficient construction using combinatorics on exponents
    def _multi_indices(n_vars: int, deg: int):
        # Yield all tuples e = (e1, ... , en) with sum(e) == deg
        if n_vars == 1:
            yield (deg,)
            return
        if deg == 0:
            yield (0,) * n_vars
            return
        # stars and bars enumeration
        from itertools import combinations
        for bars in combinations(range(deg + n_vars - 1), n_vars - 1):
            exps = []
            last = -1
            for b in bars + (deg + n_vars - 1,):
                exps.append(b - last - 1)
                last = b
            yield tuple(exps)

    # degree 1..degree (degree 0 already included as bias)
    for d in range(1, degree + 1):
        for exps in _multi_indices(n_state, d):
            # monomial = prod_j X_j ** exps[j]
            mon = np.ones(n_samples, dtype=X.dtype)
            name_parts = []
            for j, e in enumerate(exps):
                if e > 0:
                    mon *= X[:, j] ** e
                    name_parts.append(f"x{j+1}^{e}" if e > 1 else f"x{j+1}")
            feats.append(mon[:, None])
            names.append("*".join(name_parts) if name_parts else "1")  # falls back to 1 if weird
    return np.hstack(feats), names


def _trig_features(X: np.ndarray, harmonics: int = 1) -> Tuple[np.ndarray, List[str]]:
    """
    Optional trigonometric library: [sin(k * x_i), cos(k * x_i)] for k=1..harmonics.
    Including trig terms is recommended for angle-like states.            [Sec. 3; Eq. (5)]
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
    poly_degree: int = 3               # polynomial order for Θ(X)    (cf. paper examples)
    trig_harmonics: int = 0            # add sin/cos terms per state   (good for angles)
    threshold_lambda: float = 1e-3     # sparsification knob λ         [Sec. 3.1; Code 1]
    max_stlsq_iter: int = 10           # iterations of STLSQ           [Code 1 uses ~10]
    use_savgol_for_xdot: bool = True   # estimate derivatives if xdot not given
    savgol_window: int = 31            # window len for Savitzky–Golay (odd)
    savgol_polyorder: int = 3          # poly order for Savitzky–Golay
    normalize_columns: bool = True     # column-scaling for Θ(X) (improves conditioning)
    include_bias: bool = True          # include constant '1' term


class SINDyRegressor:
    """
    Sparse Identification of Nonlinear Dynamics (continuous-time) with STLSQ.

    API:
        fit(X, t, Xdot=None)      -> learn Ξ and store library info
        predict_derivative(X)     -> Θ(X) @ Ξ
        simulate(x0, t_span)      -> rollout using learned RHS
        sweep_lambda(lambdas, ...) -> accuracy/complexity curve (Pareto)

    Notes tying to paper:
    - Library building mirrors Θ(X) in Eq. (5); we expose polynomials and optional trigs.
    - STLSQ loop is the paper’s Code 1 (sequential thresholded least-squares). [Sec. 3.1]
    - For noisy measurements, we can estimate Xdot via Savitzky–Golay as suggested. [Sec. 3.1]
    - Use λ-sweep to balance error vs. number of active terms (Pareto front). [Sec. 3.2]
    """

    def __init__(self, config: Optional[SINDyConfig] = None):
        self.cfg = config or SINDyConfig()
        self.coef_: Optional[np.ndarray] = None      # Ξ (shape: n_features x n_state)
        self.feature_names_: List[str] = []
        self.column_scale_: Optional[np.ndarray] = None
        self.n_state_: Optional[int] = None

    # ---------- Public API ----------

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        Xdot: Optional[np.ndarray] = None,
    ) -> "SINDyRegressor":
        """
        Fit SINDy model from state time series.

        Args:
            X:    (T, n) state trajectory samples.
            t:    (T,) time stamps (uniformly spaced recommended).
            Xdot: (T, n) derivatives; if None and cfg.use_savgol_for_xdot=True,
                  derivatives are estimated with Savitzky–Golay. [Sec. 3.1]
        """
        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=float)
        assert X.ndim == 2 and t.ndim == 1 and X.shape[0] == t.shape[0], "Shape mismatch X vs t"
        self.n_state_ = X.shape[1]

        if Xdot is None:
            if not self.cfg.use_savgol_for_xdot:
                raise ValueError("Xdot is None and derivative estimation disabled.")
            Xdot = self._estimate_derivative_savgol(X, t)

        # Build feature library Θ(X)
        Theta, names = self._build_library(X)
        self.feature_names_ = names

        # Optional column normalization (improves conditioning)
        if self.cfg.normalize_columns and Theta.shape[1] > 0:
            scale = np.linalg.norm(Theta, axis=0)
            scale[scale == 0] = 1.0
            Theta_scaled = Theta / scale
            self.column_scale_ = scale
        else:
            Theta_scaled = Theta
            self.column_scale_ = np.ones(Theta.shape[1])

        # Solve for sparse coefficients Ξ with STLSQ (column-wise)
        Xi = self._stlsq(Theta_scaled, Xdot, lam=self.cfg.threshold_lambda, max_iter=self.cfg.max_stlsq_iter)

        # Undo scaling on coefficients so prediction uses raw Θ(X)
        Xi_unscaled = Xi / self.column_scale_[:, None]
        self.coef_ = Xi_unscaled
        return self

    def predict_derivative(self, X: np.ndarray) -> np.ndarray:
        """
        Compute f(X) = Θ(X) Ξ, i.e., predicted \dot{X}.
        """
        self._check_fitted()
        Theta, _ = self._build_library(np.asarray(X, dtype=float))
        return Theta @ self.coef_

    def simulate(
        self,
        x0: np.ndarray,
        t_span: np.ndarray,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        method: str = "RK45",
    ) -> np.ndarray:
        """
        Roll out the learned ODE \dot{x}=f(x) from initial state x0 over t_span using solve_ivp.
        The paper compares short and long rollouts to check attractor/energy behavior. [Sec. 4.2]
        """
        self._check_fitted()
        x0 = np.asarray(x0, dtype=float).ravel()
        assert x0.size == self.n_state_, "x0 dimension mismatch"
        t0, tf = float(t_span[0]), float(t_span[-1])

        def rhs(t, x):
            return self.predict_derivative(x[None, :]).ravel()

        sol = solve_ivp(rhs, (t0, tf), x0, t_eval=t_span, rtol=rtol, atol=atol, method=method)
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
        Sweep sparsification λ and compute (error, nnz) on a holdout split
        to trace the Pareto front (accuracy vs. complexity).            [Sec. 3.2]

        Returns:
            dict with arrays: 'lambda', 'val_error', 'nnz'
        """
        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=float)
        if Xdot is None and self.cfg.use_savgol_for_xdot:
            Xdot = self._estimate_derivative_savgol(X, t)
        elif Xdot is None:
            raise ValueError("Xdot is None and derivative estimation disabled.")

        # Simple last-portion holdout (temporal split)
        T = X.shape[0]
        T_val = max(int(val_split * T), 1)
        X_tr, X_val = X[:-T_val], X[-T_val:]
        t_tr, t_val = t[:-T_val], t[-T_val:]
        Xdot_tr, Xdot_val = Xdot[:-T_val], Xdot[-T_val:]

        Theta_tr, names = self._build_library(X_tr)
        Theta_val, _ = self._build_library(X_val)

        # Column scaling from training Θ only (avoid leakage)
        if self.cfg.normalize_columns and Theta_tr.shape[1] > 0:
            scale = np.linalg.norm(Theta_tr, axis=0)
            scale[scale == 0] = 1.0
            Theta_tr_s = Theta_tr / scale
            Theta_val_s = Theta_val / scale
        else:
            scale = np.ones(Theta_tr.shape[1])
            Theta_tr_s, Theta_val_s = Theta_tr, Theta_val

        val_err = []
        nnz_list = []
        for lam in lambdas:
            Xi = self._stlsq(Theta_tr_s, Xdot_tr, lam=lam, max_iter=self.cfg.max_stlsq_iter)
            Xi_unscaled = Xi / scale[:, None]
            # validation error: one-step derivative MSE on holdout
            pred = Theta_val @ Xi_unscaled
            mse = float(np.mean((pred - Xdot_val) ** 2))
            val_err.append(mse)
            nnz_list.append(int(np.sum(np.abs(Xi_unscaled) > 0)))

        return {"lambda": np.array(lambdas), "val_error": np.array(val_err), "nnz": np.array(nnz_list)}

    # ---------- Internals ----------

    def _estimate_derivative_savgol(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Savitzky–Golay differentiation (denoise + derivative) as in many SINDy demos.
        The paper also mentions TV-regularized derivatives as an alternative. [Sec. 3.1]
        """
        dt = float(np.mean(np.diff(t)))
        win = self.cfg.savgol_window
        if win >= X.shape[0]:
            # fallback to smallest odd window < T
            win = max(5, X.shape[0] - (1 - X.shape[0] % 2))
        if win % 2 == 0:
            win += 1
        X_smooth = savgol_filter(X, window_length=win, polyorder=self.cfg.savgol_polyorder, axis=0, mode="interp")
        Xdot = savgol_filter(X, window_length=win, polyorder=self.cfg.savgol_polyorder,
                             deriv=1, delta=dt, axis=0, mode="interp")
        return Xdot

    def _build_library(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Construct Θ(X) = [ bias | poly(X) | trig(X) ], mirroring Eq. (5) in the paper.
        """
        poly_deg = max(self.cfg.poly_degree, 0)
        Theta_list = []
        names: List[str] = []

        # Bias term
        if self.cfg.include_bias:
            Theta_list.append(np.ones((X.shape[0], 1), dtype=X.dtype))
            names.append("1")

        # Polynomial block
        if poly_deg > 0:
            Theta_poly, names_poly = _poly_features(X, poly_deg)
            if not self.cfg.include_bias:
                # remove the duplicated bias from the poly builder
                Theta_poly = Theta_poly[:, 1:]
                names_poly = names_poly[1:]
            Theta_list.append(Theta_poly)
            names.extend(names_poly)

        # Trig block
        if self.cfg.trig_harmonics > 0:
            Theta_trig, names_trig = _trig_features(X, harmonics=self.cfg.trig_harmonics)
            Theta_list.append(Theta_trig)
            names.extend(names_trig)

        if not Theta_list:
            return np.zeros((X.shape[0], 0), dtype=X.dtype), []
        return np.hstack(Theta_list), names

    def _stlsq(
        self,
        Theta: np.ndarray,
        Xdot: np.ndarray,
        lam: float,
        max_iter: int = 10
    ) -> np.ndarray:
        """
        Sequential Thresholded Least Squares (STLSQ) solver.     [Paper Sec. 3.1; Code 1]

        Operates column-wise on Ξ: for each state dimension k,
            Xi[:, k] <- LS on active set
            threshold small |Xi| < λ to zero
            repeat until convergence / max_iter

        Args:
            Theta: (T, p) feature matrix
            Xdot:  (T, n) derivatives
        Returns:
            Xi:    (p, n) sparse coefficients
        """
        T, p = Theta.shape
        n = Xdot.shape[1]
        # Initial LS estimate: Xi = (Theta^+ Xdot)
        Xi, *_ = np.linalg.lstsq(Theta, Xdot, rcond=None)

        for _ in range(max_iter):
            small = np.abs(Xi) < lam
            Xi[small] = 0.0
            # Refit each column on its support
            for k in range(n):
                support = ~small[:, k]
                if np.any(support):
                    # least squares on active support
                    Xi_k, *_ = np.linalg.lstsq(Theta[:, support], Xdot[:, k], rcond=None)
                    Xi[support, k] = Xi_k
                else:
                    Xi[:, k] = 0.0
        return Xi

    def _check_fitted(self):
        if self.coef_ is None:
            raise RuntimeError("SINDyRegressor is not fitted yet. Call fit() first.")
