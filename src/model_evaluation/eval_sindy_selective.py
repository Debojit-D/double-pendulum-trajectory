#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hard-coded evaluation for the saved SINDy model (double pendulum).

- Loads model from MODEL_DIR
- Evaluates a chosen subset of CSVs listed in MANIFEST (under DATA_DIR)
- Simulates with solve_ivp and computes RMSE(q), RMSE(dq), derivative MSE
- Saves:
    eval_all/eval_metrics.csv
    eval_all/<csvstem>_timeseries.png  (if MAKE_PLOTS)
    eval_all/top_features.png
    eval_all/summary.json
"""

from __future__ import annotations
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

# ===================== HARD-CODED PATHS / OPTIONS =====================
MODEL_DIR  = Path("/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal2/sindy_model")
DATA_DIR   = Path("/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal2")
MANIFEST   = Path("/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal2/double_pendulum_manifest_ideal.json")
OUT_DIR    = Path("/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal2/sindy_model/eval_all")

# Which files to evaluate: "manifest_all" | "heldout_only" | "first_k" | "random_k"
EVAL_MODE     = "heldout_only"
K             = 5                # used when EVAL_MODE is first_k or random_k
SHUFFLE_SEED  = 42               # used when EVAL_MODE is random_k

# Simulation / metrics knobs (kept aligned with your training defaults)
SIM_METHOD    = "Radau"     # ["RK45","Radau","BDF","LSODA"]
SIM_RTOL      = 1e-4
SIM_ATOL      = 1e-6
SIM_MAX_STEP  = 0.02        # passed only if your SINDy.simulate supports it
ANGLE_WRAP    = True        # wrap q to (-pi, pi] for error calc
ANGLE_IDX     = (0, 1)      # which states are angles
SAVGOL_WIN    = 31          # for ddq reconstruction (same as train if you matched it)
SAVGOL_ORDER  = 3
DECIM_STRIDE  = 1           # decimate signals before sim
BURNIN_SEC    = 0.0         # ignore first seconds when computing RMSE
MAKE_PLOTS    = True

# Optional guard-rails (only if your SINDy.simulate supports them; otherwise auto-fallback)
CLIP_Q        = np.pi
CLIP_DQ       = 50.0
WRAP_ANGLES_DURING_SIM = True
# =====================================================================

# --- Make project root importable so we can reuse training helpers ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_training.sindy import SINDyRegressor, SINDyConfig
from src.training.train_sindy import _build_X_Xdot_from_csv, _wrap_to_pi


def _collect_files(data_dir: Path, manifest: Path) -> list[Path]:
    """Gather CSV list from manifest (preferred)."""
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    meta = json.loads(manifest.read_text())
    files = [data_dir / r["csv"] for r in meta.get("runs", [])]
    if not files:
        raise RuntimeError(f"No runs listed in manifest: {manifest}")
    return files


def _choose_files(data_dir: Path, manifest: Path, model_dir: Path) -> list[Path]:
    """Pick files according to EVAL_MODE with robust fallbacks."""
    all_files = _collect_files(data_dir, manifest)

    if EVAL_MODE == "heldout_only":
        # Prefer config.json → list of heldout file names
        cfg_path = model_dir / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                hr = cfg.get("heldout_runs", None)
                if isinstance(hr, list) and all(isinstance(x, str) for x in hr):
                    return [data_dir / name for name in hr if (data_dir / name).exists()]
            except Exception:
                pass

        # Fallback to metrics.json → holdout_results[].csv
        met_path = model_dir / "metrics.json"
        if met_path.exists():
            try:
                m = json.loads(met_path.read_text())
                if isinstance(m.get("heldout_runs", None), list):
                    hr_list = m["heldout_runs"]
                    names = [
                        (r["csv"] if isinstance(r, dict) and "csv" in r else r)
                        for r in hr_list
                    ]
                else:
                    # Common case: metrics["heldout_runs"] is int; derive from holdout_results
                    res = m.get("holdout_results", [])
                    names = [r["csv"] for r in res if isinstance(r, dict) and "csv" in r]

                names = [n for n in names if isinstance(n, str)]
                if names:
                    return [data_dir / n for n in names if (data_dir / n).exists()]
            except Exception:
                pass

        print("[WARN] Could not resolve held-out list from config/metrics; using manifest_all.")
        return all_files

    if EVAL_MODE == "first_k":
        return all_files[:max(0, min(K, len(all_files)))]

    if EVAL_MODE == "random_k":
        rng = np.random.default_rng(SHUFFLE_SEED)
        if len(all_files) == 0:
            return []
        idx = rng.choice(len(all_files), size=min(K, len(all_files)), replace=False)
        return [all_files[i] for i in idx]

    # default: manifest_all
    return all_files


def _load_model(model_dir: Path) -> SINDyRegressor:
    """
    Load a trained SINDy model from MODEL_DIR.

    Important: we must recreate SINDyConfig with the SAME library
    structure as during training (physics library, angle indices, etc.).
    Otherwise Theta and coef_ will have incompatible shapes.
    """
    coef = np.load(model_dir / "coef.npy")
    names = (model_dir / "feature_names.txt").read_text().strip().splitlines()
    cfg_json = json.loads((model_dir / "config.json").read_text())

    # Rebuild config: mirror training-time settings + physics library.
    cfg = SINDyConfig(
        poly_degree=int(cfg_json["poly_degree"]),
        trig_harmonics=int(cfg_json["trig_harmonics"]),
        threshold_lambda=float(cfg_json["threshold_lambda"]),
        max_stlsq_iter=int(cfg_json["max_stlsq_iter"]),
        use_savgol_for_xdot=False,  # we supply Xdot explicitly in training/eval
        savgol_window=int(cfg_json["savgol_window"]),
        savgol_polyorder=int(cfg_json["savgol_polyorder"]),
        normalize_columns=bool(cfg_json["normalize_columns"]),
        include_bias=bool(cfg_json["include_bias"]),
        backend="numpy",   # evaluation on CPU is fine and robust
        device=None,
        # --- physics-aware library flags (MUST match training) ---
        use_physics_library=True,
        angle_idx=(0, 1),
    )

    model = SINDyRegressor(cfg)
    # Attach trained parameters + metadata
    model.coef_ = coef
    model.feature_names_ = names
    model.n_state_ = 4  # [q1, q2, dq1, dq2]

    # Sanity check: #features must match coef rows
    if coef.shape[0] != len(names):
        raise ValueError(
            f"[LOAD] coef rows ({coef.shape[0]}) != #feature_names ({len(names)}). "
            "Check that feature_names.txt and coef.npy come from the same training run."
        )

    return model


def _save_metrics_csv(rows: list[list[object]], out_csv: Path):
    header = "csv,rmse_q,rmse_dq,derivative_mse,steps"
    arr = np.array(rows, dtype=object)
    np.savetxt(out_csv, arr, fmt="%s", delimiter=",", header=header, comments="")


def _save_summary_json(rows: list[list[object]], out_json: Path):
    if not rows:
        out_json.write_text(json.dumps({"count": 0}, indent=2))
        return
    vals_q  = np.array([float(r[1]) for r in rows], dtype=float)
    vals_dq = np.array([float(r[2]) for r in rows], dtype=float)
    vals_dm = np.array([float(r[3]) for r in rows], dtype=float)
    summary = dict(
        count=len(rows),
        rmse_q_mean=float(vals_q.mean()),   rmse_q_median=float(np.median(vals_q)),
        rmse_dq_mean=float(vals_dq.mean()), rmse_dq_median=float(np.median(vals_dq)),
        deriv_mse_mean=float(vals_dm.mean()), deriv_mse_median=float(np.median(vals_dm)),
    )
    out_json.write_text(json.dumps(summary, indent=2))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = _load_model(MODEL_DIR)
    files = _choose_files(DATA_DIR, MANIFEST, MODEL_DIR)
    if not files:
        print("[WARN] No files selected for evaluation.")
        return

    rows = []
    for p in files:
        Xh, Xdot_h, th = _build_X_Xdot_from_csv(
            p, angle_wrap=ANGLE_WRAP,
            savgol_window=SAVGOL_WIN,
            savgol_polyorder=SAVGOL_ORDER,
            stride=max(1, DECIM_STRIDE),
            debug_dump=None
        )

        x0 = Xh[0]
        th_sim = th

        sim_kwargs = dict(method=SIM_METHOD, rtol=SIM_RTOL, atol=SIM_ATOL)
        extended_kwargs = dict(
            max_step=SIM_MAX_STEP,
            wrap_angles=WRAP_ANGLES_DURING_SIM,
            angle_idx=ANGLE_IDX,
            clip_q=CLIP_Q,
            clip_dq=CLIP_DQ,
        )

        # Try extended signature first; fall back to basic if class doesn't support it.
        try:
            y = model.simulate(x0, th_sim, **sim_kwargs, **extended_kwargs)
        except TypeError:
            y = model.simulate(x0, th_sim, **sim_kwargs)

        # Burn-in handling
        rel_t = th_sim - th_sim[0]
        k0 = int(np.searchsorted(rel_t, BURNIN_SEC)) if BURNIN_SEC > 0 else 0

        # Angle wrapping for error
        true_q = _wrap_to_pi(Xh[:, :2]) if ANGLE_WRAP else Xh[:, :2]
        pred_q = _wrap_to_pi(y[:, :2])  if ANGLE_WRAP else y[:, :2]

        rmse_q  = float(np.sqrt(np.mean((pred_q[k0:] - true_q[k0:])**2)))
        rmse_dq = float(np.sqrt(np.mean((y[k0:, 2:] - Xh[k0:, 2:])**2)))

        # One-step derivative fit quality
        f_pred = model.predict_derivative(Xh)
        deriv_mse = float(np.mean((f_pred[k0:] - Xdot_h[k0:])**2))

        rows.append([p.name, rmse_q, rmse_dq, deriv_mse, len(th_sim)])
        print(f"[EVAL] {p.name}: RMSE(q)={rmse_q:.4e}, RMSE(dq)={rmse_dq:.4e}, d/dt MSE={deriv_mse:.4e}")

        if MAKE_PLOTS:
            fig = plt.figure(figsize=(11, 8))
            ax = plt.subplot(2, 2, 1)
            ax.plot(th_sim, true_q[:, 0], label="q1 true")
            ax.plot(th_sim, pred_q[:, 0], "--", label="q1 pred")
            ax.set_title("q1")
            ax.legend()

            ax = plt.subplot(2, 2, 2)
            ax.plot(th_sim, true_q[:, 1], label="q2 true")
            ax.plot(th_sim, pred_q[:, 1], "--", label="q2 pred")
            ax.set_title("q2")
            ax.legend()

            ax = plt.subplot(2, 2, 3)
            ax.plot(th_sim, Xh[:, 2], label="dq1 true")
            ax.plot(th_sim, y[:, 2], "--", label="dq1 pred")
            ax.set_title("dq1")
            ax.legend()

            ax = plt.subplot(2, 2, 4)
            ax.plot(th_sim, Xh[:, 3], label="dq2 true")
            ax.plot(th_sim, y[:, 3], "--", label="dq2 pred")
            ax.set_title("dq2")
            ax.legend()

            fig.suptitle(p.name)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{p.stem}_timeseries.png", dpi=160)
            plt.close(fig)

    # Save summary CSV + JSON
    _save_metrics_csv(rows, OUT_DIR / "eval_metrics.csv")
    _save_summary_json(rows, OUT_DIR / "summary.json")
    print(f"[✓] Wrote evaluation summary: {OUT_DIR/'eval_metrics.csv'}")
    print(f"[✓] Wrote aggregate summary:  {OUT_DIR/'summary.json'}")

    # Top features bar-plot
    try:
        Xi = model.coef_
        mags = np.abs(Xi).sum(axis=1)  # total magnitude across states
        idx = np.argsort(mags)[::-1][:20]
        labels = [model.feature_names_[i] for i in idx]
        fig = plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(idx)), mags[idx])
        plt.xticks(np.arange(len(idx)), labels, rotation=60, ha="right")
        plt.title("Top 20 features by |coef| sum across states")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "top_features.png", dpi=160)
        plt.close(fig)
        print(f"[✓] Wrote feature plot:     {OUT_DIR/'top_features.png'}")
    except Exception as e:
        print(f"[WARN] feature barplot skipped: {e}")

    print(f"[✓] Plots & metrics saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
