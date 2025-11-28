#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint-space comparison plot: original vs rollout (first 10 seconds).

Run simply as:
    python3 plot_joint_comparison.py

Config:
    - ORIG_CSV:    ground-truth CSV
    - ROLLOUT_CSV: rollout CSV (LNN, etc.)
    - OUT_PNG:     output figure path

Requirements:
    - original CSV must have:  t, q1, q2, dq1, dq2
    - rollout CSV should have either:
        - the same names: q1, q2, dq1, dq2
          OR
        - predicted names: q1_pred, q2_pred, dq1_pred, dq2_pred
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------- CONFIG: EDIT THESE -----------------------
ORIG_CSV    = "/home/iitgn-robotics/path/to/original.csv"
ROLLOUT_CSV = "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal2/double_pendulum_traj_ideal_run002.csv"
OUT_PNG     = "joint_space_comparison_0to10s.png"
T_MAX       = 10.0   # seconds
# -----------------------------------------------------------------


def _get_col(arr, candidates):
    """Return the first existing column from `candidates`."""
    cols = arr.dtype.names
    for name in candidates:
        if name in cols:
            return arr[name]
    raise KeyError(f"None of the columns {candidates} found in CSV. Available: {cols}")


def main():
    # ---------- Load CSVs ----------
    orig = np.genfromtxt(ORIG_CSV, delimiter=",", names=True, dtype=float)
    roll = np.genfromtxt(ROLLOUT_CSV, delimiter=",", names=True, dtype=float)

    if "t" not in orig.dtype.names:
        raise KeyError(f"'t' column not found in original CSV: {ORIG_CSV}")

    # Use original time as reference and clip to first T_MAX seconds
    t_orig = orig["t"]
    mask_10 = t_orig <= T_MAX
    if not np.any(mask_10):
        raise RuntimeError(f"No samples within first {T_MAX} s in {ORIG_CSV}")

    t = t_orig[mask_10]
    N = len(t)

    # ---------- Extract ground-truth (clipped to 0–T_MAX) ----------
    q1_true  = _get_col(orig, ["q1"])[mask_10]
    q2_true  = _get_col(orig, ["q2"])[mask_10]
    dq1_true = _get_col(orig, ["dq1"])[mask_10]
    dq2_true = _get_col(orig, ["dq2"])[mask_10]

    # ---------- Extract rollout (predicted) ----------
    # Assume rollout is sampled at least as long; just take first N samples
    q1_pred  = _get_col(roll, ["q1", "q1_pred"])[:N]
    q2_pred  = _get_col(roll, ["q2", "q2_pred"])[:N]
    dq1_pred = _get_col(roll, ["dq1", "dq1_pred"])[:N]
    dq2_pred = _get_col(roll, ["dq2", "dq2_pred"])[:N]

    # ---------- Plot ----------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=160)
    ax11, ax12, ax21, ax22 = axes.flatten()

    ax11.plot(t, q1_true, label="q1 true")
    ax11.plot(t, q1_pred, "--", label="q1 rollout")
    ax11.set_xlabel("t [s]")
    ax11.set_ylabel("q1 [rad]")
    ax11.legend()
    ax11.set_title("q1 (0–10 s)")

    ax12.plot(t, q2_true, label="q2 true")
    ax12.plot(t, q2_pred, "--", label="q2 rollout")
    ax12.set_xlabel("t [s]")
    ax12.set_ylabel("q2 [rad]")
    ax12.legend()
    ax12.set_title("q2 (0–10 s)")

    ax21.plot(t, dq1_true, label="dq1 true")
    ax21.plot(t, dq1_pred, "--", label="dq1 rollout")
    ax21.set_xlabel("t [s]")
    ax21.set_ylabel("dq1 [rad/s]")
    ax21.legend()
    ax21.set_title("dq1 (0–10 s)")

    ax22.plot(t, dq2_true, label="dq2 true")
    ax22.plot(t, dq2_pred, "--", label="dq2 rollout")
    ax22.set_xlabel("t [s]")
    ax22.set_ylabel("dq2 [rad/s]")
    ax22.legend()
    ax22.set_title("dq2 (0–10 s)")

    fig.suptitle("Joint-Space Comparison (First 10 s): Original vs Rollout")
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    plt.close(fig)

    print(f"[✓] Saved joint space comparison (0–{T_MAX}s) to: {OUT_PNG}")


if __name__ == "__main__":
    main()
