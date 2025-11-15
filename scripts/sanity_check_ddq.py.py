#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanity check for double-pendulum acceleration targets.

- Loads one trajectory CSV
- Applies same t_max + stride as train_lnn.py
- Computes ddq via Savitzky–Golay (2nd derivative of q)
- Computes ddq via simple finite difference of dq
- Prints stats and plots ddq_sg vs ddq_fd for both joints

Use this to see if your SG ddq targets are insanely noisy / huge.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ----------------- match train_lnn.py settings -----------------
T_MAX_TRAIN = 5.0
STRIDE = 5
SAVGOL_WINDOW = 51
SAVGOL_POLYORDER = 3

REQ_COLS = ["t", "q1", "q2", "dq1", "dq2"]


def load_csv(csv_path):
    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    for c in REQ_COLS:
        if c not in arr.dtype.names:
            raise ValueError(f"[DATA] Missing required column '{c}' in {csv_path}")
    return arr


def robust_savgol_window(T: int, requested: int, poly: int) -> int:
    """
    Same logic as train_lnn._robust_savgol_window:
    return an odd window length suitable for Savitzky–Golay.
    """
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


def estimate_ddq_savgol(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Savitzky–Golay 2nd derivative on q to get ddq, mimicking train_lnn.
    q: (N, 2), t: (N,)
    returns ddq: (N, 2)
    """
    dt = float(np.mean(np.diff(t)))
    if not np.all(np.diff(t) > 0):
        raise ValueError("[TIME] t must be strictly increasing for SavGol differentiation.")

    win = robust_savgol_window(len(t), SAVGOL_WINDOW, SAVGOL_POLYORDER)
    ddq = savgol_filter(
        q,
        window_length=win,
        polyorder=SAVGOL_POLYORDER,
        deriv=2,
        delta=dt,
        axis=0,
        mode="interp",
    )
    return ddq


def main():
    # ---------- pick CSV ----------
    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal1/double_pendulum_traj_ideal_run1956.csv"
    )

    if not os.path.isfile(csv_path):
        print(f"[x] CSV not found: {csv_path}")
        sys.exit(1)

    print(f"[INFO] Loading: {csv_path}")
    data = load_csv(csv_path)

    # ---------- clamp to first T_MAX_TRAIN seconds ----------
    t = data["t"].astype(float)
    mask = t <= float(T_MAX_TRAIN)
    if not np.any(mask):
        print(f"[x] No samples with t <= {T_MAX_TRAIN}s")
        sys.exit(1)

    t = t[mask]
    q = np.vstack([data["q1"][mask], data["q2"][mask]]).T  # (N, 2)
    dq = np.vstack([data["dq1"][mask], data["dq2"][mask]]).T  # (N, 2)

    # ---------- apply same stride as training ----------
    if STRIDE > 1:
        sl = slice(0, None, STRIDE)
        t = t[sl]
        q = q[sl]
        dq = dq[sl]

    N = len(t)
    print(f"[INFO] After t_max and stride: N = {N} samples")

    # ---------- compute dt ----------
    dt_all = np.diff(t)
    dt_mean = float(np.mean(dt_all))
    dt_min = float(np.min(dt_all))
    dt_max = float(np.max(dt_all))
    print(f"[INFO] dt mean={dt_mean:.6f}, min={dt_min:.6f}, max={dt_max:.6f}")

    # ---------- ddq via Savitzky–Golay on q ----------
    ddq_sg = estimate_ddq_savgol(q, t)  # (N, 2)

    # ---------- ddq via finite difference of dq ----------
    #   ddq_fd[k] ~ (dq[k+1] - dq[k]) / dt_k
    ddq_fd = np.diff(dq, axis=0) / dt_all[:, None]  # (N-1, 2)
    t_fd = t[1:]  # align with ddq_fd

    # ---------- basic stats ----------
    def stats(name, arr):
        print(
            f"{name:10s}: min={np.min(arr):.3e}  max={np.max(arr):.3e}  "
            f"mean={np.mean(arr):.3e}  std={np.std(arr):.3e}"
        )


    print("\n[STATS] Savitzky–Golay ddq (full N samples)")
    stats("ddq1_sg", ddq_sg[:, 0])
    stats("ddq2_sg", ddq_sg[:, 1])

    print("\n[STATS] Finite-diff ddq (N-1 samples)")
    stats("ddq1_fd", ddq_fd[:, 0])
    stats("ddq2_fd", ddq_fd[:, 1])

    # ---------- plots ----------
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle("Double Pendulum Acceleration Sanity Check (SG vs Finite Diff)")

    # Joint 1
    ax = axes[0]
    ax.plot(t, ddq_sg[:, 0], label="ddq1_savgol", linewidth=1.0)
    ax.plot(t_fd, ddq_fd[:, 0], label="ddq1_fd", linewidth=1.0, alpha=0.7)
    ax.set_ylabel("ddq1 (rad/s²)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Joint 2
    ax = axes[1]
    ax.plot(t, ddq_sg[:, 1], label="ddq2_savgol", linewidth=1.0)
    ax.plot(t_fd, ddq_fd[:, 1], label="ddq2_fd", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("time t (s)")
    ax.set_ylabel("ddq2 (rad/s²)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
