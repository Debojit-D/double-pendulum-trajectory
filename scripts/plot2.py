# ------------------------------------------------------------------------------
# Plot tip trajectory (2D X–Z) from CSV + initial configuration (read from CSV)
# - Correct Y-axis rotation FK: z = -sin(theta)
# - Frame: base-relative or world (no double shifts)
# - Now: only plot first 10 seconds of data (t <= 10.0)
# - Uses seaborn for styling and saves the figure as PNG
# ------------------------------------------------------------------------------

import sys, os, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- seaborn style (paper-ish) ---
sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=0.9,
)

# --- params (match your sim/export) ---
L1 = 1.0
L2 = 1.0
BASE_X = 0.0
BASE_Z = 2.5
BASE_RELATIVE = False   # True → plot in base frame; False → world frame
T_MAX = 10.0            # seconds to plot

def load_csv(csv_path):
    """
    CSV header expected:
    t,q1,q2,dq1,dq2,tip_x,tip_y,tip_z,tip_x_rel,tip_z_rel,elbow_x,elbow_z,step_idx
    Angles q1,q2 are in radians (as saved by your recorder).
    """
    return np.genfromtxt(csv_path, delimiter=",", names=True)

def fk_yaxis_xz(q1, q2, L1, L2, base_x, base_z):
    """
    Forward kinematics in X–Z plane for joints rotating about +Y.
    Uses MuJoCo's right-handed convention: z = -sin(theta).
    q1, q2 in radians; q2 is RELATIVE to link1.
    Returns p0(base), p1(elbow), p2(tip).
    """
    th1 = q1
    th12 = q1 + q2

    p0 = (base_x, base_z)
    p1 = (base_x + L1 * math.cos(th1), base_z - L1 * math.sin(th1))
    p2 = (p1[0] + L2 * math.cos(th12), p1[1] - L2 * math.sin(th12))
    return p0, p1, p2

def plot_tip_with_overlay(data, base_relative, out_path=None, show=True):
    # Choose trajectory columns directly (avoid recomputation)
    if base_relative:
        x = data["tip_x_rel"]
        z = data["tip_z_rel"]
        base_x, base_z = 0.0, 0.0
        title = "Double Pendulum — Tip Trajectory (X–Z, Base-relative, first 10 s)"
        base_label = "Base (0, 0)"
    else:
        x = data["tip_x"]
        z = data["tip_z"]
        base_x, base_z = BASE_X, BASE_Z
        title = (
            "Double Pendulum — Tip Trajectory (X–Z, World, first 10 s)\n"
            f"Base at ({BASE_X:.2f}, {BASE_Z:.2f})"
        )
        base_label = f"Base ({BASE_X:.2f}, {BASE_Z:.2f})"

    # Initial joint angles from the recorded data (radians)
    q1_0 = float(data["q1"][0])
    q2_0 = float(data["q2"][0])
    q1_deg = math.degrees(q1_0)
    q2_deg = math.degrees(q2_0)

    # Initial configuration overlay (in the SAME frame as the trajectory)
    p0, p1, p2 = fk_yaxis_xz(q1_0, q2_0, L1, L2, base_x, base_z)

    # Markers
    x0, z0 = x[0], z[0]       # start (t ≈ 0)
    xN, zN = x[-1], z[-1]     # end (t ≤ 10 s)

    plt.figure(figsize=(6.6, 6.6))
    plt.plot(x, z, linewidth=1.8, label="Tip trajectory (0–10 s)")

    # Dotted initial links
    plt.plot(
        [p0[0], p1[0]], [p0[1], p1[1]],
        "k:", linewidth=2.0, label=f"Link 1 ({q1_deg:.1f}°)"
    )
    plt.plot(
        [p1[0], p2[0]], [p1[1], p2[1]],
        "k:", linewidth=2.0, label=f"Link 2 (+{q2_deg:.1f}° rel.)"
    )

    plt.scatter([p0[0]], [p0[1]], s=40, color="k", label=base_label)
    plt.scatter([x0], [z0], s=36, marker="^", label="Start tip")
    plt.scatter([xN], [zN], s=36, marker="s", label="End tip (≤10 s)")

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    # --- save figure if path provided ---
    if out_path is not None:
        # ensure directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[✓] Saved plot to: {out_path}")

    if show:
        plt.show()

    plt.close()

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal1/double_pendulum_traj_ideal_run1571.csv"
    if not os.path.isfile(csv_path):
        print(f"[x] CSV not found: {csv_path}")
        sys.exit(1)

    data_full = load_csv(csv_path)

    # ---- restrict to first T_MAX seconds ----
    if "t" not in data_full.dtype.names:
        print("[x] CSV has no 't' column; cannot time-filter.")
        sys.exit(1)

    mask = data_full["t"] <= T_MAX
    if not np.any(mask):
        print(f"[x] No samples with t <= {T_MAX} s; nothing to plot.")
        sys.exit(1)

    data = data_full[mask]

    suffix = "base" if BASE_RELATIVE else "world"
    out_png = os.path.join(
        os.path.expanduser("~"),
        f"tip_traj_xz_with_links_{suffix}_0to{int(T_MAX)}s.png"
    )
    plot_tip_with_overlay(data, BASE_RELATIVE, out_path=out_png, show=True)

if __name__ == "__main__":
    main()
