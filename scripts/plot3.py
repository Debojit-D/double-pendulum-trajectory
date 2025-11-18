# ------------------------------------------------------------------------------
# Plot joint-space trajectories (q1, q2, dq1, dq2) from CSV
# - Uses same CSV format as your tip-trajectory script:
#     t,q1,q2,dq1,dq2,tip_x,tip_y,tip_z,tip_x_rel,tip_z_rel,elbow_x,elbow_z,step_idx
# - Restricts to first T_MAX seconds
# - Uses seaborn "paper" style and saves a PNG
# - SUBPLOTS ARE HORIZONTAL: [ angles | velocities ]
# ------------------------------------------------------------------------------

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- seaborn style (paper-ish) ---
sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=0.9,
)

T_MAX = 10.0          # seconds to plot
ANGLE_IN_DEG = True   # True → convert q to degrees, dq to deg/s

def load_csv(csv_path):
    """
    CSV header expected:
    t,q1,q2,dq1,dq2,tip_x,tip_y,tip_z,tip_x_rel,tip_z_rel,elbow_x,elbow_z,step_idx
    """
    return np.genfromtxt(csv_path, delimiter=",", names=True)

def plot_joint_space(data, out_path=None, show=True):
    t   = data["t"]
    q1  = data["q1"]
    q2  = data["q2"]
    dq1 = data["dq1"]
    dq2 = data["dq2"]

    if ANGLE_IN_DEG:
        q1  = np.degrees(q1)
        q2  = np.degrees(q2)
        dq1 = np.degrees(dq1)
        dq2 = np.degrees(dq2)
        angle_label = "Angle (deg)"
        vel_label   = "Angular velocity (deg/s)"
    else:
        angle_label = "Angle (rad)"
        vel_label   = "Angular velocity (rad/s)"

    # --- HORIZONTAL layout: 1 row, 2 columns ---
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(6.8, 3.0))

    # --- Angles subplot (left) ---
    ax = axes[0]
    ax.plot(t, q1, label="q1", linewidth=1.8)
    ax.plot(t, q2, label="q2", linewidth=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(angle_label)
    ax.set_title(f"Joint angles (0–{T_MAX:.0f} s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # --- Angular velocities subplot (right) ---
    ax = axes[1]
    ax.plot(t, dq1, label="dq1", linewidth=1.8)
    ax.plot(t, dq2, label="dq2", linewidth=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(vel_label)
    ax.set_title(f"Joint velocities (0–{T_MAX:.0f} s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[✓] Saved joint-space plot to: {out_path}")

    if show:
        plt.show()

    plt.close(fig)

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal1/double_pendulum_traj_ideal_run1956.csv"
    if not os.path.isfile(csv_path):
        print(f"[x] CSV not found: {csv_path}")
        sys.exit(1)

    data_full = load_csv(csv_path)

    if "t" not in data_full.dtype.names:
        print("[x] CSV has no 't' column; cannot time-filter.")
        sys.exit(1)

    # Restrict to first T_MAX seconds
    mask = data_full["t"] <= T_MAX
    if not np.any(mask):
        print(f"[x] No samples with t <= {T_MAX} s; nothing to plot.")
        sys.exit(1)

    data = data_full[mask]

    out_png = os.path.join(
        os.path.expanduser("~"),
        f"joint_traj_q_dq_0to{int(T_MAX)}s.png"
    )
    plot_joint_space(data, out_path=out_png, show=True)

if __name__ == "__main__":
    main()
