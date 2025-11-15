# ------------------------------------------------------------------------------
# Plot tip trajectory (2D X–Z) from CSV + initial configuration (read from CSV)
# - Correct Y-axis rotation FK: z = -sin(theta)
# - Frame: base-relative or world (no double shifts)
# ------------------------------------------------------------------------------

import sys, os, math
import numpy as np
import matplotlib.pyplot as plt

# --- params (match your sim/export) ---
L1 = 1.0
L2 = 1.0
BASE_X = 0.0
BASE_Z = 2.5
BASE_RELATIVE = False   # True → plot in base frame; False → world frame

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
        title = "Double Pendulum — Tip Trajectory (X–Z, Base-relative)"
        base_label = "Base (0, 0)"
    else:
        x = data["tip_x"]
        z = data["tip_z"]
        base_x, base_z = BASE_X, BASE_Z
        title = f"Double Pendulum — Tip Trajectory (X–Z, World)\nBase at ({BASE_X:.2f}, {BASE_Z:.2f})"
        base_label = f"Base ({BASE_X:.2f}, {BASE_Z:.2f})"

    # Initial joint angles from the recorded data (radians)
    q1_0 = float(data["q1"][0])
    q2_0 = float(data["q2"][0])
    q1_deg = math.degrees(q1_0)
    q2_deg = math.degrees(q2_0)

    # Initial configuration overlay (in the SAME frame as the trajectory)
    p0, p1, p2 = fk_yaxis_xz(q1_0, q2_0, L1, L2, base_x, base_z)

    # Markers
    x0, z0 = x[0], z[0]
    xN, zN = x[-1], z[-1]

    plt.figure(figsize=(6.6, 6.6))
    plt.plot(x, z, linewidth=1.8, label="Tip trajectory")

    # Dotted initial links
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "k:", linewidth=2.0, label=f"Link 1 ({q1_deg:.1f}°)")
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k:", linewidth=2.0, label=f"Link 2 (+{q2_deg:.1f}° rel.)")

    plt.scatter([p0[0]], [p0[1]], s=40, color="k", label=base_label)
    plt.scatter([x0], [z0], s=36, marker="^", label="Start tip")
    plt.scatter([xN], [zN], s=36, marker="s", label="End tip")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X (m)"); plt.ylabel("Z (m)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    # if out_path:
    #     plt.savefig(out_path, dpi=200)
    #     print(f"[✓] Saved plot to: {out_path}")
    if show:
        plt.show()
    plt.close()

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal1/double_pendulum_traj_ideal_run1956.csv"
    if not os.path.isfile(csv_path):
        print(f"[x] CSV not found: {csv_path}"); sys.exit(1)

    data = load_csv(csv_path)
    suffix = "base" if BASE_RELATIVE else "world"
    out_png = os.path.join(os.path.dirname(csv_path), f"tip_traj_xz_with_links_{suffix}.png")
    plot_tip_with_overlay(data, BASE_RELATIVE, out_path=out_png, show=True)

if __name__ == "__main__":
    main()
