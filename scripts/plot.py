# ------------------------------------------------------------------------------
# Plot tip trajectory (2D X–Z) from CSV + initial configuration (270°, 80°)
# Base of pendulum is at world position (0, 2.5)
# ------------------------------------------------------------------------------

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
L1 = 1.0
L2 = 1.0

Q1_DEG = 270.0   # link 1 down from +X axis
Q2_DEG = 80.0    # link 2 relative to link 1

BASE_X = 0.0
BASE_Z = 2.5     # pivot height in world coordinates

# -------------------------------
# Data loading
# -------------------------------
def load_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    tip_x = data["tip_x"]
    tip_z = data["tip_z"]
    return tip_x, tip_z

# -------------------------------
# Forward kinematics for static overlay
# -------------------------------
def fk_links(q1_deg, q2_deg, L1, L2, base_x, base_z):
    """Compute static positions of link1 and link2 in X–Z plane."""
    th1 = np.deg2rad(q1_deg)
    th2 = np.deg2rad(q1_deg + q2_deg)

    p0 = (base_x, base_z)
    p1 = (base_x + L1 * np.cos(th1), base_z + L1 * np.sin(th1))
    p2 = (p1[0] + L2 * np.cos(th2), p1[1] + L2 * np.sin(th2))
    return p0, p1, p2

# -------------------------------
# Plot function
# -------------------------------
def plot_tip_trajectory_with_links(x, z, p0, p1, p2, base_relative=False, out_path=None, show=True):
    plt.figure(figsize=(6, 6))
    plt.plot(x, z, linewidth=1.8, label="Tip trajectory")

    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "k:", linewidth=2.0, label="Link 1 (270°)")
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k:", linewidth=2.0, label="Link 2 (80° rel.)")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")

    title = "Double Pendulum — Tip Trajectory (X–Z)"
    if base_relative:
        title += "\n(Base-relative coordinates)"
    else:
        title += "\n(World coordinates, base at (0, 2.5))"

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"[✓] Saved plot to: {out_path}")
    if show:
        plt.show()
    plt.close()

# -------------------------------
# Main
# -------------------------------
def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/description/double_pendulum_traj.csv"
    
    if not os.path.isfile(csv_path):
        print(f"[x] CSV not found: {csv_path}")
        sys.exit(1)

    tip_x, tip_z = load_csv(csv_path)

    # --- Choose frame ---
    BASE_RELATIVE = True   # ✅ Set True for pivot-origin plot, False for world-space plot

    if BASE_RELATIVE:
        # Shift so base is at (0, 0)
        x_plot = tip_x - BASE_X
        z_plot = tip_z - BASE_Z
        p0, p1, p2 = fk_links(Q1_DEG, Q2_DEG, L1, L2, 0.0, 0.0)
    else:
        # Keep world coordinates
        x_plot = tip_x
        z_plot = tip_z
        p0, p1, p2 = fk_links(Q1_DEG, Q2_DEG, L1, L2, BASE_X, BASE_Z)

    out_png = os.path.join(os.path.dirname(csv_path),
                           "tip_traj_xz_with_links_corrected.png")

    plot_tip_trajectory_with_links(
        x_plot, z_plot, p0, p1, p2,
        base_relative=BASE_RELATIVE,
        out_path=out_png,
        show=True
    )

if __name__ == "__main__":
    main()
