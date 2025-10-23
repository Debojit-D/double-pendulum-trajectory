# ------------------------------------------------------------------------------
# Load & run the Double Pendulum MJCF in MuJoCo viewer
# - Hardcode initial joint state (q1, q2 in DEGREES -> converted to radians)
# - Prints a FK sanity check at t=0 (does link-2 use q1+q2 or q1−q2?)
# ------------------------------------------------------------------------------

import time
import math
import numpy as np
import mujoco
from mujoco.viewer import launch

XML_PATH = "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/description/double_pendulum.xml"

# --------- Hardcoded initial state (DEGREES and RAD/S) ----------
Q1_DEG = 270.0   # link-1 hanging down if axis="0 1 0" and link geom along +X
Q2_DEG = 80.0    # link-2 relative angle
Q1D    = 0.0     # rad/s
Q2D    = 0.0     # rad/s
# ---------------------------------------------------------------

# Link lengths from your MJCF (each box length ≈ 1.0 m)
L1 = 1.0
L2 = 1.0
BASE = np.array([0.0, 0.0, 2.5], dtype=float)  # body A pos="0 0 2.5"

def set_initial_state(model, data, q1_deg, q2_deg, q1d=0.0, q2d=0.0):
    """Set initial joint positions (in degrees) and velocities (rad/s)."""
    q = np.deg2rad([q1_deg, q2_deg])  # MuJoCo uses radians
    data.qpos[0] = float(q[0])
    data.qpos[1] = float(q[1])
    data.qvel[0] = float(q1d)
    data.qvel[1] = float(q2d)
    mujoco.mj_forward(model, data)    # propagate state to all derived quantities

def get_tip_pos(model, data, geom_name="tip_marker"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise RuntimeError(f'Geom "{geom_name}" not found.')
    return np.array(data.geom_xpos[gid], dtype=float)

def rotY_x(vx, q):
    """Rotate a vector (vx,0,0) about +Y by angle q (right-hand rule)."""
    return np.array([math.cos(q)*vx, 0.0, math.sin(q)*vx], dtype=float)

# Load model/data
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Set your hardcoded initial state
set_initial_state(model, data, Q1_DEG, Q2_DEG, Q1D, Q2D)

# ------------------------ FK SANITY CHECK (t=0) ------------------------
q1 = float(data.qpos[0])
q2 = float(data.qpos[1])

tip_meas = get_tip_pos(model, data)

# Option A: θ2(abs) = q1 + q2
p1_A  = BASE + rotY_x(L1, q1)
tip_A = p1_A + rotY_x(L2, q1 + q2)

# Option B: θ2(abs) = q1 − q2
p1_B  = BASE + rotY_x(L1, q1)
tip_B = p1_B + rotY_x(L2, q1 - q2)

err_A = np.linalg.norm(tip_meas - tip_A)
err_B = np.linalg.norm(tip_meas - tip_B)

print("\n--- FK sanity check at t=0 ---")
print(f"q1 = {q1:.10f} rad ({np.rad2deg(q1):.3f}°), q2 = {q2:.10f} rad ({np.rad2deg(q2):.3f}°)")
print("tip_meas    :", tip_meas)
print("tip_pred A  :", tip_A, f" | err_A = {err_A:.6e}  (θ2_abs = q1 + q2)")
print("tip_pred B  :", tip_B, f" | err_B = {err_B:.6e}  (θ2_abs = q1 - q2)")
print("=> Matches:", "A (q1+q2)" if err_A < err_B else "B (q1−q2)")
print("----------------------------------------------------------------\n")
# ---------------------------------------------------------------------

# Run viewer
with launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.render()
        time.sleep(model.opt.timestep)
