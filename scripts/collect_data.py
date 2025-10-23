# ------------------------------------------------------------------------------
# Double Pendulum: run 30 s and save trajectory to CSV
# Columns: t, q1, q2, dq1, dq2, tip_x, tip_y, tip_z, tip_x_rel, tip_z_rel,
#          elbow_x, elbow_z, step_idx
# ------------------------------------------------------------------------------

import os
import numpy as np
import mujoco

XML_PATH = "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/description/double_pendulum.xml"
OUT_CSV  = os.path.join(os.path.dirname(XML_PATH), "double_pendulum_traj.csv")

TIP_GEOM_NAME = "tip_marker"     # from your MJCF
BASE_X, BASE_Z = 0.0, 2.5        # body A pos="0 0 2.5"
L1 = 1.0                         # link-1 length (your box size .5 => ~1.0)
L2 = 1.0                         # link-2 length (for reference only)

# --------- Initial state (DEGREES and RAD/S) ----------
Q1_DEG = 270.0
Q2_DEG = 80.0
Q1D    = 0.0
Q2D    = 0.0
SIM_TIME = 30.0  # seconds
# ------------------------------------------------------

def set_initial_state(model, data, q1_deg, q2_deg, q1d=0.0, q2d=0.0):
    """Set initial joint positions (deg) and velocities (rad/s)."""
    q = np.deg2rad([q1_deg, q2_deg])
    data.qpos[0] = float(q[0])
    data.qpos[1] = float(q[1])
    data.qvel[0] = float(q1d)
    data.qvel[1] = float(q2d)
    mujoco.mj_forward(model, data)

def main():
    # Load model & data
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # Cache ids we use every step
    tip_gid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TIP_GEOM_NAME)
    if tip_gid < 0:
        raise RuntimeError(f'Geom "{TIP_GEOM_NAME}" not found. Check the MJCF.')

    bodyA_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A")
    bodyB_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B")
    if bodyA_id < 0 or bodyB_id < 0:
        raise RuntimeError('Bodies "A" or "B" not found. Check the MJCF.')

    # Init state
    set_initial_state(model, data, Q1_DEG, Q2_DEG, Q1D, Q2D)

    # Pre-allocate
    dt = float(model.opt.timestep)
    N  = int(np.ceil(SIM_TIME / dt))
    # t, q1, q2, dq1, dq2, tip_x, tip_y, tip_z, tip_x_rel, tip_z_rel, elbow_x, elbow_z, step_idx
    traj = np.zeros((N, 13), dtype=float)

    # Simulate
    for i in range(N):
        t = i * dt

        # World positions
        tip = data.geom_xpos[tip_gid]          # (x,y,z) world
        elbow = data.xipos[bodyB_id]           # origin of body B (joint-2 location)

        # Fill row
        traj[i, 0]  = t
        traj[i, 1]  = data.qpos[0]
        traj[i, 2]  = data.qpos[1]
        traj[i, 3]  = data.qvel[0]
        traj[i, 4]  = data.qvel[1]
        traj[i, 5]  = tip[0]
        traj[i, 6]  = tip[1]
        traj[i, 7]  = tip[2]
        traj[i, 8]  = tip[0] - BASE_X          # tip_x relative to base
        traj[i, 9]  = tip[2] - BASE_Z          # tip_z relative to base
        traj[i,10]  = elbow[0]
        traj[i,11]  = elbow[2]
        traj[i,12]  = i

        mujoco.mj_step(model, data)

    # Save CSV
    header = (
        "t,q1,q2,dq1,dq2,"
        "tip_x,tip_y,tip_z,"
        "tip_x_rel,tip_z_rel,"
        "elbow_x,elbow_z,"
        "step_idx"
    )
    np.savetxt(OUT_CSV, traj, delimiter=",", header=header, comments="", fmt="%.10f")

    print(f"[✓] Saved {N} rows to: {OUT_CSV}")
    print("Columns:", header)

if __name__ == "__main__":
    main()
