# ------------------------------------------------------------------------------
# Double Pendulum: run 30 s and save trajectory to CSV (+ metadata JSON)
# CSV Columns (unchanged):
#   t, q1, q2, dq1, dq2, tip_x, tip_y, tip_z, tip_x_rel, tip_z_rel, elbow_x, elbow_z, step_idx
# Regimes:
#   1 = IDEAL      → damping=0.0,  frictionloss=0.0
#   2 = VISCOUS    → damping=0.04, frictionloss=0.0
#   3 = STICTION   → damping=0.02, frictionloss=0.07
# ------------------------------------------------------------------------------

import os
import json
import numpy as np
import mujoco

# ======================= USER OPTIONS =======================
XML_PATH = "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/description/double_pendulum.xml"
REGIME = 3  # <<< 1=IDEAL, 2=VISCOUS, 3=STICTION

# Preset values (per-DoF)
DAMPING_VISCOUS = 0.04
DAMPING_STICTION = 0.02
FRICTIONLOSS_STICTION = 0.07

TIP_GEOM_NAME = "tip_marker"     # from your MJCF
BASE_X, BASE_Z = 0.0, 2.5        # body A pos="0 0 2.5"
L1 = 1.0                         # link-1 length (for reference)
L2 = 1.0                         # link-2 length (for reference)

# --------- Initial state (DEGREES and RAD/S) ----------
Q1_DEG = 270.0
Q2_DEG = 80.0
Q1D    = 0.0
Q2D    = 0.0
SIM_TIME = 8.0  # seconds
# ==========================================================


def regime_tag_and_values(regime: int):
    if regime == 1:
        return "ideal", 0.0, 0.0
    elif regime == 2:
        return "viscous", DAMPING_VISCOUS, 0.0
    elif regime == 3:
        return "stiction", DAMPING_STICTION, FRICTIONLOSS_STICTION
    else:
        raise ValueError("REGIME must be 1 (IDEAL), 2 (VISCOUS), or 3 (STICTION).")


def set_joint_losses(model: mujoco.MjModel, joint_names, damping: float, frictionloss: float):
    """Set damping and frictionloss for hinge/slide joints (1 DoF each)."""
    for jname in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"Joint '{jname}' not found.")
        adr = model.jnt_dofadr[jid]  # dof address
        model.dof_damping[adr] = damping
        model.dof_frictionloss[adr] = frictionloss


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

    # Apply regime
    tag, damping, frictionloss = regime_tag_and_values(REGIME)
    set_joint_losses(model, ["j1", "j2"], damping=damping, frictionloss=frictionloss)
    print(f"[INFO] Regime='{tag}' | damping={damping:.4f}, frictionloss={frictionloss:.4f}")
    print(f"[INFO] timestep={model.opt.timestep:.6f} s")

    # Cache ids we use every step
    tip_gid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TIP_GEOM_NAME)
    if tip_gid < 0:
        raise RuntimeError(f'Geom "{TIP_GEOM_NAME}" not found. Check the MJCF.')

    bodyA_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A")
    bodyB_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B")
    if bodyA_id < 0 or bodyB_id < 0:
        raise RuntimeError('Bodies "A" or "B" not found. Check the MJCF.')

    # Initial state
    set_initial_state(model, data, Q1_DEG, Q2_DEG, Q1D, Q2D)

    # Output paths (CSV + JSON metadata)
    out_dir = "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data"
    out_csv = os.path.join(out_dir, f"double_pendulum_traj_{tag}.csv")
    out_meta = os.path.join(out_dir, f"double_pendulum_traj_{tag}.json")

    # Pre-allocate
    dt = float(model.opt.timestep)
    N  = int(np.ceil(SIM_TIME / dt))
    # t, q1, q2, dq1, dq2, tip_x, tip_y, tip_z, tip_x_rel, tip_z_rel, elbow_x, elbow_z, step_idx
    traj = np.zeros((N, 13), dtype=float)

    # Simulate
    for i in range(N):
        t = i * dt

        # World positions
        tip = data.geom_xpos[tip_gid]      # (x,y,z)
        elbow = data.xipos[bodyB_id]       # origin of body B (joint-2 location)

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
    np.savetxt(out_csv, traj, delimiter=",", header=header, comments="", fmt="%.10f")
    print(f"[✓] Saved {N} rows to: {out_csv}")

    # Save metadata JSON (useful for training scripts)
    meta = {
        "regime": tag,
        "damping": float(damping),
        "frictionloss": float(frictionloss),
        "timestep": float(dt),
        "sim_time": float(SIM_TIME),
        "initial_conditions": {
            "q1_deg": float(Q1_DEG),
            "q2_deg": float(Q2_DEG),
            "dq1": float(Q1D),
            "dq2": float(Q2D),
        },
        "model": os.path.basename(XML_PATH),
        "tip_geom": TIP_GEOM_NAME,
        "columns": header.split(","),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] Saved metadata: {out_meta}")


if __name__ == "__main__":
    main()
