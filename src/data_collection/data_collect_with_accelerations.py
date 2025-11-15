#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Double Pendulum Dataset Generator
# Runs N trajectories with varied initial conditions and saves:
#  - One CSV per run
#  - One master JSON manifest listing all runs + initial conditions
#
# CSV Columns (UPDATED):
#   t, q1, q2, dq1, dq2,
#   tip_x, tip_y, tip_z,
#   tip_x_rel, tip_z_rel,
#   elbow_x, elbow_z,
#   step_idx,
#   ddq1, ddq2
# ------------------------------------------------------------------------------

import os
import json
import math
import numpy as np
import mujoco as mj  # alias avoids shadowing issues

# ======================= USER OPTIONS =======================
XML_PATH     = "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/description/double_pendulum.xml"
OUTPUT_DIR   = "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal2"

REGIME       = 1        # 1=IDEAL, 2=VISCOUS, 3=STICTION
N_SAMPLES    = 2000     # how many distinct runs to generate
SIM_TIME     = 20.0     # seconds per run

SEED         = 42       # for reproducibility (set None for nondeterministic)

# Ranges for initial conditions (degrees & rad/s). Tune as you like.
INIT_RANGES = {
    "q1_deg": (0.0, 360.0),   # broad for chaos
    "q2_deg": (0.0, 360.0),
    "dq1":    (0.0,  0.0),    # rad/s
    "dq2":    (0.0,  0.0),    # rad/s
}
# ============================================================

# Preset values (per-DoF)
DAMPING_VISCOUS       = 0.04
DAMPING_STICTION      = 0.02
FRICTIONLOSS_STICTION = 0.07

TIP_GEOM_NAME = "tip_marker"     # from your MJCF
BASE_X, BASE_Z = 0.0, 2.5        # base body A pos="0 0 2.5" (for relative coords)

# UPDATED HEADER (now includes ddq1, ddq2)
HEADER = (
    "t,q1,q2,dq1,dq2,"
    "tip_x,tip_y,tip_z,"
    "tip_x_rel,tip_z_rel,"
    "elbow_x,elbow_z,"
    "step_idx,"
    "ddq1,ddq2"
)

def regime_tag_and_values(regime: int):
    if regime == 1:
        return "ideal", 0.0, 0.0
    elif regime == 2:
        return "viscous", DAMPING_VISCOUS, 0.0
    elif regime == 3:
        return "stiction", DAMPING_STICTION, FRICTIONLOSS_STICTION
    else:
        raise ValueError("REGIME must be 1 (IDEAL), 2 (VISCOUS), or 3 (STICTION).")

def set_joint_losses(model: mj.MjModel, joint_names, damping: float, frictionloss: float):
    """Set damping and frictionloss for hinge/slide joints (1 DoF each)."""
    for jname in joint_names:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"Joint '{jname}' not found.")
        adr = model.jnt_dofadr[jid]  # dof address
        model.dof_damping[adr] = damping
        model.dof_frictionloss[adr] = frictionloss

def set_initial_state(model, data, q1_deg, q2_deg, dq1=0.0, dq2=0.0):
    """Set initial joint positions (deg) and velocities (rad/s)."""
    q = np.deg2rad([q1_deg, q2_deg])
    data.qpos[0] = float(q[0])
    data.qpos[1] = float(q[1])
    data.qvel[0] = float(dq1)
    data.qvel[1] = float(dq2)
    # Ensure qacc etc. are consistent at t=0
    mj.mj_forward(model, data)

def sample_initials(rng: np.random.Generator):
    """Draw one sample of initial conditions from INIT_RANGES."""
    def uni(a, b): return rng.uniform(a, b)
    q1_deg = uni(*INIT_RANGES["q1_deg"])
    q2_deg = uni(*INIT_RANGES["q2_deg"])
    dq1    = uni(*INIT_RANGES["dq1"])
    dq2    = uni(*INIT_RANGES["dq2"])
    return dict(q1_deg=q1_deg, q2_deg=q2_deg, dq1=dq1, dq2=dq2)

def run_one(model, data, tip_gid, bodyB_id, sim_time: float):
    """Simulate one rollout and return the trajectory array."""
    dt = float(model.opt.timestep)
    N  = int(math.ceil(sim_time / dt))

    # Columns:
    # 0: t
    # 1: q1, 2: q2
    # 3: dq1, 4: dq2
    # 5: tip_x, 6: tip_y, 7: tip_z
    # 8: tip_x_rel, 9: tip_z_rel
    # 10: elbow_x, 11: elbow_z
    # 12: step_idx
    # 13: ddq1, 14: ddq2
    traj = np.zeros((N, 15), dtype=float)

    for i in range(N):
        t = i * dt

        # positions
        tip   = data.geom_xpos[tip_gid]  # (x,y,z)
        elbow = data.xipos[bodyB_id]     # body B origin (joint-2)

        # record state *before* stepping
        traj[i, 0]  = t
        traj[i, 1]  = data.qpos[0]
        traj[i, 2]  = data.qpos[1]
        traj[i, 3]  = data.qvel[0]
        traj[i, 4]  = data.qvel[1]
        traj[i, 5]  = tip[0]
        traj[i, 6]  = tip[1]
        traj[i, 7]  = tip[2]
        traj[i, 8]  = tip[0] - BASE_X       # relative to base X
        traj[i, 9]  = tip[2] - BASE_Z       # relative to base Z
        traj[i,10]  = elbow[0]
        traj[i,11]  = elbow[2]
        traj[i,12]  = i
        traj[i,13]  = data.qacc[0]          # ddq1 from MuJoCo
        traj[i,14]  = data.qacc[1]          # ddq2 from MuJoCo

        # step
        mj.mj_step(model, data)

    return traj, dt, N

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED) if SEED is not None else np.random.default_rng()

    # Load model/data once, reuse for all runs
    model = mj.MjModel.from_xml_path(XML_PATH)
    data  = mj.MjData(model)

    # Apply regime (override any MJCF defaults)
    tag, damping, frictionloss = regime_tag_and_values(REGIME)
    set_joint_losses(model, ["j1", "j2"], damping=damping, frictionloss=frictionloss)

    # Cache ids
    tip_gid  = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, TIP_GEOM_NAME)
    if tip_gid < 0:
        raise RuntimeError(f'Geom "{TIP_GEOM_NAME}" not found. Check the MJCF.')
    bodyB_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "B")
    if bodyB_id < 0:
        raise RuntimeError('Body "B" not found. Check the MJCF.')

    print(f"[INFO] Regime='{tag}' | damping={damping:.4f}, frictionloss={frictionloss:.4f}")
    print(f"[INFO] timestep={model.opt.timestep:.6f} s")
    print(f"[INFO] Generating {N_SAMPLES} runs to: {OUTPUT_DIR}")

    # Manifest to summarize the whole dataset
    manifest = {
        "dataset_name": f"double_pendulum_{tag}",
        "xml_model": os.path.basename(XML_PATH),
        "output_dir": OUTPUT_DIR,
        "regime": {
            "tag": tag,
            "damping": float(damping),
            "frictionloss": float(frictionloss),
        },
        "timestep": float(model.opt.timestep),
        "sim_time_per_run": float(SIM_TIME),
        "seed": int(SEED) if SEED is not None else None,
        "init_ranges": INIT_RANGES,
        "columns": HEADER.split(","),   # <-- includes ddq1, ddq2 now
        "runs": [],   # will be appended
    }

    # Generate runs
    for k in range(N_SAMPLES):
        # Sample new initials
        ic = sample_initials(rng)
        # Reset to those initials
        set_initial_state(model, data, ic["q1_deg"], ic["q2_deg"], ic["dq1"], ic["dq2"])

        # Run sim
        traj, dt, N = run_one(model, data, tip_gid, bodyB_id, SIM_TIME)

        # Save CSV
        csv_name = f"double_pendulum_traj_{tag}_run{k:03d}.csv"
        out_csv  = os.path.join(OUTPUT_DIR, csv_name)
        np.savetxt(out_csv, traj, delimiter=",", header=HEADER, comments="", fmt="%.10f")

        # Append to manifest
        manifest["runs"].append({
            "run_id": k,
            "csv": csv_name,
            "initial_conditions": {
                "q1_deg": float(ic["q1_deg"]),
                "q2_deg": float(ic["q2_deg"]),
                "dq1": float(ic["dq1"]),
                "dq2": float(ic["dq2"]),
            },
            "num_steps": int(N),
            "sim_time": float(SIM_TIME),
        })

        if (k + 1) % 5 == 0 or (k + 1) == N_SAMPLES:
            print(f"[INFO] Run {k+1}/{N_SAMPLES} saved → {csv_name}")

    # Save ONE master JSON
    manifest_path = os.path.join(OUTPUT_DIR, f"double_pendulum_manifest_{tag}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[✓] Wrote manifest: {manifest_path}")
    print(f"[✓] Dataset ready: {N_SAMPLES} CSVs in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
