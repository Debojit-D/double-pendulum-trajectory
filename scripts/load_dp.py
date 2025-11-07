# ------------------------------------------------------------------------------
# Double Pendulum viewer with quick regime switch + explicit start config
# Regimes:
#   1 = IDEAL      → damping=0.0, frictionloss=0.0
#   2 = VISCOUS    → damping=0.04, frictionloss=0.0
#   3 = STICTION   → damping=0.02, frictionloss=0.07
# ------------------------------------------------------------------------------

import time
import math
import mujoco
from mujoco.viewer import launch

# ======================= USER OPTIONS =======================
XML_PATH = "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/description/double_pendulum.xml"
KEYFRAME_NAME = "rest"
REGIME = 3  # <<< 1=IDEAL, 2=VISCOUS, 3=STICTION

# Preset values (per-DoF) — tune if desired
DAMPING_VISCOUS = 0.04
DAMPING_STICTION = 0.02
FRICTIONLOSS_STICTION = 0.07

# ---------- Start configuration ----------
# Options: "keyframe" | "zeros" | "angles"
START_MODE = "angles"

# If START_MODE == "angles", set desired initial angles & velocities.
# Angles below are in RADIANS by default; set USE_DEGREES=True to pass degrees instead.
USE_DEGREES = False
START_QPOS = (math.pi, 0.0)   # (theta1, theta2)
START_QVEL = (0.0, 0.0)         # (omega1, omega2)
# ============================================================


def set_joint_losses(model: mujoco.MjModel, joint_names, damping: float, frictionloss: float):
    """Set damping and frictionloss for the given hinge/slide joints."""
    for jname in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise ValueError(f"Joint '{jname}' not found in model.")
        adr = model.jnt_dofadr[jid]  # dof address (1 DoF for hinge/slide)
        model.dof_damping[adr] = damping
        model.dof_frictionloss[adr] = frictionloss


def apply_regime(model: mujoco.MjModel, regime: int):
    """Apply loss settings according to the selected regime."""
    if regime == 1:  # IDEAL
        damping, fl = 0.0, 0.0
        tag = "IDEAL"
    elif regime == 2:  # VISCOUS
        damping, fl = DAMPING_VISCOUS, 0.0
        tag = "VISCOUS"
    elif regime == 3:  # STICTION
        damping, fl = DAMPING_STICTION, FRICTIONLOSS_STICTION
        tag = "STICTION"
    else:
        raise ValueError("REGIME must be 1 (IDEAL), 2 (VISCOUS), or 3 (STICTION).")
    set_joint_losses(model, ["j1", "j2"], damping=damping, frictionloss=fl)
    print(f"[INFO] Applied regime={tag} | damping={damping:.4f}, frictionloss={fl:.4f}")


def reset_state(model: mujoco.MjModel, data: mujoco.MjData, mode: str, key_name: str):
    """Reset to keyframe/zeros/explicit angles."""
    mode = mode.lower()
    if mode == "keyframe":
        k_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if k_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, k_id)
            print(f"[INFO] Reset to keyframe '{key_name}'.")
        else:
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            print(f"[WARN] Keyframe '{key_name}' not found. Reset to zeros.")
    elif mode == "zeros":
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        print("[INFO] Reset to zeros.")
    elif mode == "angles":
        qpos = list(START_QPOS)
        qvel = list(START_QVEL)
        if USE_DEGREES:
            qpos = [math.radians(a) for a in qpos]
        # Assume joint order: j1, j2 (first two qpos/qvel entries)
        data.qpos[:2] = qpos
        data.qvel[:2] = qvel
        # Zero the rest (safety if model had extra DoFs)
        if model.nq > 2:
            data.qpos[2:] = 0.0
        if model.nv > 2:
            data.qvel[2:] = 0.0
        mujoco.mj_forward(model, data)
        ang_disp = qpos if not USE_DEGREES else START_QPOS
        unit = "rad" if not USE_DEGREES else "deg"
        print(f"[INFO] Reset to angles: theta=({ang_disp[0]:.4f}, {ang_disp[1]:.4f}) {unit}, "
              f"omega=({qvel[0]:.4f}, {qvel[1]:.4f}) rad/s")
    else:
        raise ValueError("START_MODE must be 'keyframe', 'zeros', or 'angles'.")


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    apply_regime(model, REGIME)
    reset_state(model, data, START_MODE, KEYFRAME_NAME)

    with launch(model, data) as viewer:
        print(f"[INFO] timestep={model.opt.timestep:.6f} s, integrator={model.opt.integrator}")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
