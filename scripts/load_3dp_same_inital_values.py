#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Three Double Pendulums in One MuJoCo Scene
#
# - Assumes XML has *three* copies of the double pendulum:
#       Pendulum A: joints  joint1_A, joint2_A   | tip geom: tip_marker_A
#       Pendulum B: joints  joint1_B, joint2_B   | tip geom: tip_marker_B
#       Pendulum C: joints  joint1_C, joint2_C   | tip geom: tip_marker_C
#
# - All three start from the SAME initial (q1, q2, q̇1, q̇2)
# - Each pendulum has DIFFERENT dynamics (e.g., damping/friction)
# - Run for a 3-second horizon and either:
#      * render to MP4 (offscreen), or
#      * open interactive viewer.
# ------------------------------------------------------------------------------

import time
import math
import numpy as np
import mujoco
from mujoco.viewer import launch
import imageio.v2 as imageio

XML_PATH = "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/description/double_pendulum_3regimes.xml"

# ===================== User flags / knobs =====================
MAKE_VIDEO   = True                 # True: render MP4, False: interactive viewer
VIDEO_PATH   = "double_pendulum_3pend.mp4"
VIDEO_FPS    = 60
VIDEO_LENGTH = 10.0                  # seconds of simulation horizon
# =============================================================

# Offscreen render resolution (square, 16:9-friendly if you crop)
RENDER_WIDTH  = 1080
RENDER_HEIGHT = 1080

# --------- Hardcoded initial state (DEGREES and RAD/S) ----------
Q1_DEG = 270.0   # link-1 down (depending on your axis and geom)
Q2_DEG = 80.0    # relative angle
Q1D    = 0.0
Q2D    = 0.0
# ---------------------------------------------------------------

PEND_SUFFIXES = ["A", "B", "C"]    # three pendulums side-by-side


def get_joint_dof_index(model, joint_name: str) -> int:
    """Return DOF index corresponding to a joint name."""
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if j_id < 0:
        raise RuntimeError(f'Joint "{joint_name}" not found.')
    return model.jnt_dofadr[j_id]


def set_initial_state_all(model, data,
                          q1_deg: float, q2_deg: float,
                          q1d: float = 0.0, q2d: float = 0.0):
    """
    Set the same (q1,q2,q1d,q2d) for all three pendulums.
    Assumes joint order: [joint1_A, joint2_A, joint1_B, joint2_B, joint1_C, joint2_C]
    or similar. You can also look up indices by name.
    """
    q1_rad, q2_rad = np.deg2rad([q1_deg, q2_deg])

    # Resolve qpos indices by joint name (safe & explicit)
    j1A = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1_A")
    j2A = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2_A")
    j1B = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1_B")
    j2B = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2_B")
    j1C = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1_C")
    j2C = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2_C")

    # qpos index for a joint is model.jnt_qposadr[j_id]
    q1A_i = model.jnt_qposadr[j1A]
    q2A_i = model.jnt_qposadr[j2A]
    q1B_i = model.jnt_qposadr[j1B]
    q2B_i = model.jnt_qposadr[j2B]
    q1C_i = model.jnt_qposadr[j1C]
    q2C_i = model.jnt_qposadr[j2C]

    # Set positions
    data.qpos[q1A_i] = q1_rad
    data.qpos[q2A_i] = q2_rad
    data.qpos[q1B_i] = q1_rad
    data.qpos[q2B_i] = q2_rad
    data.qpos[q1C_i] = q1_rad
    data.qpos[q2C_i] = q2_rad

    # DOF indices for velocities
    d1A = get_joint_dof_index(model, "joint1_A")
    d2A = get_joint_dof_index(model, "joint2_A")
    d1B = get_joint_dof_index(model, "joint1_B")
    d2B = get_joint_dof_index(model, "joint2_B")
    d1C = get_joint_dof_index(model, "joint1_C")
    d2C = get_joint_dof_index(model, "joint2_C")

    data.qvel[d1A] = q1d
    data.qvel[d2A] = q2d
    data.qvel[d1B] = q1d
    data.qvel[d2B] = q2d
    data.qvel[d1C] = q1d
    data.qvel[d2C] = q2d

    mujoco.mj_forward(model, data)


def apply_regime_params(model):
    """
    Example: set different damping / friction for each pendulum's DOFs.
    You should tune these numbers to match your 'ideal', 'viscous', 'stiction' presets.
    """

    # DOF indices
    d1A = get_joint_dof_index(model, "joint1_A")
    d2A = get_joint_dof_index(model, "joint2_A")
    d1B = get_joint_dof_index(model, "joint1_B")
    d2B = get_joint_dof_index(model, "joint2_B")
    d1C = get_joint_dof_index(model, "joint1_C")
    d2C = get_joint_dof_index(model, "joint2_C")

    # Start from some default
    # (You can also read your XML defaults before overwriting them)
    model.dof_damping[:]      = 0.0
    model.dof_frictionloss[:] = 0.0

    # Pendulum A: "ideal" (no damping / friction)
    model.dof_damping[[d1A, d2A]]      = 0.0
    model.dof_frictionloss[[d1A, d2A]] = 0.0

    # Pendulum B: "viscous" (only damping)
    model.dof_damping[[d1B, d2B]]      = 0.04   # example
    model.dof_frictionloss[[d1B, d2B]] = 0.0

    # Pendulum C: "stiction" (some damping + friction)
    model.dof_damping[[d1C, d2C]]      = 0.02   # example
    model.dof_frictionloss[[d1C, d2C]] = 0.07   # example


def main():
    # Load the 3-pendulum model
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # Apply regime parameters to each pendulum
    apply_regime_params(model)

    # Set initial state identical for all three pendulums
    set_initial_state_all(model, data, Q1_DEG, Q2_DEG, Q1D, Q2D)

    dt = model.opt.timestep
    total_steps = int(VIDEO_LENGTH / dt)

    if MAKE_VIDEO:
        # ---------- OFFSCREEN RENDER TO VIDEO ----------
        steps_per_frame = max(1, int(round(1.0 / (VIDEO_FPS * dt))))
        expected_frames = total_steps // steps_per_frame

        print(f"[VIDEO] Rendering {VIDEO_LENGTH:.2f} s "
              f"({total_steps} steps, ~{expected_frames} frames) "
              f"to {VIDEO_PATH} at {VIDEO_FPS} fps ...")

        # Ensure framebuffer is large enough
        model.vis.global_.offwidth  = max(model.vis.global_.offwidth,  RENDER_WIDTH)
        model.vis.global_.offheight = max(model.vis.global_.offheight, RENDER_HEIGHT)

        renderer = mujoco.Renderer(model, RENDER_HEIGHT, RENDER_WIDTH)

        # Camera setup: pull back so all 3 pendulums are visible
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth   = 90.0
        cam.elevation = -20.0
        cam.distance  = 8.0      # bigger distance to see all three
        cam.lookat[:] = [0.0, 0.0, 2.0]

        with imageio.get_writer(VIDEO_PATH, mode="I", fps=VIDEO_FPS) as writer:
            frame_count = 0
            for step in range(total_steps):
                mujoco.mj_step(model, data)

                if step % steps_per_frame == 0:
                    renderer.update_scene(data, camera=cam)
                    pixels = renderer.render()
                    writer.append_data(pixels)
                    frame_count += 1

                    if frame_count % max(1, expected_frames // 10) == 0:
                        print(f"[VIDEO] {100.0 * frame_count / max(1, expected_frames):5.1f}% done")

        print(f"[VIDEO] Done! Saved to {VIDEO_PATH} "
              f"(frames: {frame_count}, approx duration: {frame_count / VIDEO_FPS:.2f} s)")

    else:
        # ---------- INTERACTIVE VIEWER ----------
        with launch(model, data) as viewer:
            start = time.time()
            while viewer.is_running() and (time.time() - start) < VIDEO_LENGTH:
                mujoco.mj_step(model, data)
                viewer.render()
                time.sleep(dt)


if __name__ == "__main__":
    main()
