#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Load & run the Double Pendulum MJCF in MuJoCo viewer
# OR (if MAKE_VIDEO=True) render a high-resolution MP4 for presentations.
# - Hardcode initial joint state (q1, q2 in DEGREES -> converted to radians)
# - Prints a FK sanity check at t=0 (does link-2 use q1+q2 or q1−q2?)
# ------------------------------------------------------------------------------

import time
import math
import numpy as np
import mujoco
from mujoco.viewer import launch

import imageio.v2 as imageio  # pip install imageio[ffmpeg]

XML_PATH = "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/description/double_pendulum.xml"

# ===================== User flags / knobs =====================
MAKE_VIDEO   = False    # True: render MP4, False: open interactive viewer
VIDEO_PATH   = "double_pendulum_presentation.mp4"
VIDEO_FPS    = 60      # frame rate of output video
VIDEO_LENGTH = 20.0    # seconds of simulation to record
# =============================================================

# Offscreen render resolution (good for PPT / 16:9)
RENDER_WIDTH  = 1080
RENDER_HEIGHT = 1080

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


if MAKE_VIDEO:
    # ================== OFFSCREEN RENDER TO VIDEO ==================
    dt = model.opt.timestep
    total_steps = int(VIDEO_LENGTH / dt)

    # Choose how many sim steps per rendered frame so playback is ~real-time
    steps_per_frame = max(1, int(round(1.0 / (VIDEO_FPS * dt))))
    expected_frames = total_steps // steps_per_frame

    print(f"[VIDEO] Rendering {VIDEO_LENGTH:.2f} s "
          f"({total_steps} sim steps, ~{expected_frames} frames) "
          f"to {VIDEO_PATH} at {VIDEO_FPS} fps ...")

    # Make sure the offscreen framebuffer is big enough
    # NOTE: in Python bindings it's 'global_' (not 'global')
    model.vis.global_.offwidth  = max(model.vis.global_.offwidth,  RENDER_WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, RENDER_HEIGHT)

    # Renderer(height, width)  ← note the order!
    renderer = mujoco.Renderer(model, RENDER_HEIGHT, RENDER_WIDTH)
    
    # ---------- Custom camera so the pendulum is centered ----------
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    cam.azimuth   = 90.0    # rotate around vertical axis
    cam.elevation = -30.0   # tilt down / up
    cam.distance  = 5.0     # zoom out / in

    # Center the view on the pendulum base (adjust if needed)
    cam.lookat[:] = [0.0, 0.0, 2.0]   # same z as BASE
    # ---------------------------------------------------------------

    with imageio.get_writer(VIDEO_PATH, mode="I", fps=VIDEO_FPS) as writer:
        frame_count = 0
        for step in range(total_steps):
            mujoco.mj_step(model, data)

            # Only render every Nth step so that video is ~real-time at VIDEO_FPS
            if step % steps_per_frame == 0:
                renderer.update_scene(data, camera=cam)   # use default free camera
                pixels = renderer.render()    # (H, W, 3) uint8
                writer.append_data(pixels)
                frame_count += 1

                if frame_count % max(1, expected_frames // 10) == 0:
                    print(f"[VIDEO] {100.0 * frame_count / max(1, expected_frames):5.1f}% done")

    print(f"[VIDEO] Done! Saved to {VIDEO_PATH} "
          f"(frames: {frame_count}, approx duration: {frame_count / VIDEO_FPS:.2f} s)")

else:
    # ======================= INTERACTIVE VIEWER =======================
    with launch(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(model.opt.timestep)
