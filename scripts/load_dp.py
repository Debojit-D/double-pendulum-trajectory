# ------------------------------------------------------------------------------
# Load & run the Double Pendulum MJCF in MuJoCo viewer
# - Resets to keyframe "rest" if present, else zeros
# ------------------------------------------------------------------------------

import time
import mujoco
from mujoco.viewer import launch

XML_PATH = "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/description/double_pendulum.xml"
KEYFRAME_NAME = "rest"

# Load model/data
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Reset to keyframe "rest" (use integer id)
try:
    k_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
    # If name not found, mj_name2id returns -1
    if k_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, k_id)
    else:
        raise KeyError
except KeyError:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

# Launch viewer (context manager version is more reliable)
with launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)   # remove this line if you only want to display initial pose
        viewer.render()
        time.sleep(model.opt.timestep)
