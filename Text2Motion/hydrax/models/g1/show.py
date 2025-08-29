import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('./scene.xml')
data = mujoco.MjData(model)

# Set visual options to display collision geoms (group 3)
opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)
opt.geomgroup[3] = 1  # Enable collision geometry group

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt = opt  # Apply the options
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
