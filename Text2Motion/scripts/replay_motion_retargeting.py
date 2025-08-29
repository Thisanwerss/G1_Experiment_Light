import sys
import time
import joblib
import mujoco
import numpy as np
from hydrax import ROOT
from mujoco import MjModel, MjData
from mujoco.viewer import launch_passive
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree

# === CONFIGURATION ===
MOTION_FILE = "/home/ilyass/workspace/motion_retargeting/motion.pkl"
MODEL_XML = ROOT + "/models/g1/scene.xml"
FPS = 30
FOOT_SITES = ["left_foot", "right_foot"]
GROUND_BUFFER = 0.035  # Keep foot 5mm above ground

# === LOAD MODEL ===
model = MjModel.from_xml_path(MODEL_XML)
data = MjData(model)

# === LOAD MOTION ===
motion_data = joblib.load(MOTION_FILE)["retargeted_motion"]
print("Original motion shape:", motion_data.shape)

if motion_data.shape[1] != model.nq:
    raise ValueError(f"Mismatch: motion has {motion_data.shape[1]} DOFs but model expects {model.nq}")

# === FUNCTION: Get min foot height for a given qpos ===
def get_min_foot_height(qpos):
    data.qpos[:] = qpos
    mujoco.mj_fwdPosition(model, data)
    heights = []
    for name in FOOT_SITES:
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            heights.append(data.site_xpos[site_id][2])
        except Exception as e:
            print(f"Warning: site '{name}' not found: {e}")
    return min(heights) if heights else 0.0

# === STEP: Adjust root z per-frame to avoid floating ===
adjusted_motion = motion_data.copy()

for i, qpos in enumerate(motion_data):
    min_foot_z = get_min_foot_height(qpos)
    if min_foot_z > GROUND_BUFFER:
        z_offset = min_foot_z - GROUND_BUFFER
        adjusted_motion[i, 2] -= z_offset

print("Applied per-frame foot grounding correction.")

# === VISUALIZATION ===
with launch_passive(model, data) as viewer:
    for qpos in adjusted_motion:
        print("Pelvis z:", qpos[2])
        data.qpos[:] = qpos
        data.qvel[:] = 0
        mujoco.mj_fwdPosition(model, data)
        viewer.sync()
        time.sleep(1.0 / FPS)

    viewer.close()
    sys.exit(0)
