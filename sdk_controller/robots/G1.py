import numpy as np
from dataclasses import dataclass
import json
import os

# --- G1 Joint Configuration ---
# This is the hardcoded 'single source of truth' for joint mappings and properties.
# It was moved from global_config.json to make the core robot definition more stable.
G1_JOINT_CONFIG = {
    "left_hip_pitch_joint": {"mujoco_index": 0, "dds_index": 0, "max_torque": 88.0},
    "left_hip_roll_joint": {"mujoco_index": 1, "dds_index": 1, "max_torque": 88.0},
    "left_hip_yaw_joint": {"mujoco_index": 2, "dds_index": 2, "max_torque": 88.0},
    "left_knee_joint": {"mujoco_index": 3, "dds_index": 3, "max_torque": 139.0},
    "left_ankle_pitch_joint": {"mujoco_index": 4, "dds_index": 4, "max_torque": 50.0},
    "left_ankle_roll_joint": {"mujoco_index": 5, "dds_index": 5, "max_torque": 50.0},
    "right_hip_pitch_joint": {"mujoco_index": 6, "dds_index": 6, "max_torque": 88.0},
    "right_hip_roll_joint": {"mujoco_index": 7, "dds_index": 7, "max_torque": 88.0},
    "right_hip_yaw_joint": {"mujoco_index": 8, "dds_index": 8, "max_torque": 88.0},
    "right_knee_joint": {"mujoco_index": 9, "dds_index": 9, "max_torque": 139.0},
    "right_ankle_pitch_joint": {"mujoco_index": 10, "dds_index": 10, "max_torque": 50.0},
    "right_ankle_roll_joint": {"mujoco_index": 11, "dds_index": 11, "max_torque": 50.0},
    "waist_yaw_joint": {"mujoco_index": 12, "dds_index": 12, "max_torque": 88.0},
    "left_shoulder_pitch_joint": {"mujoco_index": 13, "dds_index": 15, "max_torque": 25.0},
    "left_shoulder_roll_joint": {"mujoco_index": 14, "dds_index": 16, "max_torque": 25.0},
    "left_shoulder_yaw_joint": {"mujoco_index": 15, "dds_index": 17, "max_torque": 25.0},
    "left_elbow_joint": {"mujoco_index": 16, "dds_index": 18, "max_torque": 25.0},
    "left_wrist_roll_joint": {"mujoco_index": 17, "dds_index": 19, "max_torque": 25.0},
    "left_wrist_pitch_joint": {"mujoco_index": 18, "dds_index": 20, "max_torque": 5.0},
    "left_wrist_yaw_joint": {"mujoco_index": 19, "dds_index": 21, "max_torque": 5.0},
    "right_shoulder_pitch_joint": {"mujoco_index": 20, "dds_index": 22, "max_torque": 25.0},
    "right_shoulder_roll_joint": {"mujoco_index": 21, "dds_index": 23, "max_torque": 25.0},
    "right_shoulder_yaw_joint": {"mujoco_index": 22, "dds_index": 24, "max_torque": 25.0},
    "right_elbow_joint": {"mujoco_index": 23, "dds_index": 25, "max_torque": 25.0},
    "right_wrist_roll_joint": {"mujoco_index": 24, "dds_index": 26, "max_torque": 25.0},
    "right_wrist_pitch_joint": {"mujoco_index": 25, "dds_index": 27, "max_torque": 25.0},
    "right_wrist_yaw_joint": {"mujoco_index": 26, "dds_index": 28, "max_torque": 5.0}
}

# --- PD Gain Profiles ---
# These profiles are hardcoded for stability and represent fundamental robot characteristics.
# The selection and scaling of these profiles are controlled by 'global_config.json'.

DEFAULT_GAINS = {
    "left_hip_pitch_joint": {"kp": 60.0, "kd": 3.0},
    "left_hip_roll_joint": {"kp": 60.0, "kd": 3.0},
    "left_hip_yaw_joint": {"kp": 60.0, "kd": 3.0},
    "left_knee_joint": {"kp": 100.0, "kd": 3.0},
    "left_ankle_pitch_joint": {"kp": 40.0, "kd": 3.0},
    "left_ankle_roll_joint": {"kp": 40.0, "kd": 3.0},
    "right_hip_pitch_joint": {"kp": 60.0, "kd": 3.0},
    "right_hip_roll_joint": {"kp": 60.0, "kd": 3.0},
    "right_hip_yaw_joint": {"kp": 60.0, "kd": 3.0},
    "right_knee_joint": {"kp": 100.0, "kd": 3.0},
    "right_ankle_pitch_joint": {"kp": 40.0, "kd": 3.0},
    "right_ankle_roll_joint": {"kp": 40.0, "kd": 3.0},
    "waist_yaw_joint": {"kp": 60.0, "kd": 3.0},
    "left_shoulder_pitch_joint": {"kp": 40.0, "kd": 3.0},
    "left_shoulder_roll_joint": {"kp": 40.0, "kd": 3.0},
    "left_shoulder_yaw_joint": {"kp": 40.0, "kd": 3.0},
    "left_elbow_joint": {"kp": 40.0, "kd": 3.0},
    "left_wrist_roll_joint": {"kp": 40.0, "kd": 3.0},
    "left_wrist_pitch_joint": {"kp": 20.0, "kd": 3.0},
    "left_wrist_yaw_joint": {"kp": 20.0, "kd": 3.0},
    "right_shoulder_pitch_joint": {"kp": 40.0, "kd": 3.0},
    "right_shoulder_roll_joint": {"kp": 40.0, "kd": 3.0},
    "right_shoulder_yaw_joint": {"kp": 40.0, "kd": 3.0},
    "right_elbow_joint": {"kp": 40.0, "kd": 3.0},
    "right_wrist_roll_joint": {"kp": 40.0, "kd": 3.0},
    "right_wrist_pitch_joint": {"kp": 40.0, "kd": 3.0},
    "right_wrist_yaw_joint": {"kp": 20.0, "kd": 3.0}
}

SAFE_GAINS = {
    # Policy Joints (Arms)
    "left_shoulder_pitch_joint": {"kp": 14.2506, "kd": 0.9072},
    "left_shoulder_roll_joint": {"kp": 14.2506, "kd": 0.9072},
    "left_shoulder_yaw_joint": {"kp": 14.2506, "kd": 0.9072},
    "left_elbow_joint": {"kp": 14.2506, "kd": 0.9072},
    "left_wrist_roll_joint": {"kp": 14.2506, "kd": 0.9072},
    "left_wrist_pitch_joint": {"kp": 16.7783, "kd": 1.0681},
    "left_wrist_yaw_joint": {"kp": 16.7783, "kd": 1.0681},
    "right_shoulder_pitch_joint": {"kp": 14.2506, "kd": 0.9072},
    "right_shoulder_roll_joint": {"kp": 14.2506, "kd": 0.9072},
    "right_shoulder_yaw_joint": {"kp": 14.2506, "kd": 0.9072},
    "right_elbow_joint": {"kp": 14.2506, "kd": 0.9072},
    "right_wrist_roll_joint": {"kp": 14.2506, "kd": 0.9072},
    "right_wrist_pitch_joint": {"kp": 16.7783, "kd": 1.0681},
    "right_wrist_yaw_joint": {"kp": 16.7783, "kd": 1.0681},
    # Non-Policy Joints (Legs and Waist)
    "left_hip_pitch_joint": {"kp": 40.1792, "kd": 1.2789},
    "left_hip_roll_joint": {"kp": 99.0984, "kd": 3.1544},
    "left_hip_yaw_joint": {"kp": 40.1792, "kd": 1.2789},
    "left_knee_joint": {"kp": 99.0984, "kd": 3.1544},
    "left_ankle_pitch_joint": {"kp": 28.5012, "kd": 0.9072},
    "left_ankle_roll_joint": {"kp": 28.5012, "kd": 0.9072},
    "right_hip_pitch_joint": {"kp": 40.1792, "kd": 1.2789},
    "right_hip_roll_joint": {"kp": 99.0984, "kd": 3.1544},
    "right_hip_yaw_joint": {"kp": 40.1792, "kd": 1.2789},
    "right_knee_joint": {"kp": 99.0984, "kd": 3.1544},
    "right_ankle_pitch_joint": {"kp": 28.5012, "kd": 0.9072},
    "right_ankle_roll_joint": {"kp": 28.5012, "kd": 0.9072},
    "waist_yaw_joint": {"kp": 40.1792, "kd": 1.2789},
    # Note: waist_roll_joint and waist_pitch_joint from the original tensor are not
    # in the controllable joint list, so they are omitted here.
}

PROFILES = {"default": DEFAULT_GAINS, "safe": SAFE_GAINS}

def load_global_config():
    """Load global configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "global_config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

# Load global configuration
_global_config = load_global_config()

# --- PD Gains Configuration ---
pd_config = _global_config.get("pd_gain_config", {})
active_profile_name = pd_config.get("active_profile", "default")
kp_scale = np.clip(pd_config.get("kp_scale", 1.0), 0.0, 1.0)
kd_scale = np.clip(pd_config.get("kd_scale", 1.0), 0.0, 1.0)

# Select the base profile, defaulting to "default" if the name is invalid
base_gains = PROFILES.get(active_profile_name, PROFILES["default"])
if active_profile_name not in PROFILES:
    print(f"Warning: '{active_profile_name}' is not a valid profile. Falling back to 'default'.")

# Apply scaling to the selected profile to get final gains
JOINT_KP = {name: gain["kp"] * kp_scale for name, gain in base_gains.items()}
JOINT_KD = {name: gain["kd"] * kd_scale for name, gain in base_gains.items()}

# G1 Robot Configuration
ROBOT_NAME = "g1"
OBJECT_NAME = "G1"### 29DOF Version

# IMU position and rotation in Vicon (adjust according to actual setup)
P_IMU_IN_VICON = np.array([0.0, 0.0, 0.0])
R_IMU_IN_VICON = np.eye(3)

# IMU position in robot base (from g1_lab.xml)
P_IMU_IN_BASE = np.array([0.04525, 0, -0.08339])
R_IMU_IN_BASE = np.eye(3)

# Control frequency
CONTROL_FREQ = 100  # 100Hz control frequency
# Using hg series DDS
# Only PD target control is used, no feedforward torque control
# G1 29DOF Version
# Joint order: left leg(6) + right leg(6) + waist(1) + left arm(7) + right arm(7) = 27 body joints
# Note: Finger joints are not included here. Finger joint control (unitree_hg::msg::dds_::HandCmd_.motor_cmd) 
# is replaced with damping around the current position (from unitree_hg::msg::dds_::HandState_.motor_state) in actual control.



STAND_UP_JOINT_POS = np.array([

            0, 0, 0,     # waist joints
            0, 0, 0, 0, 0, 0,     # left arm
            0, 0, 0, 0, 0, 0,     # right arm
            0, 0, 0, 0.0, 0, 0,  # left leg (hip, knee, ankle)
            0, 0, 0, 0.0, 0.0, 0,  # right leg
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # fingers
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # fingers
        ])  


# Hand joint gains (assuming from configuration, default to 2.0 if not found)
HAND_KP = 2.0

# Joint index mapping (for MuJoCo to DDS mapping)
# MuJoCo joint order (from g1_lab.xml)
MUJOCO_JOINT_NAMES = [
    # Legs (0-11)
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    # Waist (12)
    'waist_yaw_joint',
    # Left arm (13-19)
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    # Right arm (20-26)
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
    # Left hand fingers (27-33)
    'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
    'left_hand_middle_0_joint', 'left_hand_middle_1_joint',
    'left_hand_index_0_joint', 'left_hand_index_1_joint',
    # Right hand fingers (34-40)
    'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
    'right_hand_middle_0_joint', 'right_hand_middle_1_joint',
    'right_hand_index_0_joint', 'right_hand_index_1_joint',
]

# DDS motor index mapping (loaded from global configuration)
# mapping from MuJoCo index to DDS index
BODY_MUJOCO_TO_DDS = {}
for joint_name, config in G1_JOINT_CONFIG.items():
    BODY_MUJOCO_TO_DDS[config["mujoco_index"]] = config["dds_index"]

# Hand joint mapping: MuJoCo index -> Hand DDS index, not actually used for control
# Left hand
LEFT_HAND_MUJOCO_TO_DDS = {
    27: 0,  # thumb_0
    28: 1,  # thumb_1
    29: 2,  # thumb_2
    32: 3,  # index_0
    33: 4,  # index_1
    30: 5,  # middle_0
    31: 6,  # middle_1
}

# Right hand
RIGHT_HAND_MUJOCO_TO_DDS = {
    34: 0,  # thumb_0
    35: 1,  # thumb_1
    36: 2,  # thumb_2
    37: 3,  # index_0
    38: 4,  # index_1
    39: 5,  # middle_0
    40: 6,  # middle_1
}

# Number of body joints
NUM_BODY_JOINTS = 29  # Including unused waist_roll and waist_pitch
NUM_ACTIVE_BODY_JOINTS = 27  # Actually used body joints

# Number of hand joints
NUM_HAND_JOINTS = 7  # Per hand

# Total number of joints (for MuJoCo)
NUM_TOTAL_JOINTS = 41  # 27 body joints + 14 finger joints 

@dataclass
class G1:
    ROBOT_NAME = ROBOT_NAME
    OBJECT_NAME = OBJECT_NAME
    P_IMU_IN_VICON = P_IMU_IN_VICON
    R_IMU_IN_VICON = R_IMU_IN_VICON
    P_IMU_IN_BASE = P_IMU_IN_BASE
    R_IMU_IN_BASE = R_IMU_IN_BASE
    CONTROL_FREQ = CONTROL_FREQ
    STAND_UP_JOINT_POS = STAND_UP_JOINT_POS
    JOINT_KP = JOINT_KP
    JOINT_KD = JOINT_KD
    HAND_KP = HAND_KP
    MUJOCO_JOINT_NAMES = MUJOCO_JOINT_NAMES
    BODY_MUJOCO_TO_DDS = BODY_MUJOCO_TO_DDS
    LEFT_HAND_MUJOCO_TO_DDS = LEFT_HAND_MUJOCO_TO_DDS
    RIGHT_HAND_MUJOCO_TO_DDS = RIGHT_HAND_MUJOCO_TO_DDS
    NUM_BODY_JOINTS = NUM_BODY_JOINTS
    NUM_ACTIVE_BODY_JOINTS = NUM_ACTIVE_BODY_JOINTS
    NUM_HAND_JOINTS = NUM_HAND_JOINTS
    NUM_TOTAL_JOINTS = NUM_TOTAL_JOINTS 