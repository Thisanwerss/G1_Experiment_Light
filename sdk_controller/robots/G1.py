import numpy as np
from dataclasses import dataclass
import json
import os

def load_global_config():
    """Load global configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "global_config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

# Load global configuration
_global_config = load_global_config()

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
# Use hg series DDS
# Only use PD target control, no feedforward torque control
# G1 29DOF Version
# Joint order: left leg(6) + right leg(6) + waist(1) + left arm(7) + right arm(7) = 27 body joints
# Note: Finger joints are not included here, finger joint control (unitree_hg::msg::dds_::HandCmd_.motor_cmd) is replaced with current position (unitree_hg::msg::dds_::HandState_.motor_state) with high damping in actual control



# STAND_UP_JOINT_POS = np.array([
#     # Left Leg
#     0.0968964, -0.12121, 0.0411549, 0.0833635, -0.160106, 0.0807926,
#     #0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#     # Right Leg
#     0.0054431, -0.0883059, 0.122414, 0.369962, -0.381041, 0.0785379,
#     #0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#     # Waist
#     #-0.0640063,
#     0.0,
#     # Left Arm
#     -0.00810865, -0.209113, 0.165738, 0.0168338, 0.198281, -0.195008, 0.0899386,
#     # Right Arm
#     0.0514546, 0.677822, 1.7251, -1.44205, -1.90577, -1.54437, -1.92981,
#     # Left Hand (these are not controlled but included for completeness)
#     -0.0127106, 0.0960897, -0.00151768, 0.236564, -0.139678, -0.51143, -0.0517731,
#     # Right Hand (these are not controlled but included for completeness)
#     -0.0280968, -1.00421, -1.69945, 1.52882, 1.8316, 1.55194, 1.90963
# ], dtype=float)
STAND_UP_JOINT_POS = np.array([

            0, 0, 0,     # waist joints
            0, 0, 0, 0, 0, 0,     # left arm
            0, 0, 0, 0, 0, 0,     # right arm
            0, 0, 0, 0.0, 0, 0,  # left leg (hip, knee, ankle)
            0, 0, 0, 0.0, 0.0, 0,  # right leg
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # fingers
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # fingers
        ])  # 确保长度匹配


# PD controller gains (loaded from global configuration)
joint_config = _global_config["g1_joint_config"]

# Extract gains for different joint types for backward compatibility
LEG_KP = {}
ARM_KP = {}
WAIST_KP = 0.0

# Build legacy gain dictionaries
for joint_name, config in joint_config.items():
    kp = config["kp"]
    
    if "hip" in joint_name:
        LEG_KP["hip"] = kp
    elif "knee" in joint_name:
        LEG_KP["knee"] = kp
    elif "ankle" in joint_name:
        LEG_KP["ankle"] = kp
    elif "waist" in joint_name:
        WAIST_KP = kp
    elif "shoulder" in joint_name:
        ARM_KP["shoulder"] = kp
    elif "elbow" in joint_name:
        ARM_KP["elbow"] = kp
    elif "wrist_roll" in joint_name:
        ARM_KP["wrist_roll"] = kp
    elif "wrist_pitch" in joint_name:
        ARM_KP["wrist_pitch"] = kp
    elif "wrist_yaw" in joint_name:
        ARM_KP["wrist_yaw"] = kp

# Hand joint gains (assuming from configuration, default to 2.0 if not found)
HAND_KP = 2.0

# Damping coefficient (from configuration)
Kd = joint_config[list(joint_config.keys())[0]]["kd"]  # Use first joint's kd value

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
# Body joint mapping: MuJoCo index -> DDS index
BODY_MUJOCO_TO_DDS = {}
for joint_name, config in joint_config.items():
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
    LEG_KP = LEG_KP
    ARM_KP = ARM_KP
    WAIST_KP = WAIST_KP
    HAND_KP = HAND_KP
    Kd = Kd
    MUJOCO_JOINT_NAMES = MUJOCO_JOINT_NAMES
    BODY_MUJOCO_TO_DDS = BODY_MUJOCO_TO_DDS
    LEFT_HAND_MUJOCO_TO_DDS = LEFT_HAND_MUJOCO_TO_DDS
    RIGHT_HAND_MUJOCO_TO_DDS = RIGHT_HAND_MUJOCO_TO_DDS
    NUM_BODY_JOINTS = NUM_BODY_JOINTS
    NUM_ACTIVE_BODY_JOINTS = NUM_ACTIVE_BODY_JOINTS
    NUM_HAND_JOINTS = NUM_HAND_JOINTS
    NUM_TOTAL_JOINTS = NUM_TOTAL_JOINTS 