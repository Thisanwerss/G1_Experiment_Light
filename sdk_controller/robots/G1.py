import numpy as np
from dataclasses import dataclass

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


# PD controller gains (based on kp values from g1_lab.xml)
# Leg joint gains
LEG_KP = {
    'hip': 60.0,
    'knee': 100.0,
    'ankle': 40.0
}

# Arm joint gains  
ARM_KP = {
    'shoulder': 40.0,
    'elbow': 40.0,
    'wrist_roll': 40.0,
    'wrist_pitch': 20.0,
    'wrist_yaw': 20.0
}

# Waist joint gains
WAIST_KP = 60.0

# Hand joint gains
HAND_KP = 2.0

# Damping coefficient (general)
Kd = 3.0

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

# DDS motor index mapping (29DOF version, based on g1_joint_index_dds.md)
# Body joint mapping: MuJoCo index -> DDS index
BODY_MUJOCO_TO_DDS = {
    # Left leg
    0: 0,   # left_hip_pitch
    1: 1,   # left_hip_roll
    2: 2,   # left_hip_yaw
    3: 3,   # left_knee
    4: 4,   # left_ankle_pitch
    5: 5,   # left_ankle_roll
    # Right leg
    6: 6,   # right_hip_pitch
    7: 7,   # right_hip_roll
    8: 8,   # right_hip_yaw
    9: 9,   # right_knee
    10: 10, # right_ankle_pitch
    11: 11, # right_ankle_roll
    # Waist
    12: 12, # waist_yaw
    # Left arm
    13: 15, # left_shoulder_pitch
    14: 16, # left_shoulder_roll
    15: 17, # left_shoulder_yaw
    16: 18, # left_elbow
    17: 19, # left_wrist_roll
    18: 20, # left_wrist_pitch
    19: 21, # left_wrist_yaw
    # Right arm
    20: 22, # right_shoulder_pitch
    21: 23, # right_shoulder_roll
    22: 24, # right_shoulder_yaw
    23: 25, # right_elbow
    24: 26, # right_wrist_roll
    25: 27, # right_wrist_pitch
    26: 28, # right_wrist_yaw
}

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