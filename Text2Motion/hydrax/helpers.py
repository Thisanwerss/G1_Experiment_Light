"""
    Helper functions for the SMPL to G1 humanoid mapping.
"""

import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def smpl2utg1(pose) -> dict:
    """
    Convert SMPL pose to Unitree G1 pose.
    Parameters
    ----------
    pose : np.ndarray
        SMPL pose vector.

    Returns:
    -------
    dict
        Unitree G1 pose dictionary including base 6D pose and joint angles.
    """
    utg1_pose = {}
    # Map SMPL columns to Unitree G1 joints
    joint_mapping = {
        "left_hip_pitch_joint": 3,
        "left_hip_roll_joint": 4,
        "left_hip_yaw_joint": 5,
        "right_hip_pitch_joint": 6,
        "right_hip_roll_joint": 7,
        "right_hip_yaw_joint": 8,
        "waist_pitch_joint": 9,
        "waist_roll_joint": 10,
        "waist_yaw_joint": 11,
        "left_knee_joint": 12,
        "right_knee_joint": 15,
        "left_ankle_pitch_joint": 21,
        "left_ankle_roll_joint": 22,
        "right_ankle_pitch_joint": 24,
        "right_ankle_roll_joint": 25,
        "left_shoulder_pitch_joint": 48,
        "left_shoulder_roll_joint": 50,
        "left_shoulder_yaw_joint": 49,
        "right_shoulder_pitch_joint": 51,
        "right_shoulder_roll_joint": 53,
        "right_shoulder_yaw_joint": 52,
        "left_elbow_joint": 55,
        "right_elbow_joint": 58,
        "left_wrist_pitch_joint": 61,
        "left_wrist_roll_joint": 60,
        "left_wrist_yaw_joint": 62,
        "right_wrist_pitch_joint": 64,
        "right_wrist_roll_joint": 63,
        "right_wrist_yaw_joint": 65,
    }

    # Add base pose mapping
    base_mapping = {
        "base_pos_x": 0,
        "base_pos_y": 1,
        "base_pos_z": 2,
        "base_rot_x": 3,
        "base_rot_y": 4,
        "base_rot_z": 5,
    }

    offsets = {
        "left_shoulder_roll_joint": np.pi / 2,  # 90 degree
        "right_shoulder_roll_joint": -np.pi / 2,
        "left_elbow_joint": np.pi / 2,
        "right_elbow_joint": np.pi / 2,
        "waist_pitch_joint": 0.05,
        # 'left_hip_pitch_joint': -0.204,
        # 'right_hip_pitch_joint': -0.204,
        # 'left_knee_joint': 0.24,
        # 'right_knee_joint': 0.24
    }
    reverse = {
        "right_elbow_joint": True,
        "left_wrist_yaw_joint": True,
        "right_wrist_yaw_joint": True,
    }

    # Store joint angles
    for joint, column in joint_mapping.items():
        if joint in reverse:
            utg1_pose[joint] = -pose[:, column]
        else:
            utg1_pose[joint] = pose[:, column]
            # Apply offset if specified for this joint
        if joint in offsets:
            utg1_pose[joint] += offsets[joint]

    return utg1_pose

def rpy_to_quaternion_wxyz(roll, pitch, yaw):
    """
    Convert RPY (Roll, Pitch, Yaw) angles to a wxyz quaternion.
    
    Args:
        roll (float): Rotation around x-axis (radians)
        pitch (float): Rotation around y-axis (radians)
        yaw (float): Rotation around z-axis (radians)
    
    Returns:
        np.ndarray: Quaternion in [w, x, y, z] order
    """
    # Create a rotation object from RPY angles (in 'xyz' extrinsic order)
    rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    
    # Convert to quaternion (scipy returns xyzw by default)
    quat_xyzw = rotation.as_quat()  # [x, y, z, w]
    
    # Reorder to wxyz
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    return quat_wxyz

def interpolate_poses(qpos_start, qpos_end, num_steps):
    """
    Interpolate between two robot poses.
    
    Args:
        qpos_start: Starting pose as a numpy array [base_pos (3), base_quat (4), joint_angles (N)]
        qpos_end: Ending pose as a numpy array [base_pos (3), base_quat (4), joint_angles (N)]
        num_steps: Number of interpolation steps (including start and end)
        
    Returns:
        List of interpolated poses
    """
    # Split the poses into components
    start_pos = qpos_start[:3]
    start_quat = qpos_start[3:7]
    start_joints = qpos_start[7:]
    
    end_pos = qpos_end[:3]
    end_quat = qpos_end[3:7]
    end_joints = qpos_end[7:]
    
    # Verify the joint dimensions match
    assert len(start_joints) == len(end_joints), "Joint dimensions don't match"
    
    # Create interpolation steps (0 to 1)
    steps = np.linspace(0, 1, num_steps)
    
    # Prepare quaternion interpolation
    rots = R.from_quat([start_quat, end_quat])
    slerp = Slerp([0, 1], rots)
    
    interpolated_poses = []
    
    for t in steps:
        # Linear interpolation for position
        interp_pos = (1 - t) * start_pos + t * end_pos
        
        # Spherical interpolation for orientation
        interp_rot = slerp(t)
        interp_quat = interp_rot.as_quat()
        
        # Linear interpolation for joint angles
        interp_joints = (1 - t) * start_joints + t * end_joints
        
        # Combine all components
        interp_pose = np.concatenate([interp_pos, interp_quat, interp_joints])
        interpolated_poses.append(interp_pose)
    
    return np.array(interpolated_poses)

def interpolate_foot_position(start_pos, end_pos, num_steps):
        """Interpolate between two 3D foot positions.
        
        Args:
            start_pos: Starting foot position in world frame (3D vector)
            end_pos: Ending foot position in world frame (3D vector)
            num_steps: Number of interpolation steps
            
        Returns:
            Array of interpolated positions with shape (num_steps, 3)
        """
        # Create linear interpolation between start and end positions
        t = np.linspace(0, 1, num_steps)
        t = t.reshape(-1, 1)  # Reshape for broadcasting
        
        # Linear interpolation
        interpolated_pos = start_pos + t * (end_pos - start_pos)
        
        return interpolated_pos

# Example usage:
if __name__ == "__main__":
    # Example poses (3D position, quaternion orientation, 7 joint angles)
    qpos_quat = rpy_to_quaternion_wxyz(0, 0, 0)
    qpos_start = np.array([0, 0, 0.79, # base position (x, y, z)
                           qpos_quat[0], qpos_quat[1], qpos_quat[2], qpos_quat[3], # base orientation quaternion (r, p, y)
                           0, 0, 0, # joints angles
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0])
    
    qpos_quat = rpy_to_quaternion_wxyz(0.0064155768813237, 0.7911022467933967, 0.07008789734100017)
    qpos_end = np.array([0.09133002459676551, -0.0381701398317554, 0.699624772700574, # base position (x, y, z)
                         qpos_quat[0], qpos_quat[1], qpos_quat[2], qpos_quat[3], # base orientation quaternion (r, p, y)
                         -1.2961206613982212, 0.12015214838699605, -0.0867837966362937, # joint angles
                         0.6294453918109667, -0.12442850261905132, 0.0, 
                         -1.4052881111303315, -0.09350088626236112, 0.08316131578981155, 
                         0.7881776743113834, -0.17399333611157097, 0.0, 
                         0.009957439342661061, 0.011450427199170563, 0.5199936115716901, 
                         -1.0376856192659636, 0.3539455114453784, 0.021707573085640825, 
                         1.265661541536312, 0.005543439956227413, 0.04997754056412427, 
                         0.03545109091158724, -0.7508511862156606, -0.17700076415284222, 
                         -0.12011830943288758, 0.6395467566231723, -0.013313142160636397, 
                         0.10260241849060103, -0.05857148721725906])
    
    # Get 10 interpolated poses (including start and end)
    poses = interpolate_poses(qpos_start, qpos_end, 10)
    
    # Print the first and last poses to verify
    print("First interpolated pose:")
    print(poses[0])
    print("\nLast interpolated pose:")
    print(poses[-1])