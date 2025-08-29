from typing import Dict, List, Optional, Tuple
import mujoco.specs_test
import numpy as np
import mujoco
import pinocchio as pin
import os
import importlib
from abc import ABC
from dataclasses import dataclass

MJ2PIN_QUAT = [1,2,3,0]
PIN2MJ_QUAT = [3,0,1,2]

@dataclass
class RobotDescription(ABC):
    # Robot name
    name : str
    # MuJoCo model path (id loaded)
    xml_path : str = ""
    # Pinocchio model path (if loaded)
    urdf_path : str = ""
    # Scene path 
    xml_scene_path : str = ""
    # End-effectors frame name
    eeff_frame_name : Optional[List[str]] = None
    # Nominal configuration
    q0 : Optional[np.ndarray] = None

def _path_scene_with_floor(path_mjcf : str) -> str:
    """
    Get file path to a mujoco model with floor.
    """
    file_dir, _ = os.path.split(path_mjcf)
    scene_file_name = "scene.xml"
    scene_path = os.path.join(file_dir, scene_file_name)

    if not os.path.exists(scene_path):
        print(f"MuJoCo scene path {scene_path} not found.")
        return ""
    
    # Process scene file
    with open(scene_path, "r") as f:
        # Remove compiler if exists in scene file
        new_file = [line for line in 
                    filter(lambda str : "<compiler" not in str, f.readlines())]
    
    with open(scene_path, "w") as f:
        f.writelines(new_file)
    
    return scene_path

def get_mj_desc(robot_name : str, desc : RobotDescription = None) -> RobotDescription:
    # Get robot model file path
    robot_mj_description_module = importlib.import_module(f"robot_descriptions.{robot_name}_mj_description")
    path_mjcf = robot_mj_description_module.MJCF_PATH

    assert os.path.exists(path_mjcf), f"MJCF file not found. Invalid robot name {robot_name}"

    # Load description if exists
    desc = RobotDescription(robot_name) if desc is None else desc
    if desc.name != robot_name:
        raise ValueError(f"Wrong robot name {robot_name}.")
    
    # Get model path
    desc.xml_path = path_mjcf
    # Get scene path if exists
    desc.xml_scene_path = _path_scene_with_floor(path_mjcf)

    # Save keyframe if exists
    mj_model = mujoco.MjModel.from_xml_path(path_mjcf)
    if mj_model.nkey > 0:
        desc.q0 = mj_model.key_qpos[0].copy()

    return desc

def get_pin_desc(robot_name : str, desc : RobotDescription = None) -> RobotDescription:
    # Get robot model file path
    robot_description_module = importlib.import_module(f"robot_descriptions.{robot_name}_description")
    urdf_path = robot_description_module.URDF_PATH

    assert os.path.exists(urdf_path), f"URDF file not found. Invalid robot name {robot_name}"

    # Load description if exists
    desc = RobotDescription(robot_name) if desc is None else desc
    if desc.name != robot_name:
        raise ValueError(f"Wrong robot name {robot_name}.")
    
    desc.urdf_path = urdf_path
    return desc

def get_robot_description(robot_name : str) -> RobotDescription:
    """
    Retrieve the robot description for a given robot name.
    Base on robot description package.

    Args:
        robot_name (str): The name of the robot for which the description is to be retrieved.

    Returns:
        RobotDescription: The complete description of the robot.
    """
    # Get robot model file path
    robot_desc = get_mj_desc(robot_name)
    robot_desc = get_pin_desc(robot_name, robot_desc)

    return robot_desc

def mj_body_pos(mj_model, mj_data, frame_name: str) -> np.ndarray:
    """
    Get the body position for a given frame name in MuJoCo.
    """
    geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
    if geom_id == -1:
        raise ValueError(f"Body '{frame_name}' not found in the MuJoCo model.")

    frame_pos = mj_data.xpos[geom_id]
    return frame_pos

def mj_frame_pos(mj_model, mj_data, frame_name: str) -> np.ndarray:
    """
    Get the geom position for a given frame name in MuJoCo.
    """
    geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, frame_name)
    if geom_id == -1:
        raise ValueError(f"Geom '{frame_name}' not found in the MuJoCo model.")

    frame_pos = mj_data.geom_xpos[geom_id]
    return frame_pos

def pin_joint_name2dof(model) -> Dict[str, int]:
    """
    Get joint name to DOF index map.
    """
    pin_joint_name2dof = {}

    for joint_id in range(1, model.njoints):  # Skip the universe joint (id=0)
        joint_name = model.names[joint_id]
        joint = model.joints[joint_id]
        # Check if the joint is actuated (not a fixed joint)
        if joint.nv > 0:  # Joint has configuration variables
            pin_joint_name2dof[joint_name] = joint.idx_v  # DOF index

    return pin_joint_name2dof

def pin_frame_pos(pin_model, pin_data, frame_name: str) -> np.ndarray:
    """
    Get the frame position in base frame for a given frame name in Pinocchio.

    Args:
        frame_name (str): Name of the frame.

    Returns:
        np.ndarray: Position of the frame in the world frame.
    """
    frame_id = pin_model.getFrameId(frame_name)
    if frame_id >= len(pin_model.frames):
        raise ValueError(f"Frame '{frame_name}' not found in the Pinocchio model.")
    
    # Get frame position in the world frame
    frame_pos = pin_data.oMf[frame_id].translation
    return frame_pos

def mj_joint_name2dof(mj_model) -> Dict[str, int]:
    """
    Get joint name to DOF index map.
    """
    mj_joint_name2dof = {
        mj_model.joint(i_jnt).name : int(mj_model.joint(i_jnt).dofadr)
        for i_jnt
        in range(mj_model.njnt) # Actuated joints
    }
    return mj_joint_name2dof
    
def mj_joint_name2act_id(mj_model) -> Dict[str, int]:
    joint_name2act_id = {
        mj_model.joint(mj_model.actuator_trnid[i, 0]).name # Joint name
        : i # act id
        for i in range(mj_model.nu)
    }
    return joint_name2act_id

def mj_body_pos(mj_model, mj_data, frame_name: str) -> np.ndarray:
    """
    Get the body position for a given frame name in MuJoCo.
    """
    geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
    if geom_id == -1:
        raise ValueError(f"Body '{frame_name}' not found in the MuJoCo model.")

    frame_pos = mj_data.xpos[geom_id]
    return frame_pos

def mj_frame_pos(mj_model, mj_data, frame_name: str) -> np.ndarray:
    """
    Get the geom position for a given frame name in MuJoCo.
    """
    geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, frame_name)
    if geom_id == -1:
        raise ValueError(f"Geom '{frame_name}' not found in the MuJoCo model.")

    frame_pos = mj_data.geom_xpos[geom_id]
    return frame_pos

def mj_2_pin_state(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo to Pinocchio state format.
    Convert quaternion format:
    qw, qx, qy, qz -> qx, qy, qz, qw
    """
    q_xyzw = q_wxyz.copy()
    q_xyzw[3:7] = np.take(
        q_wxyz[3:7],
        MJ2PIN_QUAT,
        mode="clip",
        )
    return q_xyzw

def pin_2_mj_state(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo to Pinocchio state format.
    Convert quaternion format:
    qx, qy, qz, qw -> qw, qx, qy, qz
    """
    q_wxyz = q_xyzw.copy()
    q_wxyz[3:7] = np.take(
        q_xyzw[3:7],
        PIN2MJ_QUAT,
        mode="clip",
        )
    return q_wxyz

def mj_2_pin_qv(q_mj : np.ndarray, v_mj : np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo to Pinocchio state and velocities format.
    qw, qx, qy, qz -> qx, qy, qz, qw
    lin vel : global -> local
    angular vel : local -> local (no change)
    """
    q_pin = mj_2_pin_state(q_mj)
    # Transform from world to base
    b_T_W = pin.XYZQUATToSE3(q_pin[:7]).inverse()
    R = b_T_W.rotation
    p = b_T_W.translation
    p_skew = get_skew_sim_mat(p)

    # v_b = [p] @ R @ w_W + R @ v_W
    #     = [p] @ R @ R.T @ w_B + R @ v_W
    #     = [p] @ w_B + R @ v_W
    v_pin = v_mj.copy()
    v_pin[:3] = p_skew @ v_mj[3:6] + R @ v_mj[:3]
    return q_pin, v_mj

def pin_2_mj_qv(q_pin : np.ndarray, v_pin : np.ndarray) -> np.ndarray:
    """
    Convert Pinocchio to MuJoCo state and velocities format.
    qx, qy, qz, qw -> qw, qx, qy, qz
    lin vel : local -> global
    angular vel : local -> local (no change)
    """
    q_mj = pin_2_mj_state(q_mj)
    # Transform from world to base
    W_T_b = pin.XYZQUATToSE3(q_pin[:7])
    R = W_T_b.rotation
    p = W_T_b.translation
    p_skew = get_skew_sim_mat(p)

    # v_W = [p] @ R @ w_b + R @ v_b
    v_mj = v_pin.copy()
    v_mj[:3] = p_skew @ R @ v_pin[3:6] + R @ v_pin[:3]
    return q_mj, v_mj

def add_frames_from_mujoco(
    pin_model, 
    mj_model, 
    frame_geometries: List[str]
) -> None:
    """
    Add frames to a Pinocchio model based on geometries defined in a MuJoCo model,
    using the parent joint names.

    Args:
        pin_model (pin.Model): The Pinocchio model to which frames will be added.
        mj_model (mujoco.MjModel): The MuJoCo model from which geometries will be retrieved.
        frame_geometries (List): List of geometry names in MuJoCo to be added as frames in Pinocchio.
    """
    for geom_name in frame_geometries:
        # Get geometry ID in MuJoCo
        geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id == -1:
            print(f"Geometry '{geom_name}' not found in MuJoCo model. Skipping.")
            continue

        # Get parent joint of the geometry
        parent_joint_id = mj_model.body_jntadr[mj_model.geom_bodyid[geom_id]]
        parent_joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, parent_joint_id)

        # Find corresponding Pinocchio joint ID
        if parent_joint_name not in pin_model.names:
            print(f"Joint '{parent_joint_name}' not found in Pinocchio model. Skipping.")
            continue
        pin_joint_id = pin_model.getJointId(parent_joint_name)

        # Get the position and orientation of the geometry in the body frame
        geom_pos = mj_model.geom_pos[geom_id]
        geom_quat = mj_model.geom_quat[geom_id]  # Quaternion in (w, x, y, z) format
        geom_rotation = pin.Quaternion(geom_quat[0], *geom_quat[1:]).toRotationMatrix()

        # Create SE3 transformation
        geom_to_joint = pin.SE3(geom_rotation, geom_pos)

        # Add the frame to the Pinocchio model
        frame_name = f"{geom_name}"
        new_frame = pin.Frame(
            frame_name,
            pin_joint_id,
            pin_joint_id,
            geom_to_joint,
            pin.FrameType.OP_FRAME,
        )
        pin_model.addFrame(new_frame)

        print(f"Added frame '{frame_name}' to Pinocchio model.")

def copy_motor_parameters(pin_model, mj_model: mujoco.MjModel) -> None:
    """
    Update motor parameters (friction, damping, and rotor inertia) in a Pinocchio model
    from a MuJoCo model.

    Args:
        pin_model (pin.Model): The Pinocchio model to update.
        mj_model (mujoco.MjModel): The MuJoCo model to use as a reference.
    """
    # Update friction
    pin_model.friction = mj_model.dof_frictionloss.copy()
    # Update damping
    pin_model.damping = mj_model.dof_damping.copy()
    # Update armature
    pin_model.rotorInertia = mj_model.dof_armature.copy()

    # Update kinematic limits
    nu = mj_model.nu

    # Update position limits
    pin_model.upperPositionLimit = mj_model.jnt_range[:, 1].copy()
    pin_model.lowerPositionLimit = mj_model.jnt_range[:, 0].copy()

    # Update effort limits
    pin_model.effortLimit[-nu:] = np.abs(np.max(np.hstack((
        mj_model.actuator_ctrlrange[:, 0],
        mj_model.actuator_ctrlrange[:, 1],
    )), axis=0))

def transform_points(
    B_T_A : np.ndarray,
    points_A : np.ndarray
    ) -> np.ndarray:
    """
    Batch transform a set of points from frame A to frame B. 

    Args:
        A_T_B (np.ndarray): SE3 transform of frame A expressed in frame B.
        points_A (np.ndarray): set of points expressed in frame A. Shape [N, 3]

    Returns:
        points_B (np.ndarray): set of points expressed in frame B. Shape [N, 3]
    """
    
    if len(points_A.shape) < 2:
        points_A = points_A[np.newaxis, :]
        
    assert points_A.shape[-1] == 3, "Points provided are not 3d points."     
        
    # Add a fourth homogeneous coordinate (1) to each point
    ones = np.ones((points_A.shape[0], 1))
    points_A_homogeneous = np.hstack((points_A, ones))
    # Apply the transformation matrix
    points_B_homogeneous = B_T_A @ points_A_homogeneous.T
    # Convert back to 3D coordinates
    points_B = points_B_homogeneous[:3, :].T
    
    return points_B

def get_skew_sim_mat(v : np.ndarray) -> np.ndarray:
    """
    Returns the skew symmetric matrix associated to a vector v. 
    """
    M = np.array([
        [   0., -v[2],  v[1],],
        [ v[2],    0., -v[0],],
        [-v[1],  v[0],    0.,],
    ])
    return M

def quat_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    sin_roll = np.sin(roll / 2)
    cos_roll = np.cos(roll / 2)
    sin_pitch = np.sin(pitch / 2)
    cos_pitch = np.cos(pitch / 2)
    sin_yaw = np.sin(yaw / 2)
    cos_yaw = np.cos(yaw / 2)

    qx = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
    qy = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
    qz = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw
    qw = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw

    return np.array([qw, qx, qy, qz])