import numpy as np
from sdk_controller.robots import Go2
import pinocchio as pin
from mj_pin.abstract import VisualCallback, DataRecorder # type: ignore
from mj_pin.simulator import Simulator # type: ignore
from mj_pin.utils import get_robot_description   # type: ignore

# MPC Controller
robot_desc = get_robot_description("go2")
feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


Vi_T_IMU = np.eye(4)
Vi_T_IMU[:3, :3] = Go2.R_IMU_IN_VICON
Vi_T_IMU[:3, 3] = Go2.P_IMU_IN_VICON

IMU_T_Vi = np.eye(4)
IMU_T_Vi[:3, :3] = Go2.R_IMU_IN_VICON.T
IMU_T_Vi[:3, 3] = -Go2.R_IMU_IN_VICON.T @ Go2.P_IMU_IN_VICON

# print("Transform from Vicon to Base in Vicon")
# print("Position", Go2.Vi_T_Base[:3, 3])
# print("Euler", pin.rpy.matrixToRpy(Go2.Vi_T_Base[:3, :3]))

class ViconFrame(VisualCallback):
    def add_visuals(self, mj_data):
        # Get IMU position from the MuJoCo site
        imu_position = mj_data.site_xpos[0]
        imu_mat = mj_data.site_xmat[0].reshape(3, 3, order="A")
        W_T_IMU = np.eye(4)
        W_T_IMU[:3, :3] = imu_mat
        W_T_IMU[:3, 3] = imu_position
        
        W_T_Vi = W_T_IMU @ IMU_T_Vi
        # Add a sphere at the Vicon frame position
        frame_position = W_T_Vi[:3, 3]
        self.add_sphere(frame_position, radius=0.01, rgba=[0, 0, 0, 1])
        
        scale_axis = 0.05
        Vi_T_X = np.eye(4)
        Vi_T_X[:3, 3] = np.array([1, 0, 0]) * scale_axis
        
        Vi_T_Y = np.eye(4)
        Vi_T_Y[:3, 3] = np.array([0, 1, 0]) * scale_axis
        
        Vi_T_Z = np.eye(4)
        Vi_T_Z[:3, 3] = np.array([0, 0, 1]) * scale_axis
        
        W_T_X = W_T_Vi @ Vi_T_X
        frame_position = W_T_X[:3, 3]
        self.add_sphere(frame_position, radius=0.01, rgba=[1, 0, 0, 1])
        W_T_Y = W_T_Vi @ Vi_T_Y
        frame_position = W_T_Y[:3, 3]
        self.add_sphere(frame_position, radius=0.01, rgba=[0, 1, 0, 1])
        W_T_Z = W_T_Vi @ Vi_T_Z
        frame_position = W_T_Z[:3, 3]
        self.add_sphere(frame_position, radius=0.01, rgba=[0, 0, 1, 1])
        
        
        
        
# Simulator with visual callback and state data recorder
vis_feet_pos = ViconFrame()

sim = Simulator(robot_desc.xml_scene_path)
sim.run(
    visual_callback=vis_feet_pos,
    )