import numpy as np
import argparse

from mj_pin.simulator import Simulator
from mj_pin.utils import get_robot_description
from mj_pin.abstract import PinController, MjController

class MjPDController(MjController):
    def __init__(self, xml_path : str, Kp = 44., Kd = 3.):
        super().__init__(xml_path)
        self.Kp, self.Kd = Kp, Kd
        self.q_ref = None

    def compute_torques_dof(self, mj_data):
        # Get pos, vel state
        q, v = self.get_state(mj_data)
        # Set reference as the first state
        if self.q_ref is None: self.q_ref = q[-self.nu:].copy()
        # Update torques_dof
        self.torques_dof[-self.nu:] = self.Kp * (self.q_ref - q[-self.nu:]) - self.Kd * v[-self.nu:]
    
class PinPDController(PinController):
    def __init__(self, urdf_path, Kp = 44., Kd = 3.):
        super().__init__(urdf_path, floating_base_quat=True)
        self.Kp, self.Kd = Kp, Kd
        self.q_ref = None
    
    def compute_torques_dof(self, mj_data):
        # Get pos, vel state (in pinocchio format)
        q, v = self.get_state(mj_data)
        # Set reference as the first state
        if self.q_ref is None: self.q_ref = q[-self.nu:].copy()
        # Update torques_dof
        self.torques_dof[-self.nu:] = self.Kp * (self.q_ref - q[-self.nu:]) - self.Kd * v[-self.nu:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a robot with optional recording and visualization.")
    parser.add_argument("--robot_name", type=str, default="go2", help="Name of the robot to simulate.")
    parser.add_argument("--pin", default=False, action="store_true", help="Use pinocchio controller.")
    args = parser.parse_args()

    # Load robot information and paths
    robot_description = get_robot_description(args.robot_name)

    # Load the simulator
    # Will start two threads: one for the viewer and one for the physics simulation.
    # Viewer and physics are updated at different rate.
    sim = Simulator(robot_description.xml_scene_path, sim_dt=1e-3, viewer_dt=1/40, verbose=True)

    # PD Controller, called every simulation step
    pd_controller = MjPDController(robot_description.xml_scene_path)
    if args.pin:
        print("Using PinPDController...")
        # Pinocchio joint names should match MuJoCo joint names
        pd_controller = PinPDController(robot_description.urdf_path)
    else:
        print("Using MjPDController...")
        pd_controller = MjPDController(robot_description.xml_scene_path)
        
    # Run the simulation with the provided controller etc.
    sim.run(controller=pd_controller)
    
    # Set different initial state
    print("Running with a different intial state")
    q0, v0 = sim.get_initial_state()
    q0[2] += 1.
    sim.set_initial_state(q0, v0)
    sim.run(controller=pd_controller)
    