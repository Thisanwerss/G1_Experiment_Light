import numpy as np

from mj_pin.abstract import MjController
from mj_pin.simulator import Simulator
from mj_pin.utils import get_robot_description

class UpDownController(MjController):
    def __init__(self, xml_path : str, Kp = 44., Kd = 3.):
        super().__init__(xml_path)
        self.Kp, self.Kd = Kp, Kd
        self.q_ref = None
        sine_freq = 0.4
        self.gain_sine = lambda time : (np.cos(time * sine_freq) ** 2 + 0.1)

    def get_torques(self, sim_step, mj_data):
        # Get pos, vel state
        q, v = self.get_state(mj_data)
        # Set reference as the first state
        if self.q_ref is None: self.q_ref = q[-self.nu:].copy()
        # Update torques_dof
        
        time = mj_data.time
        gain = self.gain_sine(time)
        Kp_t, Kd_t = gain * self.Kp, gain * self.Kd
        self.torques_dof[-self.nu:] = Kp_t * (self.q_ref - q[-self.nu:]) - Kd_t * v[-self.nu:]
        # torque map {joint name : torque value}
        torque_map = self.get_torque_map()
        
        return torque_map

if __name__ == "__main__":
    robot_name = "go2"

    # Load robot information and paths
    robot_description = get_robot_description(robot_name)
    mj_eeff_frame_name = ["FL", "FR", "RL", "RR"]

    # Simulator
    sim = Simulator(robot_description.xml_scene_path, sim_dt=1e-3, viewer_dt=1/40, verbose=True)

    # PD Controller, called every simulation step
    pd_controller = UpDownController(robot_description.xml_path)

    # Run the simulation until there is a collision with one geometry
    # that isn't in allowed_collision 
    # It can be name or geometry id (here floor has id 0)
    allowed_collision = [0] + mj_eeff_frame_name
    sim.run(controller=pd_controller, allowed_collision=allowed_collision)