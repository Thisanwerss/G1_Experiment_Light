from typing import List
import os
import numpy as np
import argparse

from mj_pin.abstract import MjController, DataRecorder
from mj_pin.simulator import Simulator
from mj_pin.utils import get_robot_description

class UpDownController(MjController):
    def __init__(self, xml_path : str, Kp = 44., Kd = 3.):
        super().__init__(xml_path)
        self.Kp, self.Kd = Kp, Kd
        self.q_ref = None
        sine_freq = 0.6
        self.gain_sine = lambda time : (np.sin(time * sine_freq) ** 2 + 0.1)

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

class StateDataRecorder(DataRecorder):
    def __init__(
        self,
        record_dir: str = "",
        record_step: int = 1,
    ) -> None:
        """
        A simple data recorder that saves simulation data to a .npz file.
        """
        super().__init__(record_dir, record_step)
        self.data = {}
        self.reset()

    def reset(self) -> None:
        """
        Reset the recorder data.
        """
        self.data = {"time": [], "q": [], "v": [], "ctrl": [],}

    def save(self) -> None:
        """
        Save the recorded data to a file in the specified directory.
        """
        print("Number of samples collected:", len(self.data["time"]))
        if not self.record_dir:
            self.record_dir = os.getcwd()
        # Uncomment to save data
        # os.makedirs(self.record_dir, exist_ok=True)

        timestamp = self.get_date_time_str()
        file_path = os.path.join(self.record_dir, f"simulation_data_{timestamp}.npz")

        try:
            # Uncomment to save data
            # np.savez(file_path, **self.data)
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def record(self, mj_data) -> None:
        """
        Record simulation data at the current simulation step.

        Args:
            sim_step (int): Current simulation step.
            mj_data (Any): MuJoCo data object.
            **kwargs: Additional data to record.
        """
        # Record time and state
        self.data["time"].append(round(mj_data.time, 4))
        self.data["q"].append(mj_data.qpos.copy())
        self.data["v"].append(mj_data.qvel.copy())
        self.data["ctrl"].append(mj_data.ctrl.copy())
        # mj_data needs to be copied!!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a robot with optional recording and visualization.")
    parser.add_argument("--robot_name", type=str, default="go2", help="Name of the robot to simulate.")
    args = parser.parse_args()

    # Load robot information and paths
    robot_description = get_robot_description(args.robot_name)
    mj_eeff_frame_name = ["FL", "FR", "RL", "RR"]

    # Load the simulator
    # Will start two threads: one for the viewer and one for the physics simulation.
    # Viewer and physics are updated at different rate.
    sim = Simulator(robot_description.xml_scene_path, sim_dt=1e-3, viewer_dt=1/40)

    # PD Controller, called every simulation step
    pd_controller = UpDownController(robot_description.xml_path)
    
    # Data recorder, called every 10 simulation step
    record_freq = 10
    record_state_data = StateDataRecorder("./data", record_freq)

    # Run the simulation for 4 seconds
    # The data is automatically saved at the end.
    sim.run(sim_time=4.,
            controller=pd_controller,
            data_recorder=record_state_data)
    
    # Visualize recorded trajectory
    print("Replaying recording...")
    q_traj = np.array(record_state_data.data["q"])
    time_traj = np.array(record_state_data.data["time"])
    sim.visualize_trajectory(q_traj, time_traj)