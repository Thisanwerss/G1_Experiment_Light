from typing import List
import os
import numpy as np
import argparse

from mj_pin.abstract import MjController, ParallelExecutorBase
from mj_pin.simulator import Simulator
from mj_pin.utils import get_robot_description

class MjPDController(MjController):
    def __init__(self, xml_path : str, Kp = 44., Kd = 3.):
        super().__init__(xml_path)
        self.Kp, self.Kd = Kp, Kd
        self.q_ref = None

    def get_torques(self, sim_step, mj_data):
        # Get pos, vel state
        q, v = self.get_state(mj_data)
        # Set reference as the first state
        if self.q_ref is None: self.q_ref = q[-self.nu:].copy()
        # Update torques_dof
        self.torques_dof[-self.nu:] = self.Kp * (self.q_ref - q[-self.nu:]) - self.Kd * v[-self.nu:]
        # torque map {joint name : torque value}
        torque_map = self.get_torque_map()
        
        return torque_map

if __name__ == "__main__":
    robot_name = "go2"

    # Load robot information and paths
    robot_description = get_robot_description(robot_name)
    mj_eeff_frame_name = ["FL", "FR", "RL", "RR"]
    
    # Dummy executor example
    class PDGoalParallelExecutor(ParallelExecutorBase):
        def __init__(self, q0 : np.ndarray, Kp : float, Kd : float):
            # Use verbose to debug
            super().__init__(verbose=False)
            self.Kp, self.Kd = Kp, Kd
            self.q0 = q0
            
        def create_job(self, job_id):
            # Create random PD joint target
            q_ref = self.q0 + np.random.randn(*self.q0.shape) / 8.
            # Return the args to be used by run_job
            return {"q_ref" : q_ref}
        
        def run_job(self, job_id, **kwargs) -> bool:
            q_ref = kwargs.get("q_ref")
            # Create pd controller with global variable xml_path
            pd_controller = MjPDController(robot_description.xml_path, self.Kp, self.Kd)
            # Set q_ref
            pd_controller.q_ref = q_ref[-pd_controller.nu:]
            
            # Add datarecorder if you need......
            
            # Simulator
            sim = Simulator(robot_description.xml_scene_path, sim_dt=1e-3, viewer_dt=1/40)
            # Run sim in parallel without viewer (no sleep time on the physics thread), stop on collision
            sim.run(sim_time=8.,
                    use_viewer=False,
                    controller=pd_controller,
                    allowed_collision=[0] + mj_eeff_frame_name)
            # Return boolean
            success = not sim.collided
            return success

    # Run the parallel execution
    sim = Simulator(robot_description.xml_scene_path)
    q0, _ = sim.get_initial_state()
    del sim
    executor = PDGoalParallelExecutor(q0, Kp = 44., Kd = 3.)
    n_cores = 6
    n_jobs = 100
    executor.run(n_cores, n_jobs=n_jobs)