from typing import List
import os, shutil
import numpy as np
import argparse
import time

from mj_pin.abstract import MjController, DataRecorder
from mj_pin.simulator import Simulator
from mj_pin.utils import get_robot_description
import cv2

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
    
def replay_video(video_path: str, fps : int = 24):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Replay', frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            time.sleep(1/fps)
    except KeyboardInterrupt:
        pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a robot with optional recording and visualization.")
    parser.add_argument("--robot_name", type=str, default="go2", help="Name of the robot to simulate.")
    parser.add_argument("--track_base", default=False, action="store_true", help="Camera tracks the base.")
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
    
    # Edit the video settings of the simulator
    sim.vs.set_high_quality()
    sim.vs.fps = 30
    sim.vs.playback_speed = 0.5
    # Can track an object. Otherwise it follows the camera motions in the viewer.
    # Works also without viewer (example with flag --track_base)
    if args.track_base:
        # Came
        sim.vs.set_side_view()
        sim.vs.track_obj = "base" # Needs to be in the mujoco model
    print("Default saving dir", sim.vs.video_dir)
    
    # Run the simulation with the provided controller etc.
    # The data is automatically saved at the end.
    use_viewer = not args.track_base
    sim.run(sim_time=0. if use_viewer else 4.,
            controller=pd_controller,
            record_video=True,
            use_viewer=use_viewer)
    
    # Replaying the video and deleting the folder
    video_files = os.listdir(sim.vs.video_dir)
    if video_files:
        video_name = video_files[0]
        video_path = os.path.join(sim.vs.video_dir, video_name)
        print("Replaying video", video_path)
        replay_video(video_path, sim.vs.fps)
        shutil.rmtree(sim.vs.video_dir)
        print("Video", video_path, "deleted.")