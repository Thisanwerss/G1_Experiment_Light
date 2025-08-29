#!/usr/bin/env python3
"""
Vicon Ping Tool

A diagnostic tool to check the Vicon data pipeline.

Modes:
1. Direct Mode (default): Connects directly to the Vicon server and prints the raw pose data.
   - Usage: python Ping_Vicon.py --object_name <name>

2. DDS Mode: Subscribes to the DDS topic for Vicon pose and prints the received data.
   - Usage: python Ping_Vicon.py --dds --object_name <name>

3. Simulation/Visualization Mode: In either mode, add --sim to visualize the pose in MuJoCo.
   - A simple scene with a box will be loaded, and the box will be moved to the tracked object's pose.
   - Usage: python Ping_Vicon.py --object_name <name> --sim
   - Usage: python Ping_Vicon.py --dds --object_name <name> --sim
"""

import argparse
import time
import numpy as np
import threading

# MuJoCo for visualization
import mujoco
import mujoco.viewer

# Vicon direct connection
import pyvicon_datastream as pv

# DDS for subscription mode
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
from sdk_controller.topics import TOPIC_VICON_POSE


class ViconPinger:
    def __init__(self, object_name: str, use_dds: bool, sim: bool, vicon_ip: str, channel: str):
        self.object_name = object_name
        self.use_dds = use_dds
        self.sim = sim
        self.vicon_ip = vicon_ip
        self.channel = channel

        self.latest_pose = np.array([0., 0., 0.])
        self.latest_quat = np.array([1., 0., 0., 0.]) # w, x, y, z

        self.viewer = None
        self.mj_model = None
        self.mj_data = None
        
        self.running = threading.Event()
        self.running.set()

    def setup_simulation(self):
        """Sets up the MuJoCo viewer for visualization."""
        xml = f"""
        <mujoco>
          <worldbody>
            <light name="top" pos="0 0 1.5"/>
            <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
            <body name="tracked_object" pos="0 0 0">
              <freejoint/>
              <geom type="box" size=".1 .2 .05" rgba="1 0 0 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        print("✅ MuJoCo visualization setup complete.")

    def run(self):
        """Starts the appropriate pinger mode."""
        if self.sim:
            self.setup_simulation()
            # Launch the viewer in a separate thread
            viewer_thread = threading.Thread(target=self._launch_viewer, daemon=True)
            viewer_thread.start()
            # Give the viewer time to launch
            time.sleep(2)

        if self.use_dds:
            print(f"📡 Running in DDS mode. Subscribing to topic '{TOPIC_VICON_POSE}' for object '{self.object_name}'.")
            self._run_dds_pinger()
        else:
            print(f"🛰️  Running in Direct mode. Connecting to Vicon at {self.vicon_ip} for object '{self.object_name}'.")
            self._run_direct_pinger()
            
    def _launch_viewer(self):
        """Helper to launch the passive viewer."""
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as v:
            self.viewer = v
            # Keep the viewer alive
            while self.running.is_set():
                time.sleep(0.1)
        print("Viewer closed.")

    def _update_sim(self):
        """Updates the object's pose in the MuJoCo simulation."""
        if self.viewer and self.viewer.is_running():
            self.mj_data.qpos[0:3] = self.latest_pose
            self.mj_data.qpos[3:7] = self.latest_quat
            self.viewer.sync()

    def _dds_pose_handler(self, msg: PoseStamped_):
        """DDS message handler."""
        if msg.header.frame_id == self.object_name:
            p = msg.pose.position
            q = msg.pose.orientation
            self.latest_pose = np.array([p.x, p.y, p.z])
            self.latest_quat = np.array([q.w, q.x, q.y, q.z])
            
            print(f"Received Pose via DDS: Pos(x,y,z): {p.x:+.3f}, {p.y:+.3f}, {p.z:+.3f}")
            if self.sim:
                self._update_sim()

    def _run_dds_pinger(self):
        """Subscribes to DDS and prints data."""
        if self.channel == "lo":
            ChannelFactoryInitialize(1, "lo")
        else:
            ChannelFactoryInitialize(0, self.channel)
            
        sub = ChannelSubscriber(TOPIC_VICON_POSE, PoseStamped_)
        sub.Init(self._dds_pose_handler, 10)
        
        while self.running.is_set():
            time.sleep(1)

    def _run_direct_pinger(self):
        """Connects directly to Vicon and prints data."""
        client = pv.PyViconDatastream()
        while not client.is_connected() and self.running.is_set():
            print(f"Connecting to Vicon at {self.vicon_ip}...")
            client.connect(self.vicon_ip)
            time.sleep(1)
        
        if not self.running.is_set(): return
            
        client.set_stream_mode(pv.StreamMode.ServerPush)
        client.set_axis_mapping(pv.Direction.Forward, pv.Direction.Left, pv.Direction.Up)

        while self.running.is_set():
            try:
                if client.get_frame() == pv.Result.Success:
                    trans = client.get_segment_global_translation(self.object_name, self.object_name)
                    rot = client.get_segment_global_quaternion(self.object_name, self.object_name)

                    if trans is not None and rot is not None:
                        self.latest_pose = np.array(trans) / 1000.0  # Convert to meters
                        self.latest_quat = np.array(rot)
                        print(f"Direct Vicon Pose: Pos(x,y,z): {self.latest_pose[0]:+.3f}, {self.latest_pose[1]:+.3f}, {self.latest_pose[2]:+.3f}")
                        if self.sim:
                            self._update_sim()
                time.sleep(0.01) # ~100Hz
            except Exception as e:
                print(f"An error occurred in direct mode: {e}")
                break

    def stop(self):
        print("\nStopping Vicon Pinger...")
        self.running.clear()
        if self.viewer:
            self.viewer.close()

if __name__ == "__main__":
    from sdk_controller.robots import G1

    parser = argparse.ArgumentParser(description="Vicon Ping and Visualization Tool")
    parser.add_argument("--dds", action="store_true", help="Run in DDS subscription mode.")
    parser.add_argument("--sim", action="store_true", help="Enable MuJoCo visualization.")
    parser.add_argument("--object_name", type=str, default=G1.OBJECT_NAME, help="Object name to track in Vicon.")
    parser.add_argument("--vicon_ip", type=str, default="192.168.123.100:801", help="Vicon server IP and port.")
    parser.add_argument("--channel", type=str, default="lo", help="DDS channel for communication.")
    
    args = parser.parse_args()

    pinger = ViconPinger(
        object_name=args.object_name,
        use_dds=args.dds,
        sim=args.sim,
        vicon_ip=args.vicon_ip,
        channel=args.channel
    )

    try:
        pinger.run()
    except KeyboardInterrupt:
        pinger.stop()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        pinger.stop() 