#!/usr/bin/env python3
"""
Dummy DDS Controller
====================
This script sends a fixed control command directly to the G1 robot via DDS,
making it hold a slight knee-bent standing posture.

It does not depend on ZMQ or any external policies and is mainly used for testing
the DDS communication link and the robot's response to PD commands.

Usage:
1. Local loopback test (lo mode):
   python dummy_dds_controller.py --channel lo
2. Control a real robot (Vicon required):
   python dummy_dds_controller.py --channel <network_interface>

Command to launch Vicon:
ros2 launch vicon_receiver client.launch.py
"""
import argparse
import time
from typing import Dict, Any, Optional
from threading import Event
import signal
import sys
import numpy as np
import json

# --- Color Printing Utility ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(tag, message):
    tag_map = {
        "ERROR": bcolors.FAIL,
        "WARNING": bcolors.WARNING,
        "SUCCEED": bcolors.OKGREEN,
        "INFO": bcolors.OKBLUE,
    }
    color = tag_map.get(tag, bcolors.ENDC)
    print(f"{color}[{tag}]{bcolors.ENDC} {message}")


# --- Global Configuration Loading ---
try:
    with open("global_config.json", "r") as f:
        GLOBAL_CONFIG = json.load(f)
    VICON_Z_OFFSET = GLOBAL_CONFIG.get("vicon_z_offset", 0.0)
    print_colored("SUCCEED", f"Configuration loaded from global_config.json, VICON_Z_OFFSET={VICON_Z_OFFSET}")
except FileNotFoundError:
    print_colored("WARNING", "global_config.json not found, using default values.")
    VICON_Z_OFFSET = 0.0
except json.JSONDecodeError:
    print_colored("ERROR", "Failed to parse global_config.json, using default values.")
    VICON_Z_OFFSET = 0.0


# Real robot related imports
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from sdk_controller.robots.G1 import *
from sdk_controller.abstract_biped import HGSDKController
from typing import Dict, Any

# Vicon/ROS2 related imports - removed, using DDS Vicon subscription


class CEMSDKController(HGSDKController):
    """CEM Controller - Receives PD targets from an external policy via ZMQ, specialized for the G1 robot"""
    
    def __init__(self, simulate: bool = False, robot_config=None, xml_path: str = "", vicon_required: bool = True, lo_mode: bool = False, kp_scale_factor: float = 1.0, safety_profile: str = "default"):
        """
        Initializes the CEM controller.
        
        Args:
            simulate: Whether to run in simulation mode.
            robot_config: Robot configuration.
            xml_path: Path to the URDF/XML file.
            vicon_required: Whether Vicon positioning is required.
            lo_mode: Whether to run in loopback mode.
            kp_scale_factor: Kp gain scaling factor.
            safety_profile: Name of the safety profile configuration.
        """
        print_colored("INFO", "Initializing CEMSDKController (DDS Vicon mode)")
        print(f"   Simulation mode: {simulate}")
        print(f"   Vicon required: {vicon_required}")
        print(f"   Loopback mode: {lo_mode}")
        
        # Initialize HGSDKController
        super().__init__(
            simulate=simulate,
            robot_config=robot_config,
            xml_path=xml_path,
            vicon_required=vicon_required,
            lo_mode=lo_mode,
            kp_scale_factor=kp_scale_factor,
            safety_profile=safety_profile
        )
        
        # CEM control related states
        self.current_pd_targets = None
        self.waiting_for_targets = True
        self.safety_emergency_stop = False
        
        # Vicon state cache - The parent class HGSDKController handles DDS subscription automatically.
        if vicon_required:
            print("   DDS Vicon subscriber has been automatically initialized by the parent class.")
        
        print_colored("SUCCEED", "CEMSDKController initialization complete.")
        
    def update_motor_cmd(self, time: float):
        """Implements the abstract method - CEMSDKController is mainly controlled by external PD targets."""
        # This method is usually not called when using external PD targets.
        # It is kept as a placeholder or for emergency handling.
        if self.safety_emergency_stop:
            print_colored("WARNING", "Safety emergency stop: Switching to damping mode.")
            self.damping_motor_cmd()
        else:
            # If there are no external targets, use the default standing posture.
            if self.current_pd_targets is None:
                print_colored("WARNING", "No external PD targets, using default standing posture.")
                self.update_motor_cmd_from_pd_targets(STAND_UP_JOINT_POS)
    
    def get_robot_state(self) -> Optional[Dict[str, Any]]:
        """
        Gets the G1 robot state - compatible with ZMQ bridge format.
        
        Returns:
            A dictionary with robot state or None if data is invalid.
        """
        if self.lo_mode:
            # Loopback mode: return a dummy state (fixed standing posture).
            return self._get_dummy_state_for_cem()

        # Update joint states from DDS
        self.update_q_v_from_lowstate()
        self.update_hand_q_v_from_handstate()
        
        # Initialize mocap values to default
        mocap_pos_to_send = np.zeros(3)
        mocap_quat_to_send = np.array([1, 0, 0, 0])

        # Update base state from Vicon (via DDS)
        if self.vicon_required:
            p, q, v, w = None, None, None, None
            
            # Check for Vicon DDS message timeout
            current_time = time.time()
            vicon_timeout = 0.5 # seconds

            if self.last_vicon_pose is not None:
                pose_timestamp = self.last_vicon_pose.header.stamp.sec + self.last_vicon_pose.header.stamp.nanosec * 1e-9
                if current_time - pose_timestamp < vicon_timeout:
                    p = np.array([
                        self.last_vicon_pose.pose.position.x,
                        self.last_vicon_pose.pose.position.y,
                        self.last_vicon_pose.pose.position.z,
                    ])
                    q = np.array([
                        self.last_vicon_pose.pose.orientation.w,
                        self.last_vicon_pose.pose.orientation.x,
                        self.last_vicon_pose.pose.orientation.y,
                        self.last_vicon_pose.pose.orientation.z,
                    ])
            
            if self.last_vicon_twist is not None:
                twist_timestamp = self.last_vicon_twist.header.stamp.sec + self.last_vicon_twist.header.stamp.nanosec * 1e-9
                if current_time - twist_timestamp < vicon_timeout:
                    v = np.array([
                        self.last_vicon_twist.twist.linear.x,
                        self.last_vicon_twist.twist.linear.y,
                        self.last_vicon_twist.twist.linear.z,
                    ])
                    w = np.array([
                        self.last_vicon_twist.twist.angular.x,
                        self.last_vicon_twist.twist.angular.y,
                        self.last_vicon_twist.twist.angular.z,
                    ])
            
            # If Vicon data is valid, update base state
            if p is not None and q is not None and v is not None and w is not None:
                self._q[0:3] = p
                self._q[3:7] = q  # (w, x, y, z)
                self._v[0:3] = v
                self._v[3:6] = w
                # Also populate mocap fields with Vicon data to align with simulation
                mocap_pos_to_send = p.copy()
                mocap_quat_to_send = q.copy()
            else:
                # Vicon data is invalid or timed out
                print_colored("ERROR", "get_robot_state: Invalid Vicon data (DDS timeout or not received), returning None.")
                return None
        
        # Check if DDS data is valid (a simple integrity check)
        # 7: onwards are joint qpos. If they are all zero, it likely means no DDS data was received.
        if np.all(self._q[7:] == 0):
            print_colored("ERROR", "get_robot_state: Joint data is all zero, possibly no DDS data received. Returning None.")
            return None

        # Return in ZMQ compatible format
        return {
            'qpos': self._q.copy(),
            'qvel': self._v.copy(),
            'mocap_pos': mocap_pos_to_send,
            'mocap_quat': mocap_quat_to_send,
            'time': time.time()
        }

    def _get_dummy_state_for_cem(self) -> Dict[str, Any]:
        """Generates a dummy state for loopback mode."""
        dummy_qpos = np.zeros(48)
        dummy_qvel = np.zeros(47)
        dummy_qpos[2] = 0.75  # z
        dummy_qpos[3] = 1.0   # qw
        return {
            'qpos': dummy_qpos, 'qvel': dummy_qvel,
            'mocap_pos': np.zeros(3), 'mocap_quat': np.array([1,0,0,0]),
            'time': time.time()
        }
    
    def send_motor_command(self, time: float, pd_targets: Optional[np.ndarray] = None):
        """Sends motor control commands with safety checks."""
        if pd_targets is not None:
            self.current_pd_targets = pd_targets.copy()
        
        # Call parent class method
        super().send_motor_command(time, pd_targets)


class DummyController:
    """A simple DDS controller to send a fixed standing command."""
    def __init__(
        self,
        channel: str = "lo",
        control_frequency: float = 50.0,
        kp_scale_factor: float = 1.0,
        safety_profile: str = "default"
    ):
        self.channel = channel
        self.control_frequency = control_frequency
        self.kp_scale_factor = kp_scale_factor
        self.safety_profile = safety_profile
        self.control_dt = 1.0 / self.control_frequency
        
        self.running = Event()
        
        print_colored("INFO", "Initializing Dummy DDS Controller")
        print(f"   Mode: Real robot/lo mode (Channel: {channel})")
        print(f"   Control Frequency: {control_frequency} Hz")

        # 1. Initialize DDS
        self._setup_dds()

        # 2. Initialize CEM controller backend
        self.cem_controller = self._setup_cem_controller()
            
        # 3. Define target posture
        self.target_pos = self._define_target_pose()
        
        print_colored("SUCCEED", "Dummy controller initialization complete.")

    def _setup_dds(self):
        """Sets up DDS communication."""
        if self.channel == "lo":
            print("   Using lo interface (domain_id=1)")
            ChannelFactoryInitialize(1, "lo")
        else:
            print(f"   Using real network interface: {self.channel} (domain_id=0)")
            ChannelFactoryInitialize(0, self.channel)

    def _setup_cem_controller(self) -> CEMSDKController:
        """Sets up and returns a CEM controller instance."""
        print_colored("INFO", f"Setting up CEM control mode (Channel: {self.channel})...")
        controller = CEMSDKController(
            simulate=False,
            robot_config=None,
            xml_path="g1_model/g1_lab.xml",
            vicon_required=(self.channel != "lo"),
            lo_mode=(self.channel == "lo"),
            kp_scale_factor=self.kp_scale_factor,
            safety_profile=self.safety_profile
        )
        print_colored("SUCCEED", "CEM controller setup complete.")
        return controller
    
    def _define_target_pose(self) -> np.ndarray:
        """Defines and returns the target joint positions."""
        # Create a target array for 27 active body joints
        target_q = np.zeros(NUM_ACTIVE_BODY_JOINTS)
        
        # Set a slight knee bend based on mujoco_index in G1.py
        # left_knee_joint (mujoco_index: 3)
        # right_knee_joint (mujoco_index: 9)
        target_q[3] = 0.1  # Left knee
        target_q[9] = 0.1  # Right knee

        print_colored("INFO", "Target posture set (slight knee bend at 0.1 rad).")
        return target_q

    def run(self):
        """Runs the main control loop."""
        print_colored("INFO", "Starting Dummy Controller main loop.")
        self.running.set()
        
        # Before starting, wait for a valid robot state to ensure DDS and Vicon are connected
        print("   Waiting for a valid initial robot state...")
        initial_state = None
        while initial_state is None and self.running.is_set():
            initial_state = self.cem_controller.get_robot_state()
            if initial_state is None:
                if not self.running.is_set(): break
                print("  ...still waiting, retrying in 0.5s...")
                time.sleep(0.5)

        if not self.running.is_set():
            self.stop()
            return
            
        print_colored("SUCCEED", "Successfully received initial state, starting to send control commands...")

        try:
            while self.running.is_set():
                # Check if the robot state is valid
                state = self.cem_controller.get_robot_state()
                if state is None:
                    print_colored("ERROR", "Lost robot state (Vicon or DDS timeout). Stopping for safety.")
                    print("   Sending damping commands...")
                    for _ in range(5):
                        self.cem_controller.damping_motor_cmd()
                        time.sleep(0.01)
                    self.stop()
                    break

                # Send the target posture command at a fixed frequency
                self.cem_controller.send_motor_command(
                    time=time.time(), 
                    pd_targets=self.target_pos
                )
                
                time.sleep(self.control_dt)

        except KeyboardInterrupt:
            print("\n")
            print_colored("WARNING", "Interrupt signal received...")
        
        finally:
            print("   Sending final damping commands...")
            for _ in range(5):
                self.cem_controller.damping_motor_cmd()
                time.sleep(0.01)
            self.stop()

    def stop(self):
        """Stops the controller."""
        print_colored("INFO", "Stopping Dummy Controller...")
        self.running.clear()
        # The cem_controller will automatically clean up DDS resources when the parent process exits
        print_colored("SUCCEED", "Dummy Controller stopped.")


def main():
    parser = argparse.ArgumentParser(description="Dummy DDS G1 Controller")
    parser.add_argument(
        "--channel",
        type=str,
        default="lo",
        help="DDS channel: 'lo' for local loopback, or a network interface name."
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=100.0,
        help="Control frequency (Hz)."
    )
    parser.add_argument(
        "--kp_scale",
        type=float,
        default=1.0,
        help="Global Kp gain scaling factor (0.0-1.0)."
    )
    parser.add_argument(
        "--safety_profile",
        type=str,
        default="default",
        choices=["default", "conservative"],
        help="Select the safety layer profile ('default' or 'conservative')."
    )
    
    args = parser.parse_args()
    
    controller = None
    def signal_handler(sig, frame):
        print("\n")
        print_colored("WARNING", "Stop signal received.")
        if controller:
            controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    controller = DummyController(
        channel=args.channel,
        control_frequency=args.frequency,
        kp_scale_factor=args.kp_scale,
        safety_profile=args.safety_profile
    )
    
    controller.run()


if __name__ == "__main__":
    main()
