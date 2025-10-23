#!/usr/bin/env python3
"""
Simple G1 Joint Control Test Script
===================================

This script provides a minimal example for sending a continuous stream of
PD control commands to a single joint of the G1 robot using DDS.

It is designed to be a simple, focused tool for debugging hardware and
controller mode issues.

- Connects to the robot via DDS.
- Reads the current joint states and `mode_machine`.
- Continuously sends commands to hold all joints at their current position,
  except for one target joint.
- Commands the target joint (Left Shoulder Roll) to a position offset
  from its initial state.
- Allows for adjustable Kp gain and feedforward torque.

Usage:
1. Ensure ATARI_NMPC root directory is in PYTHONPATH.
2. Run from terminal: `python simple_g1_joint_test.py <network_interface>`
   Example: `python simple_g1_joint_test.py enp7s0`
   If no interface is provided, it will use the default.
"""

import sys
import time
import numpy as np
import signal
from threading import Thread, Event
import json
import os

try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    # Assuming the G1 joint definitions are in this project
    from sdk_controller.robots.G1 import (
        BODY_MUJOCO_TO_DDS, MUJOCO_JOINT_NAMES, NUM_ACTIVE_BODY_JOINTS
    )
except ImportError as e:
    print(f"❌ Module import failed: {e}", file=sys.stderr)
    print("   Please ensure ATARI_NMPC root directory is added to your PYTHONPATH.", file=sys.stderr)
    print("   Example: export PYTHONPATH=$PYTHONPATH:/path/to/ATARI_NMPC", file=sys.stderr)
    sys.exit(1)

# --- Global Config Loading ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'global_config.json')

def load_default_interface_from_config():
    """Reads the default network interface from the global config file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get('network_interface', 'enp7s0') # Fallback if key is missing
    except (FileNotFoundError, json.JSONDecodeError):
        return 'enp7s0' # Fallback if file is missing or corrupt


# --- Configuration ---
TARGET_JOINT_NAME = "left_shoulder_roll_joint"
TARGET_JOINT_MJ_INDEX = 16 # Corresponds to 'left_shoulder_roll_joint'
TARGET_POSITION_OFFSET_DEG = 10.0  # Degrees to add to the initial position

# --- PD Gains ---
# Set Kp for the target joint. A higher value means stiffer control.
KP_TARGET =40.0
# Set Kd for all joints. This provides damping.
KD_ALL = 2.0
# Set Kp for all other (non-target) joints to hold their position.
KP_HOLD = 20.0

# --- Feedforward Torque ---
# Set a feedforward torque for the target joint.
TAU_FF_TARGET = 0.0  # Nm

CONTROL_FREQUENCY_HZ = 100  # Hz to send commands

# --- Global State ---
g_low_state = None
g_initial_joint_positions = {}
g_stop_event = Event()

def low_state_handler(msg: LowState_):
    """Callback function to handle incoming LowState messages."""
    global g_low_state
    g_low_state = msg

class G1JointTest:
    def __init__(self):
        self.running = True
        self.state_received = False
        self.low_cmd_pub = None
        self.low_state_sub = None
        self.control_thread = None

    def initialize_dds(self, channel_arg):
        """Initializes DDS with the provided network interface."""
        domain_id = 0 if channel_arg != "lo" else 1
        ChannelFactoryInitialize(domain_id, channel_arg)
        self.low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_pub.Init()
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self.low_state_handler, 10)
        print(f"DDS initialized. Using network interface: {channel_arg}")

    def capture_initial_positions(self):
        """Captures the initial positions of all joints from the first valid LowState message."""
        print("Waiting for first LowState message to initialize...")
        while g_low_state is None and self.running:
            time.sleep(0.1)
        
        if not self.running:
            return

        print("✅ Received first LowState. Capturing initial joint positions.")
        
        # Capture initial positions from the first valid state message
        # IMPORTANT: We iterate through the mapping to get all relevant joints, including arms.
        for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
            mujoco_name = MUJOCO_JOINT_NAMES[mj_idx]
            g_initial_joint_positions[mujoco_name] = g_low_state.motor_state[dds_idx].q

        if TARGET_JOINT_NAME not in g_initial_joint_positions:
            print(f"❌ FATAL: Target joint '{TARGET_JOINT_NAME}' not found in the initial state mapping.", file=sys.stderr)
            print("   Please check joint name and mappings in `sdk_controller.robots.G1`.", file=sys.stderr)
            self.running = False
            return

        print(f"Initial position for {TARGET_JOINT_NAME}: {np.rad2deg(g_initial_joint_positions[TARGET_JOINT_NAME]):.2f}°")
        self.state_received = True

    def control_loop(self):
        """The main control loop that sends commands at a fixed frequency."""
        global g_low_state, g_initial_joint_positions

        crc_calculator = CRC()
        low_cmd = unitree_hg_msg_dds__LowCmd_()
        
        # Calculate target position
        initial_q = g_initial_joint_positions[TARGET_JOINT_NAME]
        target_q_rad = initial_q + np.deg2rad(TARGET_POSITION_OFFSET_DEG)
        print(f"🎯 Commanding {TARGET_JOINT_NAME} to {np.rad2deg(target_q_rad):.2f}° (Offset: {TARGET_POSITION_OFFSET_DEG}°)")

        loop_start_time = time.time()
        while self.running:
            if g_low_state is None:
                print("⚠️ LowState not received, skipping command.", file=sys.stderr)
                time.sleep(1.0 / CONTROL_FREQUENCY_HZ)
                continue

            # Set the crucial mode_machine field
            low_cmd.mode_machine = g_low_state.mode_machine

            # Iterate through all body joints defined in the mapping
            for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
                motor_cmd = low_cmd.motor_cmd[dds_idx]
                mujoco_name = MUJOCO_JOINT_NAMES[mj_idx]
                
                motor_cmd.mode = 1  # 1: PD Mode
                motor_cmd.kd = KD_ALL

                if mujoco_name == TARGET_JOINT_NAME:
                    # Command the target joint
                    motor_cmd.q = target_q_rad
                    motor_cmd.kp = KP_TARGET
                    motor_cmd.dq = 0.0
                    motor_cmd.tau = TAU_FF_TARGET
                else:
                    # Command all other joints to hold their initial position
                    if mujoco_name in g_initial_joint_positions:
                        motor_cmd.q = g_initial_joint_positions[mujoco_name]
                        motor_cmd.kp = KP_HOLD
                        motor_cmd.dq = 0.0
                        motor_cmd.tau = 0.0
            
            # Calculate CRC and publish
            low_cmd.crc = crc_calculator.Crc(low_cmd)
            self.low_cmd_pub.Write(low_cmd)

            # Maintain control frequency
            time.sleep(max(0, (1.0 / CONTROL_FREQUENCY_HZ) - (time.time() - loop_start_time)))
            loop_start_time = time.time()

        print("\nControl loop stopped. Sending damping command.")
        # Send a final damping command for safety
        for dds_idx in BODY_MUJOCO_TO_DDS.values():
            motor_cmd = low_cmd.motor_cmd[dds_idx]
            motor_cmd.mode = 1
            motor_cmd.kp = 0.0
            motor_cmd.kd = KD_ALL # Leave some damping
            motor_cmd.q = 0.0
            motor_cmd.dq = 0.0
            motor_cmd.tau = 0.0
        low_cmd.crc = crc_calculator.Crc(low_cmd)
        self.low_cmd_pub.Write(low_cmd)


def main():
    """Main function to set up DDS and start the control thread."""
    g1_joint_test = G1JointTest()

    # Determine channel from command-line argument or config file
    if len(sys.argv) > 1:
        channel_arg = sys.argv[1]
    else:
        channel_arg = load_default_interface_from_config()
        print(f"Usage: python simple_g1_joint_test.py <network_interface>")
        print(f"No interface provided. Using default '{channel_arg}' from global_config.json")

    g1_joint_test.initialize_dds(channel_arg)

    # Allow time for the first state message to arrive
    time.sleep(1)
    
    # Check if a LowState message has been received
    if not g1_joint_test.state_received:
        print("Error: No LowState message received. Please check DDS communication.")
        sys.exit(1)

    # Start the background thread for sending commands
    g1_joint_test.control_thread = Thread(target=g1_joint_test.control_loop)
    g1_joint_test.control_thread.start()

    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Stopping...")
        g1_joint_test.running = False
        g1_joint_test.control_thread.join()
        print("Program finished.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        g1_joint_test.control_thread.join()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        g1_joint_test.running = False
        g1_joint_test.control_thread.join()
        print("Test finished.")
        sys.exit(0)


if __name__ == "__main__":
    main() 