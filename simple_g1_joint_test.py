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

def control_loop(low_cmd_pub: ChannelPublisher):
    """The main control loop that sends commands at a fixed frequency."""
    global g_low_state, g_initial_joint_positions

    print("Waiting for first LowState message to initialize...")
    while g_low_state is None and not g_stop_event.is_set():
        time.sleep(0.1)
    
    if g_stop_event.is_set():
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
        return

    print(f"Initial position for {TARGET_JOINT_NAME}: {np.rad2deg(g_initial_joint_positions[TARGET_JOINT_NAME]):.2f}°")
    
    crc_calculator = CRC()
    low_cmd = unitree_hg_msg_dds__LowCmd_()
    
    # Calculate target position
    initial_q = g_initial_joint_positions[TARGET_JOINT_NAME]
    target_q_rad = initial_q + np.deg2rad(TARGET_POSITION_OFFSET_DEG)
    print(f"🎯 Commanding {TARGET_JOINT_NAME} to {np.rad2deg(target_q_rad):.2f}° (Offset: {TARGET_POSITION_OFFSET_DEG}°)")

    loop_start_time = time.time()
    while not g_stop_event.is_set():
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
        low_cmd_pub.Write(low_cmd)

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
    low_cmd_pub.Write(low_cmd)


def main():
    """Main function to set up DDS and start the control thread."""
    if len(sys.argv) > 1:
        channel_arg = sys.argv[1]
        print(f"Using network interface: {channel_arg}")
    else:
        channel_arg = "enp7s0" # Default interface
        #channel_arg = "lo" # Default interface
        print(f"No network interface provided. Defaulting to: {channel_arg}")

    # Initialize DDS
    domain_id = 0 if channel_arg != "lo" else 1
    ChannelFactoryInitialize(domain_id, channel_arg)

    # Create publisher and subscriber
    low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    low_cmd_pub.Init()

    low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    low_state_sub.Init(low_state_handler, 10)

    print("DDS initialized. Starting control thread...")
    
    control_thread = Thread(target=control_loop, args=(low_cmd_pub,))

    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Stopping...")
        g_stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    control_thread.start()
    control_thread.join()

    print("Program finished.")


if __name__ == "__main__":
    main() 