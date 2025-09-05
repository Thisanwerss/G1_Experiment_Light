#!/bin/bash
# ==============================================================================
#  G1 Robot & Vicon - Unified Network Setup & Diagnostics Script
# ==============================================================================
#
# This script performs a complete setup and self-check for running the G1 robot
# with Vicon motion capture. It MUST be run with sudo.
#
#
# Workflow:
# 1.  Sets a single, non-conflicting static IP for the network interface.
# 2.  Starts the Vicon ROS2 bridge in the background.
# 3.  Performs a series of checks:
#     - Pings the robot to verify basic network connectivity.
#     - Subscribes to the Vicon ROS2 topic to verify data flow.
#     - Subscribes to the robot's DDS state topic to verify control communication.
# 4.  Reports the success or failure of each step.
# 5.  Cleans up background processes on exit.
#
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
INTERFACE="enp7s0"
STATIC_IP="192.168.123.222/24"
ROBOT_IP="192.168.123.161"
VICON_ROS_TOPIC="/vicon/G1/G1"

# --- Environment Setup ---
# Source ROS2 and Python environments to make tools available
# Get the absolute path of the script's directory to reliably source files
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "STEP 0: Sourcing environment from script location: ${SCRIPT_DIR}"

# Source ROS2 if available
if [ -f "/opt/ros/humble/setup.bash" ]; then
    echo "   - Sourcing ROS2 Humble: /opt/ros/humble/setup.bash"
    source /opt/ros/humble/setup.bash
else
    echo "   - Warning: ROS2 setup not found. 'ros2' commands may fail."
fi

# Source local ROS2 workspace if it exists
if [ -f "${SCRIPT_DIR}/install/setup.bash" ]; then
    echo "   - Sourcing local ROS2 workspace: ${SCRIPT_DIR}/install/setup.bash"
    source "${SCRIPT_DIR}/install/setup.bash"
else
    echo "   - Warning: Local ROS2 workspace overlay not found."
fi

# Activate local Python virtual environment if it exists
# We will not source the venv directly as it's unreliable with sudo.
# Instead, we will call the python executable from within the venv directly.
echo "------------------------------------------------------"

# --- Main Script ---

# 1. Check for root privileges
if [ "$EUID" -ne 0 ]; then
  echo "❌ Error: This script must be run with sudo."
  echo "   Please run as: sudo $0"
  exit 1
fi

echo "🚀 Starting G1 Robot & Vicon Setup and Diagnostics..."
echo "======================================================"

# 2. Configure Network Interface
echo "STEP 1: Configuring network interface '${INTERFACE}'..."

if ! ip link show "$INTERFACE" > /dev/null 2>&1; then
    echo "❌ Error: Network interface '${INTERFACE}' not found."
    echo "   Please check your network interface name using 'ip a'."
    exit 1
fi

echo "   - Flushing existing IP configuration..."
ip addr flush dev "${INTERFACE}" || echo "   - (Warning: Failed to flush, proceeding anyway)"

echo "   - Assigning static IP address: ${STATIC_IP}..."
ip addr add "${STATIC_IP}" dev "${INTERFACE}"

echo "   - Activating interface..."
ip link set "${INTERFACE}" up

echo "   - Waiting 2s for network link to establish..."
sleep 2

if [[ "$(cat /sys/class/net/${INTERFACE}/operstate)" != "up" ]]; then
    echo "⚠️  Warning: Network link for '${INTERFACE}' is not active (state is DOWN)."
    echo "   Please ensure the network cable is properly connected to the robot/switch."
fi

echo "✅ Network configured successfully. Current IP:"
ip -4 -br addr show "${INTERFACE}"
echo "------------------------------------------------------"


# 3. Start Vicon ROS2 Bridge in the background
echo "STEP 2: Starting Vicon ROS2 node in the background..."
ros2 launch vicon_receiver client.launch.py &
ROS_PID=$!

# Setup a trap to automatically kill the background process on script exit
trap "echo '   - Shutting down Vicon ROS2 node (PID: $ROS_PID)...'; kill $ROS_PID 2>/dev/null" EXIT

echo "   - Vicon node started with PID: $ROS_PID. Waiting for initialization..."
sleep 5 # Give the node a few seconds to start up and connect
echo "------------------------------------------------------"

# 4. Perform Self-Checks
echo "STEP 3: Performing diagnostic self-checks..."
ALL_CHECKS_PASSED=true

# Check 3.1: Vicon ROS Topic
echo -n "   - Checking Vicon ROS2 Topic (${VICON_ROS_TOPIC})... "
# Use a timeout to prevent the script from getting stuck if the topic isn't publishing
if timeout 5s ros2 topic echo --once "$VICON_ROS_TOPIC" &> /dev/null; then
     echo "✅ SUCCESS"
else
     echo "❌ FAILED (No message received within 5s)"
     ALL_CHECKS_PASSED=false
fi

# Check 3.2: Robot DDS Communication (This replaces the unreliable ping check)
PYTHON_EXEC="python3" # Default to system python
if [ -f "${SCRIPT_DIR}/.venv/bin/python3" ]; then
    PYTHON_EXEC="${SCRIPT_DIR}/.venv/bin/python3"
    echo -n "   - Checking Robot Connectivity (via DDS using .venv python)... "
else
    echo -n "   - Checking Robot Connectivity (via DDS using system python)... "
fi

# Use an embedded Python script to check for DDS messages
if "$PYTHON_EXEC" - "$INTERFACE" <<'EOF'
import sys
import time
import threading
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# Global flag to indicate message reception
message_received = False
event = threading.Event()

def low_state_handler(msg: LowState_):
    global message_received
    message_received = True
    event.set()

def main():
    if len(sys.argv) < 2:
        print("Python Error: Network interface not provided.")
        sys.exit(1)
    
    channel = sys.argv[1]
    domain_id = 0 if channel != "lo" else 1

    try:
        ChannelFactoryInitialize(domain_id, channel)
        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        subscriber.Init(low_state_handler, 10)
        
        # Wait for a message for up to 3 seconds
        event.wait(timeout=3.0)
        
        if message_received:
            sys.exit(0) # Success
        else:
            sys.exit(1) # Failure
            
    except Exception as e:
        # print(f"Python DDS Check Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF
then
    echo "✅ SUCCESS"
else
    echo "❌ FAILED (No DDS messages received within 3s)"
    ALL_CHECKS_PASSED=false
fi

echo "------------------------------------------------------"

# 5. Final Report
echo "STEP 4: Final Report & Node Execution"
if $ALL_CHECKS_PASSED; then
    echo "🎉 All checks passed! The system is ready."
    echo "   Vicon ROS2 node is running (PID: $ROS_PID)."
    echo "   This terminal will now block to keep the node alive."
    echo "   Press Ctrl+C to shut down the Vicon node and exit."
    echo "======================================================"
    wait $ROS_PID
else
    echo "⚠️ Some checks failed. Please review the output above to diagnose the issue."
    echo "   Common issues:"
    echo "   - Robot/Vicon not powered on or not connected to the network."
    echo "   - Incorrect network interface name ('${INTERFACE}')."
    echo "   - Vicon software is not running or not tracking the object."
    echo "   Shutting down the Vicon node..."
    echo "======================================================"
    # Script will exit, and the EXIT trap will automatically kill the ROS node.
fi 