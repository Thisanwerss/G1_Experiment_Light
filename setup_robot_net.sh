#!/bin/bash

# =================================================================
#  Robot Network Setup Script
# =================================================================
#
# This script automates the configuration of the network interface
# required for DDS communication with the robot.
#
# It performs the following actions:
# 1. Checks for the specified network interface.
# 2. Assigns a static IP address if not already set.
# 3. Ensures the interface is active (UP).
#
# It requires 'sudo' privileges to modify network settings.
#
# -----------------------------------------------------------------

set -e # Exit immediately if a command exits with a non-zero status.

INTERFACE="enp7s0"
IP_ADDRESS="192.168.123.222"
CIDR="24"
FULL_IP="${IP_ADDRESS}/${CIDR}"

# --- Environment Setup ---
# Get the absolute path of the script's directory to reliably source files
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Sourcing environment..."

# Source ROS2 if available
if [ -f "/opt/ros/humble/setup.bash" ]; then
    echo "   - Sourcing ROS2 Humble: /opt/ros/humble/setup.bash"
    source /opt/ros/humble/setup.bash
else
    echo "   - Warning: ROS2 setup not found."
fi

# Source local ROS2 workspace if it exists
if [ -f "${SCRIPT_DIR}/install/setup.bash" ]; then
    echo "   - Sourcing local ROS2 workspace: ${SCRIPT_DIR}/install/setup.bash"
    source "${SCRIPT_DIR}/install/setup.bash"
else
    echo "   - Warning: Local ROS2 workspace overlay not found."
fi
echo "------------------------------------------------------"

echo "🤖 Starting Robot Network Configuration..."
echo "=========================================="
echo "Target Interface: ${INTERFACE}"
echo "Target IP:        ${FULL_IP}"
echo ""

# --- 1. Check if the interface exists ---
if ! ip link show "$INTERFACE" > /dev/null 2>&1; then
    echo "❌ Error: Network interface '${INTERFACE}' not found."
    echo "   Please verify your network card name. Use 'ip link show' to list all available interfaces."
    exit 1
fi
echo "✅ Interface '${INTERFACE}' found."

# --- 2. Check if the IP address is already configured ---
if ip addr show "$INTERFACE" | grep -q "inet ${FULL_IP}"; then
    echo "✅ IP address '${FULL_IP}' is already configured."
else
    echo "🔧 IP address not set. Attempting to assign '${FULL_IP}'..."
    # Use sudo to add the IP address.
    sudo ip addr add "${FULL_IP}" dev "${INTERFACE}"
    echo "   Successfully assigned IP address."
fi

# --- 3. Check if the interface is UP ---
# We check the operational state in /sys/class/net/<interface>/operstate
if [[ "$(cat /sys/class/net/${INTERFACE}/operstate)" == "up" ]]; then
    echo "✅ Interface '${INTERFACE}' is already UP."
else
    echo "🔧 Interface is down. Attempting to bring it UP..."
    # Use sudo to bring the interface up.
    sudo ip link set "${INTERFACE}" up
    echo "   Successfully brought interface UP."
fi

echo ""
echo "🎉 Network setup complete! You should now be able to ping the robot at 192.168.123.161."
echo "==========================================" 