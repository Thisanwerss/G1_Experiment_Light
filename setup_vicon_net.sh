#!/bin/bash
# ==============================================================================
# Configure Network for Vicon Communication
# ==============================================================================
#
# This script sets a static IP address for a specified network interface
# to ensure it's on the same subnet as the Vicon host.
#
# MUST BE RUN WITH SUDO:
#   sudo ./setup_vicon_net.sh
#
# ==============================================================================

# --- Configuration ---
INTERFACE="enp7s0"
STATIC_IP="192.168.123.10/24"
VICON_HOST="192.168.123.100"

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

# --- Main Script ---

# Check for root privileges
if [ "$EUID" -ne 0 ]; then
  echo "❌ Error: This script must be run with sudo."
  echo "   Please run as: sudo $0"
  exit 1
fi

echo "🚀 Configuring network interface '${INTERFACE}' for Vicon..."

# Check if the interface exists
if ! ip link show "$INTERFACE" > /dev/null 2>&1; then
    echo "❌ Error: Network interface '${INTERFACE}' not found."
    echo "   Please check your network interface name using 'ip a' or 'ifconfig'."
    exit 1
fi

echo "1. Flushing existing IP configuration on ${INTERFACE}..."
ip addr flush dev "${INTERFACE}"
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Failed to flush interface. This may not be an issue if it has no config."
fi

echo "2. Assigning static IP address: ${STATIC_IP} to ${INTERFACE}..."
ip addr add "${STATIC_IP}" dev "${INTERFACE}"
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to set static IP address."
    exit 1
fi

# Bring the interface up, just in case it was down
ip link set "${INTERFACE}" up

echo "✅ Successfully set IP for ${INTERFACE}."
echo ""
echo "📊 Current configuration for ${INTERFACE}:"
ip -4 -br addr show "${INTERFACE}"
echo ""

echo "🔍 Verifying ARP resolution for Vicon host (${VICON_HOST})..."
# Perform a quick ARP lookup to prime the table and check connectivity
ip neigh show "${VICON_HOST}" dev "${INTERFACE}"
echo "   (It's okay if this is empty initially, it will populate on first use)"

echo ""
echo "🎉 Network configuration complete. You can now run your Vicon client application." 