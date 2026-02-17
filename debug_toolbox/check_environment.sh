#!/bin/bash
# ==============================================================================
#  G1 Experiment Light — New Machine Environment Check & Diagnostics
# ==============================================================================
#
#  Run this script on a new machine BEFORE attempting to run any robot code.
#  It gathers all necessary information and checks every prerequisite.
#
#  Usage:
#    bash check_environment.sh            # Normal diagnostics (no sudo needed)
#    sudo bash check_environment.sh       # Full diagnostics (includes network config check)
#
# ==============================================================================

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

pass()  { echo -e "  ${GREEN}[PASS]${RESET}  $1"; }
fail()  { echo -e "  ${RED}[FAIL]${RESET}  $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
warn()  { echo -e "  ${YELLOW}[WARN]${RESET}  $1"; WARN_COUNT=$((WARN_COUNT + 1)); }
info()  { echo -e "  ${CYAN}[INFO]${RESET}  $1"; }
header(){ echo ""; echo -e "${BOLD}━━━ $1 ━━━${RESET}"; }

FAIL_COUNT=0
WARN_COUNT=0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )
CONFIG_FILE="${PROJECT_ROOT}/global_config.json"

echo ""
echo "========================================================"
echo "  G1 Experiment Light — Environment Diagnostics"
echo "  $(date)"
echo "  Machine: $(hostname)"
echo "========================================================"

# ==============================================================================
# SECTION 1: OS & Basic Info
# ==============================================================================
header "1. System Information"

info "Hostname     : $(hostname)"
info "OS           : $(uname -s -r -m)"

if [ -f /etc/os-release ]; then
    . /etc/os-release
    info "Distribution : ${PRETTY_NAME}"
fi

info "User         : $(whoami) (UID=$(id -u))"
info "Kernel       : $(uname -r)"

if [ "$EUID" -eq 0 ]; then
    info "Running as   : root (full diagnostics enabled)"
else
    warn "Not running as root. Some network checks will be limited. Run with sudo for full diagnostics."
fi

# ==============================================================================
# SECTION 2: Python Environment
# ==============================================================================
header "2. Python Environment"

# 2a. Python version
if command -v python3 &> /dev/null; then
    PY_VER=$(python3 --version 2>&1)
    PY_PATH=$(which python3)
    PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    info "python3      : ${PY_VER} (${PY_PATH})"
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 8 ]; then
        pass "Python >= 3.8 satisfied"
    else
        fail "Python >= 3.8 required, got ${PY_VER}"
    fi
else
    fail "python3 not found in PATH"
fi

# 2b. pip
if command -v pip3 &> /dev/null || python3 -m pip --version &> /dev/null; then
    PIP_VER=$(python3 -m pip --version 2>&1 | head -1)
    pass "pip available: ${PIP_VER}"
else
    fail "pip3 / python3 -m pip not available"
fi

# 2c. venv check
if [ -d "${PROJECT_ROOT}/.venv" ]; then
    info "Virtual env  : ${PROJECT_ROOT}/.venv (found)"
    PYTHON_EXEC="${PROJECT_ROOT}/.venv/bin/python3"
else
    info "Virtual env  : not found (using system python3)"
    PYTHON_EXEC="python3"
fi

# 2d. Key Python packages
header "3. Python Dependencies"

check_python_pkg() {
    local pkg_import=$1
    local pkg_label=$2
    if "$PYTHON_EXEC" -c "import ${pkg_import}" 2>/dev/null; then
        pass "${pkg_label} installed"
        return 0
    else
        fail "${pkg_label} NOT installed"
        return 1
    fi
}

check_python_pkg "unitree_sdk2py"           "unitree_sdk2py (Unitree SDK2 Python)"
SDK_OK=$?

check_python_pkg "cyclonedds"               "cyclonedds (DDS middleware)"
check_python_pkg "numpy"                    "numpy"

# Check cyclonedds version specifically
if "$PYTHON_EXEC" -c "import cyclonedds" 2>/dev/null; then
    CDDS_VER=$("$PYTHON_EXEC" -c "import cyclonedds; print(cyclonedds.__version__)" 2>/dev/null || echo "unknown")
    if [ "$CDDS_VER" = "0.10.2" ]; then
        pass "cyclonedds version is 0.10.2 (required)"
    else
        warn "cyclonedds version is ${CDDS_VER} (expected 0.10.2 — may cause issues)"
    fi
fi

# Check if unitree_sdk2py IDL messages are accessible
if [ "$SDK_OK" -eq 0 ]; then
    if "$PYTHON_EXEC" -c "from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_" 2>/dev/null; then
        pass "HandState_ IDL message importable"
    else
        fail "HandState_ IDL message NOT importable (SDK may be incomplete)"
    fi
    if "$PYTHON_EXEC" -c "from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_" 2>/dev/null; then
        pass "LowState_ IDL message importable"
    else
        fail "LowState_ IDL message NOT importable (SDK may be incomplete)"
    fi
fi

# ==============================================================================
# SECTION 4: Network Interfaces
# ==============================================================================
header "4. Network Interfaces (this is the most important section)"

echo ""
echo -e "  ${BOLD}All available network interfaces:${RESET}"
echo -e "  ─────────────────────────────────────────────────────────"
printf "  ${BOLD}%-18s %-10s %-22s %s${RESET}\n" "INTERFACE" "STATE" "IPv4 ADDRESS" "TYPE"
echo -e "  ─────────────────────────────────────────────────────────"

ETHERNET_IFACES=()

for iface in /sys/class/net/*; do
    IFACE_NAME=$(basename "$iface")
    
    # Skip loopback
    [ "$IFACE_NAME" = "lo" ] && continue

    # Get state
    STATE=$(cat "$iface/operstate" 2>/dev/null || echo "unknown")
    
    # Get type
    if [ -d "$iface/wireless" ]; then
        TYPE="WiFi"
    elif [[ "$IFACE_NAME" == docker* ]] || [[ "$IFACE_NAME" == br-* ]] || [[ "$IFACE_NAME" == veth* ]]; then
        TYPE="Virtual/Docker"
    elif [[ "$IFACE_NAME" == virbr* ]]; then
        TYPE="Virtual/Libvirt"
    else
        TYPE="Ethernet"
        ETHERNET_IFACES+=("$IFACE_NAME")
    fi
    
    # Get IPv4 address
    IPV4=$(ip -4 addr show "$IFACE_NAME" 2>/dev/null | grep -oP 'inet \K[\d./]+' | head -1)
    [ -z "$IPV4" ] && IPV4="(none)"
    
    # Color state
    if [ "$STATE" = "up" ]; then
        STATE_FMT="${GREEN}${STATE}${RESET}"
    elif [ "$STATE" = "down" ]; then
        STATE_FMT="${RED}${STATE}${RESET}"
    else
        STATE_FMT="${YELLOW}${STATE}${RESET}"
    fi
    
    printf "  %-18s %-22s %-22s %s\n" "$IFACE_NAME" "$(echo -e $STATE_FMT)" "$IPV4" "$TYPE"
done

echo -e "  ─────────────────────────────────────────────────────────"
echo ""

if [ ${#ETHERNET_IFACES[@]} -eq 0 ]; then
    fail "No Ethernet interfaces detected!"
else
    info "Ethernet interfaces found: ${ETHERNET_IFACES[*]}"
    echo ""
    echo -e "  ${BOLD}>> You need to identify which Ethernet interface connects to the G1 robot.${RESET}"
    echo -e "  ${BOLD}>> Typical names: enp*s*, eth*, eno*${RESET}"
fi

# ==============================================================================
# SECTION 5: global_config.json
# ==============================================================================
header "5. Project Configuration (global_config.json)"

if [ -f "$CONFIG_FILE" ]; then
    pass "Config file found: ${CONFIG_FILE}"
    
    if command -v jq &> /dev/null; then
        CONFIGURED_IFACE=$(jq -r '.network_interface' "$CONFIG_FILE" 2>/dev/null)
        info "Configured interface: '${CONFIGURED_IFACE}'"
        
        # Check if the configured interface actually exists
        if [ -d "/sys/class/net/${CONFIGURED_IFACE}" ]; then
            pass "Interface '${CONFIGURED_IFACE}' exists on this machine"
            
            IFACE_STATE=$(cat "/sys/class/net/${CONFIGURED_IFACE}/operstate" 2>/dev/null || echo "unknown")
            if [ "$IFACE_STATE" = "up" ]; then
                pass "Interface '${CONFIGURED_IFACE}' is UP"
            else
                warn "Interface '${CONFIGURED_IFACE}' state is '${IFACE_STATE}' (should be 'up')"
            fi
            
            IFACE_IP=$(ip -4 addr show "$CONFIGURED_IFACE" 2>/dev/null | grep -oP 'inet \K[\d./]+' | head -1)
            if [ -n "$IFACE_IP" ]; then
                info "Interface '${CONFIGURED_IFACE}' has IP: ${IFACE_IP}"
                # Check if it's in the robot subnet
                if [[ "$IFACE_IP" == 192.168.123.* ]]; then
                    pass "IP is in the robot subnet (192.168.123.x)"
                else
                    warn "IP '${IFACE_IP}' is NOT in the expected robot subnet 192.168.123.x"
                fi
            else
                warn "Interface '${CONFIGURED_IFACE}' has no IPv4 address assigned"
            fi
        else
            fail "Interface '${CONFIGURED_IFACE}' does NOT exist on this machine!"
            echo ""
            echo -e "  ${RED}${BOLD}>> ACTION REQUIRED: Update 'network_interface' in global_config.json${RESET}"
            echo -e "  ${RED}   Choose one of: ${ETHERNET_IFACES[*]}${RESET}"
            echo ""
        fi
    else
        warn "jq not installed — cannot parse config. Install with: sudo apt install jq"
        info "Raw config file content:"
        cat "$CONFIG_FILE" | while IFS= read -r line; do echo "    $line"; done
    fi
else
    fail "Config file NOT found at ${CONFIG_FILE}"
fi

# ==============================================================================
# SECTION 6: Robot Connectivity (ping)
# ==============================================================================
header "6. Robot Connectivity"

ROBOT_IP="192.168.123.161"
info "Robot IP (expected): ${ROBOT_IP}"

if ping -c 1 -W 2 "$ROBOT_IP" &> /dev/null; then
    pass "Robot is reachable via ping (${ROBOT_IP})"
else
    warn "Robot NOT reachable via ping (${ROBOT_IP}). This could mean:"
    echo "        - Robot is not powered on"
    echo "        - Network cable not connected"
    echo "        - This machine's IP is not in 192.168.123.x subnet"
    echo "        - You haven't run the network setup yet (sudo bash setup_robot_net.sh)"
fi

# ==============================================================================
# SECTION 7: DDS Communication Check
# ==============================================================================
header "7. DDS Communication Check"

if [ "$SDK_OK" -eq 0 ]; then
    if [ -d "/sys/class/net/${CONFIGURED_IFACE}" ]; then
        info "Attempting DDS subscription on interface '${CONFIGURED_IFACE}' for 3 seconds..."
        
        DDS_RESULT=$("$PYTHON_EXEC" -c "
import sys, time, threading
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
except ImportError as e:
    print(f'IMPORT_ERROR:{e}')
    sys.exit(2)

received = False
event = threading.Event()

def cb(msg):
    global received
    received = True
    event.set()

try:
    ChannelFactoryInitialize(0, '${CONFIGURED_IFACE}', False)
except TypeError:
    ChannelFactoryInitialize(0, '${CONFIGURED_IFACE}')

sub = ChannelSubscriber('rt/lowstate', LowState_)
sub.Init(cb, 10)
event.wait(timeout=3.0)

if received:
    print('DDS_OK')
else:
    print('DDS_TIMEOUT')
" 2>&1)
        
        if [[ "$DDS_RESULT" == *"DDS_OK"* ]]; then
            pass "DDS communication OK — received LowState_ messages from robot"
        elif [[ "$DDS_RESULT" == *"DDS_TIMEOUT"* ]]; then
            warn "No DDS messages received within 3s. Robot may not be publishing or network not ready."
        elif [[ "$DDS_RESULT" == *"IMPORT_ERROR"* ]]; then
            fail "SDK import error during DDS check: ${DDS_RESULT}"
        else
            warn "DDS check returned unexpected output: ${DDS_RESULT}"
        fi
    else
        warn "Skipping DDS check — configured interface '${CONFIGURED_IFACE}' not found on this machine"
    fi
else
    warn "Skipping DDS check — unitree_sdk2py not installed"
fi

# ==============================================================================
# SECTION 8: Additional Tools
# ==============================================================================
header "8. Additional Tools"

if command -v jq &> /dev/null; then
    pass "jq installed (needed by setup_robot_net.sh)"
else
    warn "jq NOT installed. Install with: sudo apt install jq"
fi

if command -v ros2 &> /dev/null; then
    info "ROS2 found: $(ros2 --version 2>/dev/null || echo 'unknown version')"
else
    info "ROS2 not found (optional, only needed for Vicon)"
fi

if command -v git &> /dev/null; then
    pass "git installed"
else
    warn "git not installed"
fi

# ==============================================================================
# SECTION 9: Quick-Reference Commands
# ==============================================================================
header "9. Quick-Reference Commands"

echo ""
echo "  If you need to fix anything, here are the useful commands:"
echo ""
echo "  # List all network interfaces"
echo "  ip link show"
echo ""
echo "  # Show interface details with IP addresses"
echo "  ip -4 addr show"
echo ""
echo "  # Check a specific interface"
echo "  ip addr show <INTERFACE_NAME>"
echo ""
echo "  # Assign a static IP to connect to the robot"
echo "  sudo ip addr add 192.168.123.222/24 dev <INTERFACE_NAME>"
echo "  sudo ip link set <INTERFACE_NAME> up"
echo ""
echo "  # Or use the project's setup script (reads interface from global_config.json)"
echo "  sudo bash ${PROJECT_ROOT}/setup_robot_net.sh"
echo ""
echo "  # Update the network interface in config"
echo "  # Edit: ${CONFIG_FILE}"
echo "  # Change \"network_interface\": \"...\" to your interface name"
echo ""
echo "  # Install Python dependencies"
echo "  cd ${PROJECT_ROOT}/external_deps/unitree_sdk2_python && pip install -e ."
echo ""
echo "  # Test hand state subscriber"
echo "  python3 ${SCRIPT_DIR}/dex3_hand_state_sub.py --iface <INTERFACE_NAME>"
echo ""

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
header "SUMMARY"

echo ""
if [ "$FAIL_COUNT" -eq 0 ] && [ "$WARN_COUNT" -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}All checks passed! Environment looks ready.${RESET}"
elif [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "  ${YELLOW}${BOLD}${WARN_COUNT} warning(s), 0 failures. Review warnings above.${RESET}"
else
    echo -e "  ${RED}${BOLD}${FAIL_COUNT} failure(s), ${WARN_COUNT} warning(s). Please fix the failures above.${RESET}"
fi
echo ""
echo "========================================================"
