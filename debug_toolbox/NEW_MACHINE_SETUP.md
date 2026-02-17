# New Machine Setup Guide — G1 Experiment Light

This guide walks you through setting up a **new PC** to communicate with the Unitree G1 robot and run the debug/control tools in this project.

> **TL;DR** — The only thing that *must* change per-machine is the **network interface name**. Everything else (IPs, DDS domain, topics) is fixed.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Install Python Dependencies](#3-install-python-dependencies)
4. [Identify Your Network Interface](#4-identify-your-network-interface)
5. [Update the Configuration](#5-update-the-configuration)
6. [Configure the Network](#6-configure-the-network)
7. [Verify Everything Works](#7-verify-everything-works)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Ubuntu 20.04+ (or any Linux with systemd networking) |
| **Python** | >= 3.8 |
| **pip** | `sudo apt install python3-pip` |
| **jq** | `sudo apt install jq` (used by `setup_robot_net.sh`) |
| **Network** | An Ethernet port connected to the G1 robot (direct or via switch) |

---

## 2. Clone the Repository

```bash
git clone <your-repo-url> G1_Experiment_Light
cd G1_Experiment_Light
```

---

## 3. Install Python Dependencies

The only required Python package is the **Unitree SDK2 Python** library, which is bundled in the repo under `external_deps/`.

```bash
# Option A: Install directly into system Python (simplest)
cd external_deps/unitree_sdk2_python
pip install -e .

# Option B: Use a virtual environment (recommended for clean installs)
cd /path/to/G1_Experiment_Light
python3 -m venv .venv
source .venv/bin/activate
cd external_deps/unitree_sdk2_python
pip install -e .
```

This will automatically install the sub-dependencies:
- `cyclonedds==0.10.2` (DDS communication middleware — **version must be 0.10.2**)
- `numpy`
- `opencv-python`

### Quick Verification

```bash
python3 -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('SDK OK')"
```

---

## 4. Identify Your Network Interface

This is the **most important step** and the primary thing that changes between machines.

### How to find your interface name

```bash
# Method 1: List all interfaces (recommended)
ls /sys/class/net/

# Method 2: Show detailed info with IP addresses
ip link show

# Method 3: Compact view with addresses
ip -br addr show
```

### Example output

```
lo               UNKNOWN        127.0.0.1/8
enp14s0          UP             192.168.123.222/24    <-- This is the Ethernet port
wlp15s0          UP             192.168.1.105/24      <-- This is WiFi
docker0          DOWN           172.17.0.1/16         <-- Ignore (Docker)
```

### How to identify the right interface

| Clue | What to look for |
|---|---|
| **Name pattern** | `enp*s*`, `eth*`, `eno*` are physical Ethernet ports |
| **Ignore these** | `lo` (loopback), `docker*`, `br-*`, `veth*`, `virbr*` (virtual), `wlp*`/`wlan*` (WiFi) |
| **Physical test** | Plug/unplug the Ethernet cable and run `ip link show` — the interface that toggles between `UP` and `DOWN` is the one |
| **Multiple Ethernet ports** | If your machine has 2+ Ethernet ports, use the one physically connected to the robot |

### Common interface name examples across machines

| Machine type | Typical name |
|---|---|
| Desktop with single NIC | `enp7s0`, `enp14s0`, `eno1` |
| Laptop with USB-Ethernet | `enx<mac>` (long hex name) |
| Older Linux / Embedded | `eth0`, `eth1` |
| NUC / Mini PC | `eno1`, `enp2s0` |

---

## 5. Update the Configuration

Edit the project's central config file:

```bash
# File: global_config.json (in project root)
nano global_config.json
```

Change the `network_interface` value to match your machine:

```json
{
  "network_interface": "enp14s0"
}
```

> **Note:** All Python scripts and `setup_robot_net.sh` read this field, so you only need to change it in one place.

---

## 6. Configure the Network

The robot communicates on the `192.168.123.x` subnet. Your PC needs a static IP on this subnet on the correct Ethernet interface.

### Option A: Use the project setup script (recommended)

```bash
sudo bash setup_robot_net.sh
```

This script reads `network_interface` from `global_config.json` and:
1. Flushes existing IPs on that interface
2. Assigns `192.168.123.222/24`
3. Brings the interface up
4. Tests DDS connectivity to the robot

### Option B: Manual setup

```bash
# Replace <IFACE> with your interface name, e.g. enp14s0
sudo ip addr flush dev <IFACE>
sudo ip addr add 192.168.123.222/24 dev <IFACE>
sudo ip link set <IFACE> up
```

### Fixed Network Parameters (do NOT change)

| Parameter | Value | Notes |
|---|---|---|
| PC static IP | `192.168.123.222/24` | Always this, on every machine |
| Robot IP | `192.168.123.161` | G1 robot default |
| Subnet | `192.168.123.0/24` | — |
| DDS Domain ID | `0` | Default for all scripts |

---

## 7. Verify Everything Works

### Step 1: Run the diagnostic script

```bash
# Basic check (no sudo needed)
bash debug_toolbox/check_environment.sh

# Full check (includes network validation)
sudo bash debug_toolbox/check_environment.sh
```

This will check:
- Python version and dependencies
- Network interfaces and their states
- Whether `global_config.json` points to a valid interface
- Robot reachability (ping)
- DDS communication (subscribes for 3 seconds)

### Step 2: Ping the robot

```bash
ping 192.168.123.161
```

### Step 3: Test a subscriber

```bash
# Uses the interface from global_config.json
python3 debug_toolbox/dex3_hand_state_sub.py

# Or specify interface explicitly
python3 debug_toolbox/dex3_hand_state_sub.py --iface enp14s0
```

---

## 8. Troubleshooting

### "Interface `enp7s0` not found"

The interface name from the old machine doesn't exist on yours. Follow [Section 4](#4-identify-your-network-interface) to find the correct name, then update `global_config.json`.

### "No DDS messages received"

1. **Check cable:** Ensure the Ethernet cable is plugged in and the link light is on.
2. **Check IP:** Run `ip addr show <IFACE>` — you should see `192.168.123.222/24`.
3. **Check robot:** Is the robot powered on?
4. **Check interface state:** `cat /sys/class/net/<IFACE>/operstate` should say `up`.
5. **Firewall:** DDS uses UDP multicast. Disable firewall temporarily: `sudo ufw disable`.

### "cyclonedds version mismatch"

The SDK requires **exactly** `cyclonedds==0.10.2`. If you have a different version:

```bash
pip install cyclonedds==0.10.2
```

### "ImportError: unitree_sdk2py"

You haven't installed the SDK yet. Go to [Section 3](#3-install-python-dependencies).

### Network config doesn't persist after reboot

The `ip addr add` command is temporary. After reboot you must either:
- Re-run `sudo bash setup_robot_net.sh`, or
- Set up a permanent Netplan / NetworkManager config (out of scope for this guide)

---

## Quick Checklist (copy-paste version)

```bash
# 1. Install deps
sudo apt install python3-pip jq -y
cd external_deps/unitree_sdk2_python && pip install -e . && cd ../..

# 2. Find your Ethernet interface
ls /sys/class/net/

# 3. Update config (replace YOUR_IFACE with the actual name)
# Edit global_config.json -> "network_interface": "YOUR_IFACE"

# 4. Setup network
sudo bash setup_robot_net.sh

# 5. Run diagnostics
bash debug_toolbox/check_environment.sh

# 6. Test
python3 debug_toolbox/dex3_hand_state_sub.py
```
