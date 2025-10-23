#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G1 Robot State Visualizer (English Version)
============================================

This tool provides an advanced graphical user interface to monitor the complete
low-level state of the G1 robot in real-time. It is a pure monitoring tool and
does NOT send any commands to the robot.

Features:
1.  Connects to the robot via DDS to subscribe to state topics.
2.  Decodes and displays comprehensive state information from:
    - LowState: Body joint states, IMU, tick, crc, etc.
    - HandState: Detailed states for left and right hands, including joint states,
      IMU, power, and pressure sensors.
    - BmsState: Battery management system status.
    - Odometry and Secondary IMU.
3.  Provides detailed motor state for each joint: q, dq, ddq, torque, temperature,
    voltage, and error flags.
4.  Real-time error monitoring and diagnosis:
    - A prominent status panel shows device and motor errors.
    - Decodes error flags from the 'motorstate' field.
    - Provides a dictionary of possible causes for each error code.
5.  Organized UI with tabs for Body, Left Hand, Right Hand, and other sensors.

Dependencies:
- PySide6
- numpy
- unitree_sdk2py

Usage:
1. Ensure PySide6 is installed: `pip install PySide6`
2. Ensure ATARI_NMPC root directory is in PYTHONPATH.
3. Run: `python g1_state_visualizer_en.py`
"""

import sys
import time
import numpy as np
import signal
import os
import subprocess
import socket
import json

# PySide6 imports for the UI
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QScrollArea, QGroupBox, QFormLayout, QComboBox,
        QFrame, QGridLayout, QPushButton, QProgressBar, QTextEdit, QTabWidget,
        QMessageBox
    )
    from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread, QTimer
    from PySide6.QtGui import QFont, QColor
    PYSIDE6_AVAILABLE = True
except ImportError as e:
    PYSIDE6_AVAILABLE = False
    print("=" * 80)
    print("❌ DEPENDENCY ERROR: PySide6 is not installed")
    print("=" * 80)
    print("\nPySide6 is required for the GUI interface.")
    print("\n📦 To install PySide6, run:")
    print("    pip install PySide6")
    print("\n   or if using conda:")
    print("    conda install -c conda-forge pyside6")
    print("\n💡 If you're in a virtual environment, make sure it's activated first.")
    print("=" * 80)
    sys.exit(1)

# DDS and robot specific imports
try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelSubscriber
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        LowState_, HandState_, BmsState_, LowCmd_, HandCmd_, IMUState_
    )
    UNITREE_SDK_AVAILABLE = True
except ImportError as e:
    UNITREE_SDK_AVAILABLE = False
    print("=" * 80)
    print("❌ DEPENDENCY ERROR: unitree_sdk2py is not installed or not found")
    print("=" * 80)
    print(f"\nError details: {e}")
    print("\n📦 To install unitree_sdk2py:")
    print("    1. Clone the repository:")
    print("       git clone https://github.com/unitreerobotics/unitree_sdk2_python.git")
    print("    2. Install it:")
    print("       cd unitree_sdk2_python")
    print("       pip install -e .")
    print("\n💡 Or if it's already in external_deps/:")
    print("    cd external_deps/unitree_sdk2_python")
    print("    pip install -e .")
    print("=" * 80)
    sys.exit(1)

# Project-specific imports
try:
    from sdk_controller.robots.G1 import (
        MUJOCO_JOINT_NAMES, BODY_MUJOCO_TO_DDS, NUM_ACTIVE_BODY_JOINTS,
        LEFT_HAND_MUJOCO_TO_DDS, RIGHT_HAND_MUJOCO_TO_DDS
    )
    PROJECT_IMPORTS_AVAILABLE = True
except ImportError as e:
    PROJECT_IMPORTS_AVAILABLE = False
    print("=" * 80)
    print("❌ PROJECT IMPORT ERROR: Cannot import from sdk_controller.robots.G1")
    print("=" * 80)
    print(f"\nError details: {e}")
    print("\n🔧 Possible solutions:")
    print("    1. Add ATARI_NMPC root directory to your PYTHONPATH:")
    print(f"       export PYTHONPATH=$PYTHONPATH:{os.path.dirname(os.path.abspath(__file__))}")
    print("\n    2. Or run from the correct directory:")
    print(f"       cd {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
    print("       python ATARI_NMPC/g1_state_visualizer_en.py")
    print("\n    3. Check if sdk_controller/robots/G1.py exists:")
    expected_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sdk_controller", "robots", "G1.py")
    print(f"       Expected location: {expected_path}")
    print(f"       File exists: {os.path.exists(expected_path)}")
    print("=" * 80)
    sys.exit(1)

# --- Constants ---
RAD_TO_DEG = 180.0 / np.pi
NUM_HAND_DDS_MOTORS = 7
NUM_BODY_DDS_MOTORS = 35
NUM_HAND_PRESSURE_SENSORS = 12
CONNECTION_TIMEOUT_MS = 10000  # 10 seconds timeout for initial connection

# --- Helper Functions ---
def get_available_network_interfaces():
    """Get list of available network interfaces on the system."""
    interfaces = []
    try:
        # Try using 'ip' command (Linux)
        result = subprocess.run(['ip', 'link', 'show'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if ':' in line and not line.startswith(' '):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        iface = parts[1].strip()
                        if iface and iface != 'lo':
                            interfaces.append(iface)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback: try reading /sys/class/net/
        try:
            net_path = '/sys/class/net'
            if os.path.exists(net_path):
                interfaces = [d for d in os.listdir(net_path) if d != 'lo']
        except:
            pass
    
    # Always include common interfaces
    common = ['enp7s0', 'eth0', 'wlan0', 'lo']
    for iface in common:
        if iface not in interfaces:
            interfaces.append(iface)
    
    return interfaces

def check_network_interface_status(interface):
    """Check if a network interface is up and has an IP address."""
    try:
        result = subprocess.run(['ip', 'addr', 'show', interface],
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            output = result.stdout
            is_up = 'state UP' in output or 'UP' in output.split('\n')[0]
            has_ip = 'inet ' in output
            
            # Extract IP address if available
            ip_addr = None
            for line in output.split('\n'):
                if 'inet ' in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        ip_addr = parts[1].split('/')[0]
                        break
            
            return {
                'exists': True,
                'is_up': is_up,
                'has_ip': has_ip,
                'ip_address': ip_addr
            }
    except:
        pass
    
    return {
        'exists': False,
        'is_up': False,
        'has_ip': False,
        'ip_address': None
    }

def get_network_diagnostics():
    """Get comprehensive network diagnostics for troubleshooting."""
    diagnostics = []
    diagnostics.append("=== Network Interface Diagnostics ===\n")
    
    interfaces = get_available_network_interfaces()
    diagnostics.append(f"Available interfaces: {', '.join(interfaces)}\n")
    
    for iface in interfaces[:5]:  # Check first 5 interfaces
        status = check_network_interface_status(iface)
        if status['exists']:
            state = "UP" if status['is_up'] else "DOWN"
            ip_info = f" (IP: {status['ip_address']})" if status['has_ip'] else " (No IP)"
            diagnostics.append(f"  • {iface}: {state}{ip_info}")
        else:
            diagnostics.append(f"  • {iface}: Not found")
    
    diagnostics.append("\n💡 Common Robot Network Configurations:")
    diagnostics.append("  • G1 Robot typically uses: 192.168.123.x network")
    diagnostics.append("  • For real robot: Use the interface connected to robot (e.g., enp7s0, eth0)")
    diagnostics.append("  • For simulation: Use 'lo' (loopback)")
    
    return "\n".join(diagnostics)

# --- Error Dictionaries ---
DEVICE_ERROR_CODES = {
    0x01: "Upper control command error",
    0x02: "Lower feedback data timeout",
    0x04: "IMU feedback data timeout",
    0x08: "Motor feedback data timeout",
    0x10: "Battery feedback data timeout",
    0x20: "Unitree physical remote control feedback data timeout",
    0x40: "Battery model error",
    0x80: "Soft start error",
    0x100: "Motor status error",
    0x200: "Motor overcurrent protection, low limit protection triggered",
    0x400: "Motor undervoltage protection, high limit protection triggered",
    0x800: "Motor overcurrent protection, high limit protection triggered",
    0x1000: "Soft emergency stop button pressed",
    0x2000: "SN error",
    0x4000: "Upper layer model error",
    0x8000: "Lower layer model error",
    0x10000: "USB device error",
    0x40000: "Hip IMU data timeout",
    0x80000: "Mainboard detects battery undervoltage protection error",
    0x100000: "Mainboard detects motor undervoltage protection error",
}

MOTOR_ERROR_CODES = {
    0x01: "Overcurrent",
    0x02: "Transient overvoltage",
    0x04: "Continuous overvoltage",
    0x08: "Transient undervoltage",
    0x10: "Chip overheating",
    0x20: "MOS overheating/overcooling",
    0x40: "MOS temperature abnormal",
    0x80: "Housing overheating/overcooling",
    0x100: "Housing temperature abnormal",
    0x200: "Winding overheating",
    0x400: "Rotor encoder 1 error",
    0x800: "Rotor encoder 2 error",
    0x1000: "Output encoder error",
    0x2000: "Calibration/BOOT data error",
    0x4000: "Abnormal reset",
    0x8000: "Motor locked, master control authentication error",
    0x10000: "Chip verification error",
    0x20000: "Calibration mode warning",
    0x40000: "Communication checksum error",
    0x80000: "Driver version too low",
    0x40000000: "Motor side judgment, PC connection timeout",
    0x80000000: "PC side judgment, motor disconnection timeout",
}


# --- DDS Communication Backend ---

class DDSReceiver(QObject):
    """Handles all DDS state subscription in a background thread."""
    connectionStatusChanged = Signal(bool)
    newStateReceived = Signal(dict)
    connectionError = Signal(str, str)  # Signal(error_type, error_message)

    def __init__(self):
        super().__init__()
        self.running = False
        self.subscribers = {}
        self.last_states = {
            'low': None, 'left_hand': None, 'right_hand': None,
            'bms': None, 'odom': None, 'secondary_imu': None
        }
        self.is_connected = False
        self.current_channel = None
        self.connection_start_time = None

    @Slot(str)
    def start(self, channel: str):
        self.current_channel = channel
        self.connection_start_time = time.time()
        
        print(f"🚀 Starting DDS listener on channel: {channel}...")
        print(f"   Current network interface: {channel}")
        
        # Check network interface status
        status = check_network_interface_status(channel)
        if not status['exists']:
            error_msg = f"Network interface '{channel}' does not exist on this system."
            print(f"❌ {error_msg}", file=sys.stderr)
            self.connectionError.emit("interface_not_found", error_msg)
            return
        
        if not status['is_up']:
            error_msg = f"Network interface '{channel}' is DOWN. Please bring it up first."
            print(f"⚠️  {error_msg}", file=sys.stderr)
            if channel != 'lo':  # Don't fail for loopback
                self.connectionError.emit("interface_down", error_msg)
                return
        
        if not status['has_ip'] and channel != 'lo':
            warning_msg = f"Network interface '{channel}' has no IP address assigned."
            print(f"⚠️  {warning_msg}", file=sys.stderr)
            print(f"   This might be OK for DDS multicast, but could cause issues.")
        
        if status['ip_address']:
            print(f"   Interface IP: {status['ip_address']}")
        
        try:
            domain_id = 0 if channel != "lo" else 1
            print(f"   Using DDS Domain ID: {domain_id}")
            ChannelFactoryInitialize(domain_id, channel)

            # State Subscribers
            self.subscribers['low'] = ChannelSubscriber("rt/lowstate", LowState_)
            self.subscribers['left_hand'] = ChannelSubscriber("rt/dex3/left/state", HandState_)
            self.subscribers['right_hand'] = ChannelSubscriber("rt/dex3/right/state", HandState_)
            self.subscribers['bms'] = ChannelSubscriber("rt/lf/bmsstate", BmsState_)
            self.subscribers['odom'] = ChannelSubscriber("rt/odommodestate", IMUState_)
            self.subscribers['secondary_imu'] = ChannelSubscriber("rt/secondary_imu", IMUState_)

            self.subscribers['low'].Init(lambda msg: self._state_handler('low', msg))
            self.subscribers['left_hand'].Init(lambda msg: self._state_handler('left_hand', msg))
            self.subscribers['right_hand'].Init(lambda msg: self._state_handler('right_hand', msg))
            self.subscribers['bms'].Init(lambda msg: self._state_handler('bms', msg))
            self.subscribers['odom'].Init(lambda msg: self._state_handler('odom', msg))
            self.subscribers['secondary_imu'].Init(lambda msg: self._state_handler('secondary_imu', msg))

            self.running = True
            print("✅ DDS listener started successfully.")
            print("⏳ Waiting for robot data... (this may take a few seconds)")
            
        except Exception as e:
            error_msg = f"DDS initialization failed: {e}"
            print(f"❌ {error_msg}", file=sys.stderr)
            print("\n🔧 Troubleshooting:")
            print("   1. Check if the robot is powered on and connected")
            print("   2. Verify network connectivity: ping 192.168.123.10 (or robot IP)")
            print("   3. Check if DDS is properly configured")
            print("   4. Try a different network interface")
            self.connectionError.emit("dds_init_failed", error_msg)

    def _state_handler(self, key, msg):
        self.last_states[key] = msg
        if not self.is_connected:
            self.is_connected = True
            self.connectionStatusChanged.emit(True)
        self._process_state()

    def _process_state(self):
        """Process the latest state messages and emit the result."""
        if not self.running:
            return

        state_data = {
            'low_state': {}, 'body_motors': [{} for _ in range(NUM_BODY_DDS_MOTORS)],
            'left_hand_state': {}, 'right_hand_state': {}, 'bms': {},
            'odom': {}, 'secondary_imu': {}
        }

        # Process LowState
        if self.last_states['low']:
            low = self.last_states['low']
            state_data['low_state'] = {
                'version': low.version, 'mode_pr': low.mode_pr,
                'mode_machine': low.mode_machine, 'tick': low.tick,
                'imu': self._extract_imu_data(low.imu_state),
                'crc': low.crc
            }
            for i in range(NUM_BODY_DDS_MOTORS):
                state_data['body_motors'][i] = self._extract_motor_data(low.motor_state[i])

        # Process Left Hand State
        if self.last_states['left_hand']:
            left = self.last_states['left_hand']
            state_data['left_hand_state'] = {
                'imu': self._extract_imu_data(left.imu_state),
                'power_v': left.power_v, 'power_a': left.power_a,
                'motors': [self._extract_motor_data(m) for m in left.motor_state],
                'pressure': self._extract_pressure_data(left.press_sensor_state)
            }

        # Process Right Hand State
        if self.last_states['right_hand']:
            right = self.last_states['right_hand']
            state_data['right_hand_state'] = {
                'imu': self._extract_imu_data(right.imu_state),
                'power_v': right.power_v, 'power_a': right.power_a,
                'motors': [self._extract_motor_data(m) for m in right.motor_state],
                'pressure': self._extract_pressure_data(right.press_sensor_state)
            }

        # Process other states
        if self.last_states['bms']:
            state_data['bms'] = {'soc': self.last_states['bms'].soc}
        if self.last_states['odom']:
            state_data['odom'] = self._extract_imu_data(self.last_states['odom'])
        if self.last_states['secondary_imu']:
            state_data['secondary_imu'] = self._extract_imu_data(self.last_states['secondary_imu'])

        self.newStateReceived.emit(state_data)

    def _extract_imu_data(self, imu):
        return {
            'quaternion': imu.quaternion, 'gyroscope': imu.gyroscope,
            'accelerometer': imu.accelerometer, 'rpy': imu.rpy,
            'temperature': imu.temperature
        }

    def _extract_motor_data(self, motor):
        return {
            'mode': motor.mode, 'q': motor.q, 'dq': motor.dq, 'ddq': motor.ddq,
            'tau_est': motor.tau_est, 'temperature': motor.temperature,
            'vol': motor.vol, 'motorstate': motor.motorstate
        }

    def _extract_pressure_data(self, press_sensors):
        if not press_sensors: return []
        # Assuming one pressure sensor state object
        return {
            'pressure': press_sensors[0].pressure,
            'temperature': press_sensors[0].temperature
        }


    @Slot()
    def stop(self):
        self.running = False
        print("🛑 Stopping DDS listener...")

# --- UI Components ---
class MotorStateWidget(QGroupBox):
    """A widget to display a single motor's detailed state."""
    def __init__(self, title, dds_id):
        super().__init__(title)
        self.dds_id = dds_id

        layout = QGridLayout(self)
        self.dds_id_label = QLabel(f"DDS Idx: <b>{dds_id}</b>")
        self.q_label = QLabel("q: --.-°")
        self.dq_label = QLabel("dq: --.- rad/s")
        self.ddq_label = QLabel("ddq: --.- rad/s²")
        self.tau_label = QLabel("τ: --.- Nm")
        self.temp_label = QLabel("T: --/-- °C")
        self.vol_label = QLabel("V: --.- V")
        self.mode_label = QLabel("Mode: -")
        self.error_label = QLabel("Err: <b>-</b>")
        self.error_label.setStyleSheet("color: green")


        layout.addWidget(self.dds_id_label, 0, 0)
        layout.addWidget(self.mode_label, 0, 1)
        layout.addWidget(self.q_label, 1, 0)
        layout.addWidget(self.dq_label, 1, 1)
        layout.addWidget(self.ddq_label, 2, 0)
        layout.addWidget(self.tau_label, 2, 1)
        layout.addWidget(self.temp_label, 3, 0)
        layout.addWidget(self.vol_label, 3, 1)
        layout.addWidget(self.error_label, 0, 2, 1, 2)

    def update_state(self, motor_data):
        if not motor_data: return
        self.q_label.setText(f"q: {motor_data['q'] * RAD_TO_DEG:.1f}°")
        self.dq_label.setText(f"dq: {motor_data['dq']:.2f} rad/s")
        self.ddq_label.setText(f"ddq: {motor_data['ddq']:.2f} rad/s²")
        self.tau_label.setText(f"τ: {motor_data['tau_est']:.2f} Nm")
        self.temp_label.setText(f"T: {motor_data['temperature'][0]}/{motor_data['temperature'][1]}°C")
        self.vol_label.setText(f"V: {motor_data['vol']:.1f} V")
        self.mode_label.setText(f"Mode: {motor_data['mode']}")
        
        error_code = motor_data.get('motorstate', 0)
        if error_code != 0:
            self.error_label.setText(f"Err: <b>0x{error_code:X}</b>")
            self.error_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.error_label.setText("Err: <b>OK</b>")
            self.error_label.setStyleSheet("color: green;")

class BatteryStateWidget(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setFont(QFont("Arial", 11, QFont.Bold))
        layout = QHBoxLayout(self)
        self.soc_bar = QProgressBar()
        self.soc_bar.setRange(0, 100)
        self.soc_bar.setValue(0)
        self.soc_bar.setTextVisible(True)
        self.soc_bar.setFormat("%p%")
        layout.addWidget(QLabel("SoC:"))
        layout.addWidget(self.soc_bar)

    def update_state(self, bms_data):
        soc = bms_data.get('soc', 0)
        self.soc_bar.setValue(soc)
        if soc < 20:
            self.soc_bar.setStyleSheet("QProgressBar::chunk { background-color: #d9534f; border-radius: 4px; }")
        elif soc < 50:
            self.soc_bar.setStyleSheet("QProgressBar::chunk { background-color: #f0ad4e; border-radius: 4px; }")
        else:
            self.soc_bar.setStyleSheet("QProgressBar::chunk { background-color: #5cb85c; border-radius: 4px; }")

class IMUStateWidget(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setFont(QFont("Arial", 11, QFont.Bold))
        layout = QFormLayout(self)
        self.quat_label = QLabel("<b>[-, -, -, -]</b>")
        self.gyro_label = QLabel("<b>[-, -, -]</b> rad/s")
        self.accel_label = QLabel("<b>[-, -, -]</b> m/s²")
        self.rpy_label = QLabel("<b>[-, -, -]</b> °")
        self.temp_label = QLabel("<b>-</b> °C")
        layout.addRow("Quat (w,x,y,z):", self.quat_label)
        layout.addRow("Gyro (x,y,z):", self.gyro_label)
        layout.addRow("Accel (x,y,z):", self.accel_label)
        layout.addRow("RPY (r,p,y):", self.rpy_label)
        layout.addRow("Temp:", self.temp_label)

    def update_state(self, imu_data):
        if not imu_data: return
        quat = imu_data.get('quaternion', [0,0,0,0])
        gyro = imu_data.get('gyroscope', [0,0,0])
        accel = imu_data.get('accelerometer', [0,0,0])
        rpy = imu_data.get('rpy', [0,0,0])
        temp = imu_data.get('temperature', 0)
        self.quat_label.setText(f"<b>[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]</b>")
        self.gyro_label.setText(f"<b>[{gyro[0]:.3f}, {gyro[1]:.3f}, {gyro[2]:.3f}]</b> rad/s")
        self.accel_label.setText(f"<b>[{accel[0]:.3f}, {accel[1]:.3f}, {accel[2]:.3f}]</b> m/s²")
        self.rpy_label.setText(f"<b>[{rpy[0]*RAD_TO_DEG:.1f}, {rpy[1]*RAD_TO_DEG:.1f}, {rpy[2]*RAD_TO_DEG:.1f}]</b> °")
        self.temp_label.setText(f"<b>{temp}</b> °C")

class ErrorDisplayWidget(QGroupBox):
    """Widget to display system and motor error codes and their meanings."""
    def __init__(self, title):
        super().__init__(title)
        self.setFont(QFont("Arial", 11, QFont.Bold))
        self.setStyleSheet("QGroupBox { border: 2px solid gray; border-radius: 5px; margin-top: 1ex; } "
                           "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; "
                           "padding: 0 3px; }")

        layout = QHBoxLayout(self)
        self.status_label = QLabel("SYSTEM OK")
        font = QFont("Arial", 14, QFont.Bold)
        self.status_label.setFont(font)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.set_status(True) # Start with OK status

        self.details_box = QTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setFont(QFont("Monospace", 9))

        layout.addWidget(self.status_label, 1)
        layout.addWidget(self.details_box, 3)

    def set_status(self, is_ok):
        if is_ok:
            self.status_label.setText("✅ SYSTEM OK")
            self.setStyleSheet("background-color: #dff0d8;")
        else:
            self.status_label.setText("❌ SYSTEM ERROR")
            self.setStyleSheet("background-color: #f2dede;")

    def update_errors(self, body_motors, left_hand_motors, right_hand_motors, device_error_code=0):
        error_messages = []
        has_error = False

        # Check device-level errors (not available in LowState, placeholder)
        if device_error_code != 0:
            has_error = True
            for code, msg in DEVICE_ERROR_CODES.items():
                if device_error_code & code:
                    error_messages.append(f"[DEVICE] 0x{code:X}: {msg}")

        # Check body motor errors
        for i, motor_data in enumerate(body_motors):
            if motor_data and motor_data['motorstate'] != 0:
                has_error = True
                code = motor_data['motorstate']
                for err_code, msg in MOTOR_ERROR_CODES.items():
                    if code & err_code:
                        error_messages.append(f"[BODY MTR {i}] 0x{err_code:X}: {msg}")
        
        # Check hand motor errors
        for i, motor_data in enumerate(left_hand_motors):
             if motor_data and motor_data['motorstate'] != 0:
                has_error = True; code = motor_data['motorstate']
                for err, msg in MOTOR_ERROR_CODES.items():
                    if code & err: error_messages.append(f"[L.HAND MTR {i}] 0x{err:X}: {msg}")
        
        for i, motor_data in enumerate(right_hand_motors):
             if motor_data and motor_data['motorstate'] != 0:
                has_error = True; code = motor_data['motorstate']
                for err, msg in MOTOR_ERROR_CODES.items():
                    if code & err: error_messages.append(f"[R.HAND MTR {i}] 0x{err:X}: {msg}")

        self.set_status(not has_error)
        if has_error:
            self.details_box.setText("\n".join(error_messages))
        else:
            self.details_box.setText("No errors detected.")

# --- Global Config Loading ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# Navigate up one level to find the root directory where global_config.json is
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_FILE = os.path.join(ROOT_DIR, 'global_config.json')

def load_default_interface_from_config():
    """Reads the default network interface from the global config file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get('network_interface', 'enp7s0')
    except (FileNotFoundError, json.JSONDecodeError):
        return 'enp7s0'


# --- Main Application Window ---
class G1VisualizerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("G1 Robot State Visualizer (English)")
        self.setGeometry(100, 100, 1200, 950)

        # Setup DDS backend
        self.dds_thread = QThread()
        self.dds_receiver = DDSReceiver()
        self.dds_receiver.moveToThread(self.dds_thread)

        # UI elements
        self.body_motor_widgets = {} # Keyed by mujoco_name
        self.left_hand_motor_widgets = {}
        self.right_hand_motor_widgets = {}
        
        # Connection timeout timer
        self.connection_timeout_timer = QTimer()
        self.connection_timeout_timer.setSingleShot(True)
        self.connection_timeout_timer.timeout.connect(self.on_connection_timeout)

        self._init_ui()
        self._create_motor_widgets()
        self._connect_signals()
        self._populate_network_interfaces()

        self.dds_thread.start()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Top Control Panel ---
        top_panel = QFrame(); top_panel.setFrameShape(QFrame.StyledPanel)
        top_layout = QHBoxLayout(top_panel)
        self.connection_status_label = QLabel("🔴 Disconnected")
        self.connection_status_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.channel_combo = QComboBox()
        default_interface = load_default_interface_from_config()
        common_interfaces = [default_interface, "lo", "eth0", "wlan0"]
        unique_interfaces = list(dict.fromkeys(common_interfaces))
        self.channel_combo.addItems(unique_interfaces)
        self.channel_combo.setCurrentText(default_interface)
        
        self.connect_button = QPushButton("Connect DDS")
        self.connect_button.clicked.connect(self.toggle_connection)
        self.diagnose_button = QPushButton("🔍 Network Info")
        self.diagnose_button.clicked.connect(self.show_network_diagnostics)
        top_layout.addWidget(self.connection_status_label)
        top_layout.addWidget(QLabel("Network Interface:"))
        top_layout.addWidget(self.channel_combo)
        top_layout.addWidget(self.connect_button)
        top_layout.addWidget(self.diagnose_button)
        top_layout.addStretch()
        main_layout.addWidget(top_panel)

        # --- Error Panel ---
        self.error_widget = ErrorDisplayWidget("Error Status")
        main_layout.addWidget(self.error_widget)

        # --- Tabbed Interface ---
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self._create_body_tab()
        self._create_hand_tabs()
        self._create_other_sensors_tab()

    def _create_body_tab(self):
        body_tab = QWidget()
        body_layout = QVBoxLayout(body_tab)
        
        # General LowState Info
        low_state_box = QGroupBox("LowState Info")
        low_state_layout = QGridLayout(low_state_box)
        self.tick_label = QLabel("Tick: -")
        self.crc_label = QLabel("CRC: -")
        self.mode_pr_label = QLabel("Mode PR: -")
        self.mode_machine_label = QLabel("Mode Machine: -")
        low_state_layout.addWidget(self.tick_label, 0, 0)
        low_state_layout.addWidget(self.crc_label, 0, 1)
        low_state_layout.addWidget(self.mode_pr_label, 1, 0)
        low_state_layout.addWidget(self.mode_machine_label, 1, 1)
        body_layout.addWidget(low_state_box)

        self.body_imu_widget = IMUStateWidget("Body IMU State")
        body_layout.addWidget(self.body_imu_widget)

        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        body_layout.addWidget(scroll_area)
        
        self.body_scroll_content = QWidget()
        self.body_scroll_layout = QVBoxLayout(self.body_scroll_content)
        self.body_scroll_layout.setAlignment(Qt.AlignTop)
        scroll_area.setWidget(self.body_scroll_content)

        self.tabs.addTab(body_tab, "🤖 Body")

    def _create_hand_tabs(self):
        # Left Hand
        left_tab = QWidget()
        left_layout = QVBoxLayout(left_tab)
        self.left_hand_imu_widget = IMUStateWidget("Left Hand IMU")
        self.left_hand_power_widget = QGroupBox("Left Hand Power")
        power_layout_l = QFormLayout(self.left_hand_power_widget)
        self.left_power_v_label = QLabel("- V")
        self.left_power_a_label = QLabel("- A")
        power_layout_l.addRow("Voltage:", self.left_power_v_label)
        power_layout_l.addRow("Current:", self.left_power_a_label)

        self.left_hand_pressure_widget = QGroupBox("Left Hand Pressure Sensors")
        pressure_layout_l = QGridLayout(self.left_hand_pressure_widget)
        self.left_pressure_labels = [QLabel("P: - | T: -") for _ in range(NUM_HAND_PRESSURE_SENSORS)]
        for i, label in enumerate(self.left_pressure_labels):
            pressure_layout_l.addWidget(label, i // 4, i % 4)

        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        self.left_hand_scroll_content = QWidget()
        self.left_hand_scroll_layout = QVBoxLayout(self.left_hand_scroll_content)
        left_scroll.setWidget(self.left_hand_scroll_content)

        left_layout.addWidget(self.left_hand_imu_widget)
        left_layout.addWidget(self.left_hand_power_widget)
        left_layout.addWidget(self.left_hand_pressure_widget)
        left_layout.addWidget(left_scroll)
        self.tabs.addTab(left_tab, "🖐️ Left Hand")

        # Right Hand
        right_tab = QWidget(); right_layout = QVBoxLayout(right_tab)
        self.right_hand_imu_widget = IMUStateWidget("Right Hand IMU")
        self.right_hand_power_widget = QGroupBox("Right Hand Power")
        power_layout_r = QFormLayout(self.right_hand_power_widget)
        self.right_power_v_label = QLabel("- V")
        self.right_power_a_label = QLabel("- A")
        power_layout_r.addRow("Voltage:", self.right_power_v_label)
        power_layout_r.addRow("Current:", self.right_power_a_label)

        self.right_hand_pressure_widget = QGroupBox("Right Hand Pressure Sensors")
        pressure_layout_r = QGridLayout(self.right_hand_pressure_widget)
        self.right_pressure_labels = [QLabel("P: - | T: -") for _ in range(NUM_HAND_PRESSURE_SENSORS)]
        for i, label in enumerate(self.right_pressure_labels):
            pressure_layout_r.addWidget(label, i // 4, i % 4)

        right_scroll = QScrollArea(); right_scroll.setWidgetResizable(True)
        self.right_hand_scroll_content = QWidget()
        self.right_hand_scroll_layout = QVBoxLayout(self.right_hand_scroll_content)
        right_scroll.setWidget(self.right_hand_scroll_content)

        right_layout.addWidget(self.right_hand_imu_widget)
        right_layout.addWidget(self.right_hand_power_widget)
        right_layout.addWidget(self.right_hand_pressure_widget)
        right_layout.addWidget(right_scroll)
        self.tabs.addTab(right_tab, "🖐️ Right Hand")


    def _create_other_sensors_tab(self):
        other_tab = QWidget()
        other_layout = QVBoxLayout(other_tab)
        self.battery_widget = BatteryStateWidget("Battery State")
        self.odom_widget = IMUStateWidget("Odometry State")
        self.secondary_imu_widget = IMUStateWidget("Secondary IMU State")
        other_layout.addWidget(self.battery_widget)
        other_layout.addWidget(self.odom_widget)
        other_layout.addWidget(self.secondary_imu_widget)
        other_layout.addStretch()
        self.tabs.addTab(other_tab, "🔋 Sensors")

    def _create_motor_widgets(self):
        # Body
        groups = {
            "Left Leg": (0, 6), "Right Leg": (6, 12), "Waist": (12, 13),
            "Left Arm": (13, 20), "Right Arm": (20, 27)
        }
        for group_name, (start_idx, end_idx) in groups.items():
            group_box = QGroupBox(group_name); group_box.setFont(QFont("Arial", 11, QFont.Bold))
            group_layout = QVBoxLayout(group_box)
            for mj_idx in range(start_idx, end_idx):
                if mj_idx not in BODY_MUJOCO_TO_DDS: continue
                mujoco_name = MUJOCO_JOINT_NAMES[mj_idx]
                dds_id = BODY_MUJOCO_TO_DDS[mj_idx]
                widget = MotorStateWidget(mujoco_name, dds_id)
                group_layout.addWidget(widget)
                self.body_motor_widgets[mujoco_name] = widget
            self.body_scroll_layout.addWidget(group_box)
        
        # Hands
        for mj_idx, dds_idx in LEFT_HAND_MUJOCO_TO_DDS.items():
            name = MUJOCO_JOINT_NAMES[mj_idx]
            widget = MotorStateWidget(name, dds_idx)
            self.left_hand_motor_widgets[name] = widget
            self.left_hand_scroll_layout.addWidget(widget)

        for mj_idx, dds_idx in RIGHT_HAND_MUJOCO_TO_DDS.items():
            name = MUJOCO_JOINT_NAMES[mj_idx]
            widget = MotorStateWidget(name, dds_idx)
            self.right_hand_motor_widgets[name] = widget
            self.right_hand_scroll_layout.addWidget(widget)


    def _connect_signals(self):
        self.dds_receiver.connectionStatusChanged.connect(self.on_connection_status_changed)
        self.dds_receiver.newStateReceived.connect(self.on_new_state_received)
        self.dds_receiver.connectionError.connect(self.on_connection_error)
        self.connect_button.clicked.connect(self.on_connect_clicked)
        self.destroyed.connect(self.dds_thread.quit)
        self.destroyed.connect(self.dds_thread.wait)
    
    def _populate_network_interfaces(self):
        """Populate the network interface dropdown with available interfaces."""
        interfaces = get_available_network_interfaces()
        self.channel_combo.clear()
        self.channel_combo.addItems(interfaces)
        # Set default to first non-lo interface or 'lo' if none found
        default_idx = 0
        for i, iface in enumerate(interfaces):
            if iface != 'lo':
                default_idx = i
                break
        self.channel_combo.setCurrentIndex(default_idx)
    
    def on_connect_clicked(self):
        """Handle connect button click."""
        selected_channel = self.channel_combo.currentText()
        self.connection_status_label.setText("🟡 Connecting...")
        self.connect_button.setEnabled(False)
        self.channel_combo.setEnabled(False)
        
        # Start connection timeout timer
        self.connection_timeout_timer.start(CONNECTION_TIMEOUT_MS)
        
        # Start DDS receiver
        self.dds_receiver.start(selected_channel)
    
    def on_connection_timeout(self):
        """Handle connection timeout."""
        if not self.dds_receiver.is_connected:
            self.connection_status_label.setText("🔴 Connection Timeout")
            self.connect_button.setEnabled(True)
            self.channel_combo.setEnabled(True)
            
            current_channel = self.channel_combo.currentText()
            status = check_network_interface_status(current_channel)
            
            error_msg = f"Failed to receive data from robot after {CONNECTION_TIMEOUT_MS/1000:.0f} seconds.\n\n"
            error_msg += f"Current network interface: {current_channel}\n"
            
            if status['ip_address']:
                error_msg += f"Interface IP: {status['ip_address']}\n"
            
            error_msg += "\n🔧 Troubleshooting Steps:\n\n"
            error_msg += "1. Check Robot Connection:\n"
            error_msg += "   • Is the robot powered on?\n"
            error_msg += "   • Is the robot in the correct mode?\n"
            error_msg += "   • Is your computer connected to the robot's network?\n\n"
            
            error_msg += "2. Verify Network Connectivity:\n"
            error_msg += "   • Try: ping 192.168.123.10 (robot IP)\n"
            error_msg += "   • Check if you can see robot network traffic\n\n"
            
            error_msg += "3. Try Different Network Interface:\n"
            error_msg += "   • Click 'Network Info' to see available interfaces\n"
            error_msg += "   • Select the interface connected to robot\n"
            error_msg += "   • For simulation, use 'lo' (loopback)\n\n"
            
            error_msg += "4. Check DDS Configuration:\n"
            error_msg += "   • Ensure DDS domain ID matches robot configuration\n"
            error_msg += "   • Check firewall settings (may block DDS multicast)\n"
            
            QMessageBox.warning(self, "Connection Timeout", error_msg)
    
    def show_network_diagnostics(self):
        """Show detailed network diagnostics."""
        diagnostics = get_network_diagnostics()
        QMessageBox.information(self, "Network Diagnostics", diagnostics)
    
    @Slot(str, str)
    def on_connection_error(self, error_type, error_message):
        """Handle connection errors from DDS receiver."""
        self.connection_timeout_timer.stop()
        self.connection_status_label.setText("🔴 Connection Failed")
        self.connect_button.setEnabled(True)
        self.channel_combo.setEnabled(True)
        
        current_channel = self.channel_combo.currentText()
        
        if error_type == "interface_not_found":
            msg = f"❌ Network Interface Error\n\n"
            msg += f"The interface '{current_channel}' does not exist on your system.\n\n"
            msg += "📋 Available interfaces:\n"
            interfaces = get_available_network_interfaces()
            for iface in interfaces:
                status = check_network_interface_status(iface)
                if status['exists']:
                    state = "UP ✓" if status['is_up'] else "DOWN ✗"
                    ip = f" ({status['ip_address']})" if status['ip_address'] else ""
                    msg += f"  • {iface}: {state}{ip}\n"
            msg += "\n💡 Please select a valid interface from the dropdown."
            
        elif error_type == "interface_down":
            msg = f"❌ Network Interface Down\n\n"
            msg += f"The interface '{current_channel}' is currently DOWN.\n\n"
            msg += "🔧 To bring it up, try:\n"
            msg += f"  sudo ip link set {current_channel} up\n\n"
            msg += "Or check your network settings to enable this interface."
            
        elif error_type == "dds_init_failed":
            msg = f"❌ DDS Initialization Failed\n\n"
            msg += f"Error: {error_message}\n\n"
            msg += "🔧 Possible solutions:\n"
            msg += "1. Check if unitree_sdk2py is properly installed\n"
            msg += "2. Verify DDS is correctly configured\n"
            msg += "3. Try a different network interface\n"
            msg += "4. Check system logs for more details\n\n"
            msg += "💡 Click 'Network Info' for detailed diagnostics"
            
        else:
            msg = f"❌ Connection Error\n\n{error_message}"
        
        QMessageBox.critical(self, "Connection Error", msg)

    @Slot(bool)
    def on_connection_status_changed(self, is_connected):
        if is_connected:
            # Stop timeout timer on successful connection
            self.connection_timeout_timer.stop()
            self.connection_status_label.setText("🟢 Connected")
            self.connect_button.setEnabled(False)
            self.channel_combo.setEnabled(False)
            print(f"✅ Successfully connected and receiving data!")
        else:
            self.connection_status_label.setText("🔴 Disconnected")
            self.connect_button.setEnabled(True)
            self.channel_combo.setEnabled(True)

    @Slot(dict)
    def on_new_state_received(self, state_data):
        # Update Body
        if state_data['low_state']:
            low = state_data['low_state']
            self.tick_label.setText(f"Tick: {low['tick']}")
            self.crc_label.setText(f"CRC: {low['crc']}")
            self.mode_pr_label.setText(f"Mode PR: {low['mode_pr']}")
            self.mode_machine_label.setText(f"Mode Machine: {low['mode_machine']}")
            self.body_imu_widget.update_state(low['imu'])

        for name, widget in self.body_motor_widgets.items():
            dds_id = widget.dds_id
            if dds_id < len(state_data['body_motors']):
                widget.update_state(state_data['body_motors'][dds_id])

        # Update Left Hand
        if state_data['left_hand_state']:
            left = state_data['left_hand_state']
            self.left_hand_imu_widget.update_state(left['imu'])
            self.left_power_v_label.setText(f"{left['power_v']:.2f} V")
            self.left_power_a_label.setText(f"{left['power_a']:.2f} A")
            for name, widget in self.left_hand_motor_widgets.items():
                dds_id = widget.dds_id
                if dds_id < len(left['motors']):
                    widget.update_state(left['motors'][dds_id])
            if 'pressure' in left and left['pressure']:
                press = left['pressure']['pressure']
                temp = left['pressure']['temperature']
                for i in range(NUM_HAND_PRESSURE_SENSORS):
                    self.left_pressure_labels[i].setText(f"P:{press[i]:.1f}|T:{temp[i]:.1f}")


        # Update Right Hand
        if state_data['right_hand_state']:
            right = state_data['right_hand_state']
            self.right_hand_imu_widget.update_state(right['imu'])
            self.right_power_v_label.setText(f"{right['power_v']:.2f} V")
            self.right_power_a_label.setText(f"{right['power_a']:.2f} A")
            for name, widget in self.right_hand_motor_widgets.items():
                dds_id = widget.dds_id
                if dds_id < len(right['motors']):
                    widget.update_state(right['motors'][dds_id])
            if 'pressure' in right and right['pressure']:
                press = right['pressure']['pressure']
                temp = right['pressure']['temperature']
                for i in range(NUM_HAND_PRESSURE_SENSORS):
                    self.right_pressure_labels[i].setText(f"P:{press[i]:.1f}|T:{temp[i]:.1f}")

        # Update other sensors
        self.battery_widget.update_state(state_data['bms'])
        self.odom_widget.update_state(state_data['odom'])
        self.secondary_imu_widget.update_state(state_data['secondary_imu'])
        
        # Update errors
        left_motors = state_data['left_hand_state'].get('motors', [])
        right_motors = state_data['right_hand_state'].get('motors', [])
        self.error_widget.update_errors(state_data['body_motors'], left_motors, right_motors)


    def closeEvent(self, event):
        self.dds_receiver.stop()
        self.dds_thread.quit()
        self.dds_thread.wait()
        super().closeEvent(event)

# --- Main Execution ---
def print_startup_info():
    """Print helpful startup information."""
    print("=" * 80)
    print("🤖 G1 Robot State Visualizer (English Version)")
    print("=" * 80)
    print("\n📋 System Check:")
    print(f"   ✓ PySide6: Available")
    print(f"   ✓ unitree_sdk2py: Available")
    print(f"   ✓ Project imports: Available")
    print(f"   ✓ Python: {sys.version.split()[0]}")
    
    print("\n🌐 Network Interfaces:")
    interfaces = get_available_network_interfaces()
    for iface in interfaces[:5]:
        status = check_network_interface_status(iface)
        if status['exists']:
            state = "UP" if status['is_up'] else "DOWN"
            ip = f" ({status['ip_address']})" if status['ip_address'] else ""
            symbol = "✓" if status['is_up'] else "✗"
            print(f"   {symbol} {iface}: {state}{ip}")
    
    print("\n💡 Quick Start:")
    print("   1. Select the correct network interface from the dropdown")
    print("   2. Click 'Connect' button")
    print("   3. Wait for data (may take a few seconds)")
    print("   4. Use 'Network Info' button for diagnostics if connection fails")
    
    print("\n📖 Common Network Configurations:")
    print("   • Real Robot: Use ethernet interface connected to robot")
    print("     (e.g., enp7s0, eth0) - Robot typically at 192.168.123.10")
    print("   • Simulation: Use 'lo' (loopback interface)")
    
    print("\n🔧 Troubleshooting:")
    print("   • Connection timeout? Check robot power and network cable")
    print("   • Interface not found? Use 'Network Info' to see available interfaces")
    print("   • No data received? Verify robot is in the correct mode")
    print("   • DDS errors? Check firewall settings (may block multicast)")
    
    print("\n" + "=" * 80)
    print("Starting GUI...")
    print("=" * 80 + "\n")

def main():
    """Main entry point."""
    # Print startup information
    print_startup_info()
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # Create and run application
    app = QApplication(sys.argv)
    
    try:
        window = G1VisualizerUI()
        window.show()
        
        print("✅ GUI started successfully. Use the interface to connect to the robot.\n")
        
        sys.exit(app.exec())
    except Exception as e:
        print("=" * 80)
        print("❌ FATAL ERROR: Failed to start application")
        print("=" * 80)
        print(f"\nError: {e}")
        print("\n🔧 Please check:")
        print("   1. All dependencies are properly installed")
        print("   2. PYTHONPATH includes ATARI_NMPC directory")
        print("   3. You have necessary permissions")
        print("\n💡 For help, check the documentation or run with --help")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

