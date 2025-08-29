#!/usr/bin/env python3
"""
G1 Robot State Visualizer
=========================

This tool provides a graphical user interface to monitor the state of the G1 robot in real-time.
It allows users to:
1. Connect to the robot via DDS.
2. Monitor all joint positions and velocities in real-time.
3. Monitor IMU (Inertial Measurement Unit) data: quaternion, gyroscope, accelerometer.
4. It does NOT send any commands to the robot, making it safe for monitoring.

Dependencies:
- PySide6
- numpy
- unitree_sdk2py

Usage:
1. Ensure PySide6 is installed: `pip install PySide6`
2. Ensure ATARI_NMPC root directory is in PYTHONPATH.
3. Run: `python g1_state_visualizer.py`
"""

import sys
import time
import numpy as np
import signal
import os

# PySide6 imports for the UI
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QScrollArea, QGroupBox, QFormLayout, QComboBox, 
    QFrame, QGridLayout, QPushButton, QProgressBar
)
from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread
from PySide6.QtGui import QFont

# DDS and robot specific imports
try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelSubscriber
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        LowState_, HandState_, BmsState_, LowCmd_, HandCmd_
    )
    # Project-specific imports
    from sdk_controller.robots.G1 import (
        MUJOCO_JOINT_NAMES, BODY_MUJOCO_TO_DDS, NUM_ACTIVE_BODY_JOINTS,
        LEFT_HAND_MUJOCO_TO_DDS, RIGHT_HAND_MUJOCO_TO_DDS
    )
except ImportError as e:
    print(f"❌ Module import failed: {e}", file=sys.stderr)
    print("   Please ensure ATARI_NMPC root directory is added to your PYTHONPATH.", file=sys.stderr)
    print("   Example: export PYTHONPATH=$PYTHONPATH:/path/to/ATARI_NMPC", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
RAD_TO_DEG = 180.0 / np.pi
NUM_HAND_DDS_MOTORS = 7

# --- DDS Communication Backend ---

class DDSReceiver(QObject):
    """Handles all DDS state subscription in a background thread."""
    connectionStatusChanged = Signal(bool)
    newStateReceived = Signal(dict)
    commandReceived = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.low_state_sub = None
        self.hand_state_sub = None
        self.bms_state_sub = None
        self.low_cmd_sub = None
        self.hand_cmd_sub = None
        
        self.last_low_state = None
        self.last_hand_state = None
        self.last_bms_state = None
        self.is_connected = False
        
        # Store the latest full state
        self.current_q = {} # Keyed by mujoco_name
        self.current_v = {} # Keyed by mujoco_name
        self.imu_data = {}  # Store IMU data

    @Slot(str)
    def start(self, channel: str):
        print(f"🚀 正在启动DDS监听器，通道: {channel}...")
        try:
            domain_id = 0 if channel != "lo" else 1
            ChannelFactoryInitialize(domain_id, channel)

            # State Subscribers
            self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
            self.hand_state_sub = ChannelSubscriber("rt/handstate", HandState_)
            self.bms_state_sub = ChannelSubscriber("rt/lf/bmsstate", BmsState_)
            self.low_state_sub.Init(self._low_state_handler, 10)
            self.hand_state_sub.Init(self._hand_state_handler, 10)
            self.bms_state_sub.Init(self._bms_state_handler, 10)

            # Command Subscribers
            self.low_cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
            self.hand_cmd_sub = ChannelSubscriber("rt/handcmd", HandCmd_)
            self.low_cmd_sub.Init(self._low_cmd_handler, 10)
            self.hand_cmd_sub.Init(self._hand_cmd_handler, 10)
            
            self.running = True
            print("✅ DDS监听器启动成功。")
        except Exception as e:
            print(f"❌ DDS初始化失败: {e}", file=sys.stderr)
            self.connectionStatusChanged.emit(False)

    def _low_state_handler(self, msg: LowState_):
        self.last_low_state = msg
        if not self.is_connected:
            self.is_connected = True
            self.connectionStatusChanged.emit(True)
        self._process_state()

    def _hand_state_handler(self, msg: HandState_):
        self.last_hand_state = msg
        self._process_state()

    def _bms_state_handler(self, msg: BmsState_):
        self.last_bms_state = msg
        self._process_state()

    def _low_cmd_handler(self, msg: LowCmd_):
        """Handles incoming LowCmd messages to monitor external commands."""
        if not self.running: return
        cmd_data = {'body': {}}
        for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
            if mj_idx < NUM_ACTIVE_BODY_JOINTS:
                mj_name = MUJOCO_JOINT_NAMES[mj_idx]
                motor_cmd = msg.motor_cmd[dds_idx]
                cmd_data['body'][mj_name] = {
                    'q': motor_cmd.q, 'kp': motor_cmd.kp,
                    'dq': motor_cmd.dq, 'kd': motor_cmd.kd,
                    'tau': motor_cmd.tau
                }
        self.commandReceived.emit(cmd_data)

    def _hand_cmd_handler(self, msg: HandCmd_):
        """Handles incoming HandCmd messages."""
        if not self.running: return
        cmd_data = {'left_hand': {}, 'right_hand': {}}
        # The motor_cmd array in HandCmd has 14 elements. Left is 0-6, right is 7-13.
        for mj_idx, dds_idx in LEFT_HAND_MUJOCO_TO_DDS.items():
            mj_name = MUJOCO_JOINT_NAMES[mj_idx]
            motor_cmd = msg.motor_cmd[dds_idx]
            cmd_data['left_hand'][mj_name] = {
                'q': motor_cmd.q, 'kp': motor_cmd.kp,
                'dq': motor_cmd.dq, 'kd': motor_cmd.kd,
                'tau': motor_cmd.tau
            }
        for mj_idx, dds_idx in RIGHT_HAND_MUJOCO_TO_DDS.items():
             cmd_array_idx = dds_idx + NUM_HAND_DDS_MOTORS
             mj_name = MUJOCO_JOINT_NAMES[mj_idx]
             motor_cmd = msg.motor_cmd[cmd_array_idx]
             cmd_data['right_hand'][mj_name] = {
                'q': motor_cmd.q, 'kp': motor_cmd.kp,
                'dq': motor_cmd.dq, 'kd': motor_cmd.kd,
                'tau': motor_cmd.tau
             }
        self.commandReceived.emit(cmd_data)

    def _process_state(self):
        """Process the latest state messages and emit the result."""
        if not self.running:
            return
        
        state_data = {'body': {}, 'left_hand': {}, 'right_hand': {}, 'imu': {}, 'bms': {}}
        
        # Process body state and IMU
        if self.last_low_state:
            for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
                if mj_idx < NUM_ACTIVE_BODY_JOINTS:
                    mj_name = MUJOCO_JOINT_NAMES[mj_idx]
                    motor = self.last_low_state.motor_state[dds_idx]
                    state_data['body'][mj_name] = {'q': motor.q, 'dq': motor.dq}
            
            # Extract IMU data
            imu = self.last_low_state.imu_state
            state_data['imu'] = {
                'quaternion': imu.quaternion,
                'gyroscope': imu.gyroscope,
                'accelerometer': imu.accelerometer
            }

        # Process hand state
        if self.last_hand_state:
            for mj_idx, dds_idx in LEFT_HAND_MUJOCO_TO_DDS.items():
                mj_name = MUJOCO_JOINT_NAMES[mj_idx]
                motor = self.last_hand_state.motor_state[dds_idx]
                state_data['left_hand'][mj_name] = {'q': motor.q, 'dq': motor.dq}

            for mj_idx, dds_idx in RIGHT_HAND_MUJOCO_TO_DDS.items():
                mj_name = MUJOCO_JOINT_NAMES[mj_idx]
                motor = self.last_hand_state.motor_state[dds_idx + NUM_HAND_DDS_MOTORS]
                state_data['right_hand'][mj_name] = {'q': motor.q, 'dq': motor.dq}
        
        # Process battery state
        if self.last_bms_state:
            bms = self.last_bms_state
            # Assuming 'soc' is the State of Charge field (percentage 0-100)
            state_data['bms'] = {'soc': bms.soc}

        self.newStateReceived.emit(state_data)

    @Slot()
    def stop(self):
        self.running = False
        print("🛑 Stopping DDS listener...")

# --- UI Components ---

class JointStateWidget(QGroupBox):
    """A widget to display a single joint's state."""
    def __init__(self, title, joint_info):
        super().__init__(title)
        self.joint_info = joint_info
        
        layout = QGridLayout(self)
        self.dds_id_label = QLabel(f"DDS Idx: <b>{joint_info['dds_id']}</b>")
        self.q_label = QLabel("Pos: <b>--.-°</b>")
        self.dq_label = QLabel("Vel: <b>--.- rad/s</b>")
        
        layout.addWidget(self.dds_id_label, 0, 0)
        layout.addWidget(self.q_label, 0, 1)
        layout.addWidget(self.dq_label, 0, 2)

        # Labels for command state
        self.cmd_q_label = QLabel("Cmd Pos: --.-°")
        self.cmd_kp_label = QLabel("Cmd Kp: --.-")
        
        font = self.cmd_q_label.font()
        font.setPointSize(font.pointSize() - 1)
        self.cmd_q_label.setFont(font)
        self.cmd_kp_label.setFont(font)
        
        layout.addWidget(self.cmd_q_label, 1, 0, 1, 2)
        layout.addWidget(self.cmd_kp_label, 1, 2, 1, 1)


    def update_state(self, q, dq):
        self.q_label.setText(f"Pos: <b>{q * RAD_TO_DEG:.1f}°</b>")
        self.dq_label.setText(f"Vel: <b>{dq:.2f} rad/s</b>")

    def update_command_state(self, cmd: dict):
        self.cmd_q_label.setText(f"Cmd Pos: {cmd['q'] * RAD_TO_DEG:.1f}°")
        self.cmd_kp_label.setText(f"Cmd Kp: {cmd['kp']:.1f}")


class BatteryStateWidget(QGroupBox):
    """A widget to display battery state."""
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
        
        # Change color based on battery level
        if soc < 20:
            # Red
            self.soc_bar.setStyleSheet("QProgressBar::chunk { background-color: #d9534f; border-radius: 4px; }")
        elif soc < 50:
            # Yellow
            self.soc_bar.setStyleSheet("QProgressBar::chunk { background-color: #f0ad4e; border-radius: 4px; }")
        else:
            # Green
            self.soc_bar.setStyleSheet("QProgressBar::chunk { background-color: #5cb85c; border-radius: 4px; }")

class IMUStateWidget(QGroupBox):
    """A widget to display IMU state."""
    def __init__(self, title):
        super().__init__(title)
        self.setFont(QFont("Arial", 11, QFont.Bold))
        
        layout = QFormLayout(self)
        self.quat_label = QLabel("<b>[-, -, -, -]</b>")
        self.gyro_label = QLabel("<b>[-, -, -]</b> rad/s")
        self.accel_label = QLabel("<b>[-, -, -]</b> m/s²")
        
        layout.addRow("Quaternion (w,x,y,z):", self.quat_label)
        layout.addRow("Gyroscope (x,y,z):", self.gyro_label)
        layout.addRow("Accelerometer (x,y,z):", self.accel_label)

    def update_state(self, imu_data):
        quat = imu_data.get('quaternion', [0,0,0,0])
        gyro = imu_data.get('gyroscope', [0,0,0])
        accel = imu_data.get('accelerometer', [0,0,0])
        
        self.quat_label.setText(f"<b>[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]</b>")
        self.gyro_label.setText(f"<b>[{gyro[0]:.3f}, {gyro[1]:.3f}, {gyro[2]:.3f}]</b> rad/s")
        self.accel_label.setText(f"<b>[{accel[0]:.3f}, {accel[1]:.3f}, {accel[2]:.3f}]</b> m/s²")

# --- Main Application Window ---

class G1VisualizerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("G1 Robot State Visualizer")
        self.setGeometry(100, 100, 800, 900)

        # Setup DDS backend
        self.dds_thread = QThread()
        self.dds_receiver = DDSReceiver()
        self.dds_receiver.moveToThread(self.dds_thread)

        # UI elements
        self.joint_widgets = {} # Keyed by mujoco_name
        self._init_ui()
        self._create_joint_widgets()
        
        # Connect signals and slots
        self._connect_signals()

        self.dds_thread.start()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Top Control Panel ---
        top_panel = QFrame()
        top_panel.setFrameShape(QFrame.StyledPanel)
        top_layout = QHBoxLayout(top_panel)

        self.connection_status_label = QLabel("🔴 Disconnected")
        self.connection_status_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["enp7s0", "lo", "eth0", "wlan0"])
        self.channel_combo.setCurrentText("enp7s0")
        self.channel_combo.setEditable(True)
        self.connect_button = QPushButton("Connect")

        top_layout.addWidget(self.connection_status_label)
        top_layout.addWidget(QLabel("Network Interface:"))
        top_layout.addWidget(self.channel_combo)
        top_layout.addWidget(self.connect_button)
        top_layout.addStretch()
        main_layout.addWidget(top_panel)
        
        # --- Status Panel (IMU and Battery) ---
        status_panel = QFrame()
        status_layout = QHBoxLayout(status_panel)

        self.imu_widget = IMUStateWidget("IMU State")
        self.battery_widget = BatteryStateWidget("Battery State")
        
        status_layout.addWidget(self.imu_widget)
        status_layout.addWidget(self.battery_widget)
        main_layout.addWidget(status_panel)

        # --- Scroll Area for Joints ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        self.scroll_content_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        scroll_area.setWidget(self.scroll_content_widget)

    def _create_joint_widgets(self):
        """Dynamically create display widgets for all joints."""
        
        # Group definitions
        groups = {
            "Left Leg": (0, 6), "Right Leg": (6, 12), "Waist": (12, 13),
            "Left Arm": (13, 20), "Right Arm": (20, 27),
            "Left Hand": (27, 34), "Right Hand": (34, 41)
        }
        
        for group_name, (start_idx, end_idx) in groups.items():
            group_box = QGroupBox(group_name)
            group_box.setFont(QFont("Arial", 11, QFont.Bold))
            group_layout = QVBoxLayout(group_box)
            
            is_hand_group = "Hand" in group_name
            
            for mj_idx in range(start_idx, end_idx):
                mujoco_name = MUJOCO_JOINT_NAMES[mj_idx]
                
                if is_hand_group:
                    mapping = LEFT_HAND_MUJOCO_TO_DDS if "Left" in group_name else RIGHT_HAND_MUJOCO_TO_DDS
                    if mj_idx not in mapping: continue
                    dds_id = mapping[mj_idx]
                else: # Body group
                    if mj_idx not in BODY_MUJOCO_TO_DDS: continue
                    dds_id = BODY_MUJOCO_TO_DDS[mj_idx]

                joint_info = {
                    'mujoco_name': mujoco_name,
                    'dds_id': dds_id,
                }
                
                widget = JointStateWidget(mujoco_name, joint_info)
                group_layout.addWidget(widget)
                self.joint_widgets[mujoco_name] = widget
            
            self.scroll_layout.addWidget(group_box)

    def _connect_signals(self):
        # DDS Receiver -> UI
        self.dds_receiver.connectionStatusChanged.connect(self.on_connection_status_changed)
        self.dds_receiver.newStateReceived.connect(self.on_new_state_received)
        self.dds_receiver.commandReceived.connect(self.on_command_received)
        
        # UI -> DDS Receiver
        self.connect_button.clicked.connect(
            lambda: self.dds_receiver.start(self.channel_combo.currentText())
        )
        
        # Window closing event
        self.destroyed.connect(self.dds_thread.quit)
        self.destroyed.connect(self.dds_thread.wait)

    @Slot(bool)
    def on_connection_status_changed(self, is_connected):
        if is_connected:
            self.connection_status_label.setText("🟢 Connected")
            self.connect_button.setEnabled(False)
            self.channel_combo.setEnabled(False)
        else:
            self.connection_status_label.setText("🔴 Disconnected")
            self.connect_button.setEnabled(True)
            self.channel_combo.setEnabled(True)

    @Slot(dict)
    def on_new_state_received(self, state_data):
        # Update joints
        all_joint_states = {**state_data['body'], **state_data['left_hand'], **state_data['right_hand']}
        for name, widget in self.joint_widgets.items():
            if name in all_joint_states:
                widget.update_state(all_joint_states[name]['q'], all_joint_states[name]['dq'])
        
        # Update IMU
        if 'imu' in state_data:
            self.imu_widget.update_state(state_data['imu'])

        # Update Battery
        if 'bms' in state_data and state_data['bms']:
            self.battery_widget.update_state(state_data['bms'])

    @Slot(dict)
    def on_command_received(self, cmd_data):
        all_cmds = {**cmd_data.get('body',{}), **cmd_data.get('left_hand',{}), **cmd_data.get('right_hand',{})}
        for name, widget in self.joint_widgets.items():
            if name in all_cmds:
                widget.update_command_state(all_cmds[name])

    def closeEvent(self, event):
        self.dds_receiver.stop()
        self.dds_thread.quit()
        self.dds_thread.wait()
        super().closeEvent(event)

# --- Main Execution ---

def main():
    # Graceful shutdown on Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    window = G1VisualizerUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 