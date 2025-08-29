#!/usr/bin/env python3
"""
G1 Robot Single Joint Functionality Test UI
===========================================

This tool provides a graphical user interface for independently testing each joint's response on the G1 robot.
It allows users to:
1. Connect to the robot via DDS.
2. Monitor all joint positions and velocities in real-time.
3. Send small incremental PD control commands to individual joints.
4. Adjust PD control Kp gains.
5. View safety torque limits for each joint.
6. Use emergency stop functionality to put all joints in damping mode.

Dependencies:
- PySide6
- numpy
- mujoco
- unitree_sdk2py

Usage:
1. Ensure PySide6 is installed: `pip install PySide6`
2. Ensure ATARI_NMPC root directory is in PYTHONPATH.
3. Run: `python g1_joint_tester_ui.py`
"""

import sys
import time
import numpy as np
from functools import partial
import signal
import os

# PySide6 imports for the UI
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QDoubleSpinBox, QScrollArea, QGroupBox,
    QFormLayout, QComboBox, QFrame, QSizePolicy, QGridLayout
)
from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread, QTimer
from PySide6.QtGui import QPalette, QColor, QFont

# DDS and robot specific imports
try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        LowCmd_, LowState_, HandCmd_, HandState_
    )
    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__HandCmd_, unitree_hg_msg_dds__MotorCmd_
    )
    from unitree_sdk2py.utils.crc import CRC
    import mujoco
    # Project-specific imports
    from sdk_controller.robots.G1 import (
        MUJOCO_JOINT_NAMES, BODY_MUJOCO_TO_DDS, NUM_ACTIVE_BODY_JOINTS,
        LEFT_HAND_MUJOCO_TO_DDS, RIGHT_HAND_MUJOCO_TO_DDS, NUM_HAND_JOINTS, Kd
    )
    from sdk_controller.abstract_biped import HGSafetyLayer
except ImportError as e:
    print(f"❌ Module import failed: {e}", file=sys.stderr)
    print("   Please ensure ATARI_NMPC root directory is added to your PYTHONPATH.", file=sys.stderr)
    print("   Example: export PYTHONPATH=$PYTHONPATH:/path/to/ATARI_NMPC", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
ACTION_TIMEOUT_S = 20
UI_UPDATE_RATE_HZ = 30
NUM_BODY_DDS_MOTORS = 35
NUM_HAND_DDS_MOTORS = 7

# --- DDS Communication Backend ---

class DDSController(QObject):
    """Handles all DDS communication in a background thread."""
    connectionStatusChanged = Signal(bool)
    newStateReceived = Signal(dict)
    commandReceived = Signal(dict)
    actionFinished = Signal(str)
    actionStarted = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.crc = CRC()
        self.low_state_sub = None
        self.hand_state_sub = None
        self.low_cmd_sub = None
        self.hand_cmd_sub = None
        self.low_cmd_pub = None
        self.hand_cmd_pub = None
        
        self.last_low_state = None
        self.last_hand_state = None
        self.is_connected = False
        
        self.action_timer = None # Will be created in the correct thread

        # Store the latest full state
        self.current_q = {} # Keyed by mujoco_name
        self.current_v = {} # Keyed by mujoco_name

    @Slot(str)
    def start(self, channel: str):
        print(f"🚀 正在启动DDS控制器，通道: {channel}...")
        
        # Create timer in this thread to avoid QObject cross-thread errors
        self.action_timer = QTimer()
        self.action_timer.setSingleShot(True)
        self.action_timer.timeout.connect(self._on_action_timeout)

        try:
            domain_id = 0 if channel != "lo" else 1
            ChannelFactoryInitialize(domain_id, channel)

            # Publishers
            self.low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self.hand_cmd_pub = ChannelPublisher("rt/handcmd", HandCmd_)
            self.low_cmd_pub.Init()
            self.hand_cmd_pub.Init()

            # State Subscribers
            self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
            self.hand_state_sub = ChannelSubscriber("rt/handstate", HandState_)
            self.low_state_sub.Init(self._low_state_handler, 10)
            self.hand_state_sub.Init(self._hand_state_handler, 10)
            
            # Command Subscribers
            self.low_cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
            self.hand_cmd_sub = ChannelSubscriber("rt/handcmd", HandCmd_)
            self.low_cmd_sub.Init(self._low_cmd_handler, 10)
            self.hand_cmd_sub.Init(self._hand_cmd_handler, 10)

            self.running = True
            print("✅ DDS控制器启动成功。")
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
        
        state_data = {'body': {}, 'left_hand': {}, 'right_hand': {}}
        
        # Process body state
        if self.last_low_state:
            for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
                if mj_idx < NUM_ACTIVE_BODY_JOINTS:
                    mj_name = MUJOCO_JOINT_NAMES[mj_idx]
                    motor = self.last_low_state.motor_state[dds_idx]
                    self.current_q[mj_name] = motor.q
                    self.current_v[mj_name] = motor.dq
                    state_data['body'][mj_name] = {'q': motor.q, 'dq': motor.dq}

        # Process hand state
        if self.last_hand_state:
            for mj_idx, dds_idx in LEFT_HAND_MUJOCO_TO_DDS.items():
                mj_name = MUJOCO_JOINT_NAMES[mj_idx]
                motor = self.last_hand_state.motor_state[dds_idx]
                self.current_q[mj_name] = motor.q
                self.current_v[mj_name] = motor.dq
                state_data['left_hand'][mj_name] = {'q': motor.q, 'dq': motor.dq}

            for mj_idx, dds_idx in RIGHT_HAND_MUJOCO_TO_DDS.items():
                mj_name = MUJOCO_JOINT_NAMES[mj_idx]
                motor = self.last_hand_state.motor_state[dds_idx + NUM_HAND_DDS_MOTORS]
                self.current_q[mj_name] = motor.q
                self.current_v[mj_name] = motor.dq
                state_data['right_hand'][mj_name] = {'q': motor.q, 'dq': motor.dq}
        
        self.newStateReceived.emit(state_data)

    @Slot(dict)
    def send_test_command(self, cmd_info: dict):
        """Sends a PD command to a single joint."""
        if not self.is_connected or not self.running:
            print("⚠️ Not connected, cannot send command.")
            return

        target_mj_name = cmd_info['mujoco_name']
        print(f"🚀 Sending test command to joint: {target_mj_name}...")
        self.actionStarted.emit(target_mj_name)

        # --- Body Command ---
        low_cmd = unitree_hg_msg_dds__LowCmd_()
        for i in range(NUM_BODY_DDS_MOTORS):
            low_cmd.motor_cmd[i].mode = 1
            low_cmd.motor_cmd[i].kp = 0.0
            low_cmd.motor_cmd[i].kd = 2.0 # Default damping
            low_cmd.motor_cmd[i].q = 0.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        
        # Set all other body joints to hold current position
        for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
            mj_name = MUJOCO_JOINT_NAMES[mj_idx]
            if mj_name in self.current_q:
                low_cmd.motor_cmd[dds_idx].q = self.current_q[mj_name]
                low_cmd.motor_cmd[dds_idx].kp = cmd_info['kp_default'] # Use a default holding Kp

        # Set target joint command
        if cmd_info['type'] == 'body':
            dds_idx = cmd_info['dds_id']
            target_q = self.current_q.get(target_mj_name, 0) + cmd_info['delta_rad']
            low_cmd.motor_cmd[dds_idx].q = target_q
            low_cmd.motor_cmd[dds_idx].kp = cmd_info['kp']
            low_cmd.motor_cmd[dds_idx].kd = Kd # Use configured damping
            print(f"   - Body joint {target_mj_name} (DDS {dds_idx}): q_target={target_q:.3f} rad, Kp={cmd_info['kp']:.1f}")

        low_cmd.crc = self.crc.Crc(low_cmd)
        self.low_cmd_pub.Write(low_cmd)

        # --- Hand Command ---
        hand_cmd = unitree_hg_msg_dds__HandCmd_()
        hand_motor_cmds = []

        # Set all hand joints to hold current position
        for mj_idx in list(LEFT_HAND_MUJOCO_TO_DDS.keys()) + list(RIGHT_HAND_MUJOCO_TO_DDS.keys()):
            mj_name = MUJOCO_JOINT_NAMES[mj_idx]
            motor_cmd = unitree_hg_msg_dds__MotorCmd_()
            motor_cmd.mode = 1
            motor_cmd.q = self.current_q.get(mj_name, 0)
            motor_cmd.kp = cmd_info['kp_default'] # Default holding Kp
            motor_cmd.dq = 0.0
            motor_cmd.kd = 2.0
            motor_cmd.tau = 0.0
            hand_motor_cmds.append(motor_cmd)

        # Set target hand joint command
        if cmd_info['type'] in ['left_hand', 'right_hand']:
            dds_idx = cmd_info['dds_id']
            # Left hand is 0-6, right hand is 7-13 in the command array
            cmd_array_idx = dds_idx if cmd_info['type'] == 'left_hand' else dds_idx + NUM_HAND_DDS_MOTORS
            target_q = self.current_q.get(target_mj_name, 0) + cmd_info['delta_rad']
            hand_motor_cmds[cmd_array_idx].q = target_q
            hand_motor_cmds[cmd_array_idx].kp = cmd_info['kp']
            hand_motor_cmds[cmd_array_idx].kd = Kd # Use configured damping
            print(f"   - Hand joint {target_mj_name} (DDS {dds_idx}): q_target={target_q:.3f} rad, Kp={cmd_info['kp']:.1f}")
        
        hand_cmd.motor_cmd = hand_motor_cmds
        self.hand_cmd_pub.Write(hand_cmd)
        
        self.action_timer.setProperty("joint_name", target_mj_name)
        self.action_timer.start(ACTION_TIMEOUT_S * 1000)

    @Slot()
    def emergency_stop(self):
        """Send damping commands to all joints."""
        if not self.is_connected:
            return
        print("🛑 Emergency stop! Sending damping commands...")
        self.action_timer.stop()
        
        # Body damping
        low_cmd = unitree_hg_msg_dds__LowCmd_()
        for i in range(NUM_BODY_DDS_MOTORS):
            low_cmd.motor_cmd[i].mode = 1
            low_cmd.motor_cmd[i].q = 0.0
            low_cmd.motor_cmd[i].kp = 0.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].kd = 2.0  # Light damping
            low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.crc = self.crc.Crc(low_cmd)
        self.low_cmd_pub.Write(low_cmd)

        # Hand damping
        hand_cmd = unitree_hg_msg_dds__HandCmd_()
        motor_cmds = []
        for i in range(NUM_HAND_DDS_MOTORS * 2):
            motor_cmd = unitree_hg_msg_dds__MotorCmd_()
            motor_cmd.mode = 1
            motor_cmd.q = 0.0
            motor_cmd.kp = 0.0
            motor_cmd.dq = 0.0
            motor_cmd.kd = 2.0
            motor_cmd.tau = 0.0
            motor_cmds.append(motor_cmd)
        hand_cmd.motor_cmd = motor_cmds
        self.hand_cmd_pub.Write(hand_cmd)
        
        self.actionFinished.emit("All joints")

    def _on_action_timeout(self):
        joint_name = self.action_timer.property("joint_name")
        print(f"⏱️ Action timeout: {joint_name}")
        self.actionFinished.emit(joint_name)

    @Slot()
    def stop(self):
        self.running = False
        print("🛑 Stopping DDS controller...")
        # Clean up DDS resources if necessary (subscribers/publishers)
        # The underlying SDK handles this reasonably well, but could add explicit cleanup.


# --- UI Components ---

class JointControlWidget(QGroupBox):
    """A widget to display and control a single joint."""
    testCommandRequested = Signal(dict)

    def __init__(self, title, joint_info, torque_limit):
        super().__init__(title)
        self.joint_info = joint_info
        self.torque_limit = torque_limit
        
        self.setCheckable(True)
        self.setChecked(False)
        self.toggled.connect(self._toggle_details)

        # Main layout
        self.main_layout = QVBoxLayout()
        
        # Top-level info (always visible)
        top_layout = QGridLayout()
        self.dds_id_label = QLabel(f"DDS Idx: <b>{joint_info['dds_id']}</b>")
        self.q_label = QLabel("Pos: <b>--.-°</b>")
        self.dq_label = QLabel("Vel: <b>--.- rad/s</b>")
        top_layout.addWidget(self.dds_id_label, 0, 0)
        top_layout.addWidget(self.q_label, 0, 1)
        top_layout.addWidget(self.dq_label, 0, 2)
        self.main_layout.addLayout(top_layout)

        # Collapsible details widget
        self.details_widget = QWidget()
        details_layout = QVBoxLayout(self.details_widget)
        details_layout.setContentsMargins(0, 5, 0, 0)

        # --- Incoming Command Group ---
        incoming_group = QGroupBox("接收到的指令")
        incoming_layout = QFormLayout(incoming_group)
        self.q_target_label = QLabel("<b>--.-°</b>")
        self.kp_target_label = QLabel("<b>--.-</b>")
        self.dq_target_label = QLabel("<b>--.-</b>")
        self.kd_target_label = QLabel("<b>--.-</b>")
        self.tau_ff_label = QLabel("<b>--.- Nm</b>")
        incoming_layout.addRow("Tgt Pos:", self.q_target_label)
        incoming_layout.addRow("Tgt Kp:", self.kp_target_label)
        incoming_layout.addRow("Tgt Vel:", self.dq_target_label)
        incoming_layout.addRow("Tgt Kd:", self.kd_target_label)
        incoming_layout.addRow("Tgt Tau:", self.tau_ff_label)
        details_layout.addWidget(incoming_group)

        # --- Outgoing Control Group ---
        outgoing_group = QGroupBox("发送测试指令")
        outgoing_layout = QGridLayout(outgoing_group)

        self.tau_limit_label = QLabel(f"Max Torque: <b>{torque_limit:.1f} Nm</b>")
        self.kp_slider = QSlider(Qt.Horizontal)
        self.kp_slider.setRange(0, 200)
        self.kp_slider.setValue(60)
        self.kp_slider.setSingleStep(1)
        self.kp_spinbox = QDoubleSpinBox()
        self.kp_spinbox.setRange(0, 200)
        self.kp_spinbox.setValue(60)
        self.kp_spinbox.setSingleStep(0.5)
        self.kp_spinbox.setDecimals(1)
        self.plus_button = QPushButton("+5°")
        self.minus_button = QPushButton("-5°")
        self.plus_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.minus_button.setFont(QFont("Arial", 12, QFont.Bold))

        outgoing_layout.addWidget(self.tau_limit_label, 0, 0, 1, 3)
        outgoing_layout.addWidget(QLabel("Kp Gain:"), 1, 0)
        outgoing_layout.addWidget(self.kp_slider, 1, 1)
        outgoing_layout.addWidget(self.kp_spinbox, 1, 2)
        outgoing_layout.addWidget(self.minus_button, 2, 0, 1, 1)
        outgoing_layout.addWidget(self.plus_button, 2, 1, 1, 2)
        details_layout.addWidget(outgoing_group)

        self.main_layout.addWidget(self.details_widget)
        self.setLayout(self.main_layout)

        # Initial state
        self.details_widget.setVisible(False)

        # Connect signals
        self.kp_slider.valueChanged.connect(lambda val: self.kp_spinbox.setValue(val))
        self.kp_spinbox.valueChanged.connect(lambda val: self.kp_slider.setValue(int(val)))
        self.plus_button.clicked.connect(self._on_plus_clicked)
        self.minus_button.clicked.connect(self._on_minus_clicked)

    def _toggle_details(self, checked):
        self.details_widget.setVisible(checked)

    def _on_plus_clicked(self):
        self._emit_command(5.0 * DEG_TO_RAD)

    def _on_minus_clicked(self):
        self._emit_command(-5.0 * DEG_TO_RAD)

    def _emit_command(self, delta_rad):
        cmd_info = self.joint_info.copy()
        cmd_info['delta_rad'] = delta_rad
        cmd_info['kp'] = self.kp_spinbox.value()
        cmd_info['kp_default'] = 20.0 # Default Kp for holding other joints
        self.testCommandRequested.emit(cmd_info)

    def update_state(self, q, dq):
        self.q_label.setText(f"Pos: <b>{q * RAD_TO_DEG:.1f}°</b>")
        self.dq_label.setText(f"Vel: <b>{dq:.2f} rad/s</b>")

    def update_command_state(self, cmd: dict):
        self.q_target_label.setText(f"<b>{cmd['q'] * RAD_TO_DEG:.1f}°</b>")
        self.kp_target_label.setText(f"<b>{cmd['kp']:.1f}</b>")
        self.dq_target_label.setText(f"<b>{cmd['dq']:.2f} rad/s</b>")
        self.kd_target_label.setText(f"<b>{cmd['kd']:.1f}</b>")
        self.tau_ff_label.setText(f"<b>{cmd['tau']:.2f} Nm</b>")

    def set_enabled_controls(self, enabled):
        self.plus_button.setEnabled(enabled)
        self.minus_button.setEnabled(enabled)
        self.kp_slider.setEnabled(enabled)
        self.kp_spinbox.setEnabled(enabled)

# --- Main Application Window ---

class G1TesterUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("G1 Joint Single Point Test Tool")
        self.setGeometry(100, 100, 800, 900)

        # Setup DDS backend
        self.dds_thread = QThread()
        self.dds_controller = DDSController()
        self.dds_controller.moveToThread(self.dds_thread)

        # UI elements
        self.joint_widgets = {} # Keyed by mujoco_name
        self._init_ui()
        self._init_safety_layer()
        self._create_joint_controls()
        
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
        # Add common network interfaces, user can also type
        self.channel_combo.addItems(["enp7s0", "lo", "eth0", "wlan0"])
        self.channel_combo.setCurrentText("enp7s0")
        self.channel_combo.setEditable(True)
        self.connect_button = QPushButton("Connect")
        self.emergency_stop_button = QPushButton("🚨 Emergency Stop")
        self.emergency_stop_button.setStyleSheet("background-color: #d9534f; color: white;")
        self.emergency_stop_button.setFont(QFont("Arial", 12, QFont.Bold))

        top_layout.addWidget(self.connection_status_label)
        top_layout.addWidget(QLabel("Network Interface:"))
        top_layout.addWidget(self.channel_combo)
        top_layout.addWidget(self.connect_button)
        top_layout.addStretch()
        top_layout.addWidget(self.emergency_stop_button)
        main_layout.addWidget(top_panel)
        
        # --- Scroll Area for Joints ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        self.scroll_content_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        scroll_area.setWidget(self.scroll_content_widget)

    def _init_safety_layer(self):
        try:
            # Find a valid XML path. Start with g1_lab.xml
            xml_path = "g1_model/g1_lab.xml"
            if not os.path.exists(xml_path):
                # Fallback to another potential path if needed
                xml_path = "g1_model/scene.xml" 
            if not os.path.exists(xml_path):
                raise FileNotFoundError("Cannot find G1 XML model file (g1_lab.xml or scene.xml)")

            mj_model = mujoco.MjModel.from_xml_path(xml_path)
            self.safety_layer = HGSafetyLayer(mj_model)
            print("✅ Safety layer initialized successfully.")
        except Exception as e:
            print(f"❌ Safety layer initialization failed: {e}", file=sys.stderr)
            self.safety_layer = None
            # Create dummy torque limits if safety layer fails
            self.dummy_torque_limits = {i: 25.0 for i in range(NUM_BODY_DDS_MOTORS)}

    def _create_joint_controls(self):
        """Dynamically create control widgets for all joints."""
        
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
                    joint_type = "left_hand" if "Left" in group_name else "right_hand"
                    if mj_idx not in mapping: continue
                    dds_id = mapping[mj_idx]
                    torque_limit = 2.0 # Hand torque limits are smaller
                else: # Body group
                    if mj_idx not in BODY_MUJOCO_TO_DDS: continue
                    dds_id = BODY_MUJOCO_TO_DDS[mj_idx]
                    joint_type = "body"
                    if self.safety_layer:
                        torque_limit = self.safety_layer.torque_limits.get(dds_id, 0.0)
                    else:
                        torque_limit = self.dummy_torque_limits.get(dds_id, 0.0)

                joint_info = {
                    'mujoco_name': mujoco_name,
                    'dds_id': dds_id,
                    'mj_idx': mj_idx,
                    'type': joint_type
                }
                
                widget = JointControlWidget(mujoco_name, joint_info, torque_limit)
                widget.testCommandRequested.connect(self.dds_controller.send_test_command)
                group_layout.addWidget(widget)
                self.joint_widgets[mujoco_name] = widget
            
            self.scroll_layout.addWidget(group_box)

    def _connect_signals(self):
        # DDS Controller -> UI
        self.dds_controller.connectionStatusChanged.connect(self.on_connection_status_changed)
        self.dds_controller.newStateReceived.connect(self.on_new_state_received)
        self.dds_controller.commandReceived.connect(self.on_command_received)
        self.dds_controller.actionStarted.connect(self.on_action_started)
        self.dds_controller.actionFinished.connect(self.on_action_finished)
        
        # UI -> DDS Controller
        self.connect_button.clicked.connect(
            lambda: self.dds_controller.start(self.channel_combo.currentText())
        )
        self.emergency_stop_button.clicked.connect(self.dds_controller.emergency_stop)
        
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
        all_states = {**state_data['body'], **state_data['left_hand'], **state_data['right_hand']}
        for name, widget in self.joint_widgets.items():
            if name in all_states:
                widget.update_state(all_states[name]['q'], all_states[name]['dq'])

    @Slot(dict)
    def on_command_received(self, cmd_data):
        all_cmds = {**cmd_data.get('body',{}), **cmd_data.get('left_hand',{}), **cmd_data.get('right_hand',{})}
        for name, widget in self.joint_widgets.items():
            if name in all_cmds:
                widget.update_command_state(all_cmds[name])

    @Slot(str)
    def on_action_started(self, joint_name):
        print(f"UI: 锁定控件，正在测试 {joint_name}")
        for name, widget in self.joint_widgets.items():
            is_target = (name == joint_name)
            widget.set_enabled_controls(False)
            if is_target:
                widget.setStyleSheet("QGroupBox { border: 2px solid #5bc0de; }") # Highlight blue
            else:
                widget.setStyleSheet("")
        self.emergency_stop_button.setEnabled(False)

    @Slot(str)
    def on_action_finished(self, joint_name):
        print(f"UI: 解锁控件，测试完成/超时 {joint_name}")
        for name, widget in self.joint_widgets.items():
            widget.set_enabled_controls(True)
            widget.setStyleSheet("")
        self.emergency_stop_button.setEnabled(True)

    def closeEvent(self, event):
        self.dds_controller.stop()
        self.dds_thread.quit()
        self.dds_thread.wait()
        super().closeEvent(event)

# --- Main Execution ---

def main():
    # Graceful shutdown on Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    window = G1TesterUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 