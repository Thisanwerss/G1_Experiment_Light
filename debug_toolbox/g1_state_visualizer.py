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
from threading import Lock
from typing import Optional, Tuple
import json

# --- Color Printing Utility ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(tag, message, **kwargs):
    tag_map = {
        "ERROR": bcolors.FAIL,
        "WARNING": bcolors.WARNING,
        "SUCCEED": bcolors.OKGREEN,
        "INFO": bcolors.OKBLUE,
    }
    color = tag_map.get(tag, bcolors.ENDC)
    print(f"{color}[{tag}]{bcolors.ENDC} {message}", **kwargs)

# --- Global Configuration Loading ---
try:
    with open("global_config.json", "r") as f:
        GLOBAL_CONFIG = json.load(f)
    VICON_Z_OFFSET = GLOBAL_CONFIG.get("vicon_z_offset", 0.0)
    print_colored("SUCCEED", f"Loaded configuration from global_config.json, VICON_Z_OFFSET={VICON_Z_OFFSET}")
except FileNotFoundError:
    print_colored("WARNING", "global_config.json not found, using default values.")
    VICON_Z_OFFSET = 0.0
except json.JSONDecodeError:
    print_colored("ERROR", "Failed to parse global_config.json, using default values.")
    VICON_Z_OFFSET = 0.0

# MuJoCo and Vicon/ROS2 imports
try:
    import mujoco
    import mujoco.viewer
except ImportError as e:
    print_colored("WARNING", f"MuJoCo import failed: {e}. 3D visualization will be disabled.", file=sys.stderr)
    mujoco = None


# PySide6 imports for the UI
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QScrollArea, QGroupBox, QFormLayout, QComboBox, 
    QFrame, QGridLayout, QPushButton, QProgressBar, QCheckBox
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
    from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_, TwistStamped_
    # Project-specific imports
    from sdk_controller.robots.G1 import (
        MUJOCO_JOINT_NAMES, BODY_MUJOCO_TO_DDS, NUM_ACTIVE_BODY_JOINTS,
        LEFT_HAND_MUJOCO_TO_DDS, RIGHT_HAND_MUJOCO_TO_DDS
    )
    from sdk_controller.topics import TOPIC_VICON_POSE, TOPIC_VICON_TWIST
except ImportError as e:
    print_colored("ERROR", f"Module import failed: {e}", file=sys.stderr)
    print_colored("ERROR", "   Please ensure ATARI_NMPC root directory is added to your PYTHONPATH.", file=sys.stderr)
    print_colored("ERROR", "   Example: export PYTHONPATH=$PYTHONPATH:/path/to/ATARI_NMPC", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
RAD_TO_DEG = 180.0 / np.pi
NUM_HAND_DDS_MOTORS = 7

# --- Vicon Subscriber (adapted from zmq_dds_bridge.py) ---


# --- MuJoCo Visualization Thread ---

class MujocoVisualizer(QThread):
    finished = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.mj_model = None
        self.mj_data = None
        self.viewer = None
        
        self.state_data = None
        self.last_state_time = 0
        self.state_lock = Lock()

        self.joint_name_to_qpos_addr = {}

    @Slot(dict)
    def update_robot_state(self, state_data):
        with self.state_lock:
            self.state_data = state_data
            self.last_state_time = time.time()

    def run(self):
        self.running = True

        try:
            # Load model directly from path, assuming vicon_frame is now in scene.xml
            # Using from_xml_path correctly handles relative paths for includes.
            script_dir = os.path.dirname(__file__)
            # Handle case where script is run from the project root
            if not script_dir:
                script_dir = "."
            model_path = os.path.join(script_dir, "g1_model/scene.xml")
            
            self.mj_model = mujoco.MjModel.from_xml_path(model_path)
            self.mj_data = mujoco.MjData(self.mj_model)

            for i in range(self.mj_model.njnt):
                jnt_name = self.mj_model.joint(i).name
                # Use integer value for mjJNT_FREE (0) for robustness across mujoco versions
                if self.mj_model.joint(i).type != 0: # 0 is mjJNT_FREE
                    qpos_addr = self.mj_model.joint(i).qposadr[0]
                    self.joint_name_to_qpos_addr[jnt_name] = qpos_addr

        except Exception as e:
            print_colored("ERROR", f"Failed to load MuJoCo model: {e}", file=sys.stderr)
            self.running = False

        if self.running:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        while self.running and self.viewer and self.viewer.is_running():
            loop_start_time = time.time()
            
            # --- Robot State Data ---
            with self.state_lock:
                local_state_data = self.state_data
                state_age = time.time() - self.last_state_time
            
            if not local_state_data or state_age > 0.5:
                print_colored("INFO", " DDS robot state not available. Halting visualization update. ", end='\r', flush=True)
                time.sleep(0.1) # Prevent busy-waiting
                continue

            # --- Vicon Data ---
            vicon_data = local_state_data.get('vicon', {})
            p = vicon_data.get('p')
            q = vicon_data.get('q')

            if p is None or q is None or not vicon_data.get('is_active', False):
                # This will be true if DDS vicon data stream stops
                print_colored("INFO", " Vicon DDS data not available. Halting visualization update. ", end='\r', flush=True)
                time.sleep(0.1) # Prevent busy-waiting
                continue
            
            # Set the free joint pose from Vicon data
            qpos_addr = self.mj_model.joint('floating_base_joint').qposadr[0]
            self.mj_data.qpos[qpos_addr:qpos_addr+3] = p
            self.mj_data.qpos[qpos_addr+3:qpos_addr+7] = q

            # --- Robot DDS Joint State Data ---
            # --- Update MuJoCo Joint Positions ---
            all_joint_states = {**local_state_data.get('body', {}), 
                                **local_state_data.get('left_hand', {}), 
                                **local_state_data.get('right_hand', {})}

            for name, state in all_joint_states.items():
                if name in self.joint_name_to_qpos_addr:
                    qpos_addr = self.joint_name_to_qpos_addr[name]
                    self.mj_data.qpos[qpos_addr] = state['q']

            # --- Step Simulation and Render ---
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self.viewer.sync()
            
            elapsed = time.time() - loop_start_time
            sleep_time = max(0, 1/60.0 - elapsed)
            time.sleep(sleep_time)

        if self.viewer:
            self.viewer.close()
        print_colored("INFO", "3D Visualizer stopped.")
        self.finished.emit()

    def stop(self):
        self.running = False

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
        self.bms_state_sub = None
        
        # Hand state subscribers (one for each hand, lf and hf)
        self.left_hand_state_sub_lf = None
        self.left_hand_state_sub_hf = None
        self.right_hand_state_sub_lf = None
        self.right_hand_state_sub_hf = None

        # Vicon subscribers
        self.vicon_pose_sub = None
        self.vicon_twist_sub = None

        # Hand command subscribers
        self.low_cmd_sub = None
        self.left_hand_cmd_sub_lf = None
        self.left_hand_cmd_sub_hf = None
        self.right_hand_cmd_sub_lf = None
        self.right_hand_cmd_sub_hf = None
        
        self.last_low_state = None
        self.last_left_hand_state = None
        self.last_right_hand_state = None
        self.last_bms_state = None
        self.last_vicon_pose = None
        self.last_vicon_twist = None
        self.vicon_last_update_time = 0
        self.is_connected = False
        
        # Store the latest full state
        self.current_q = {} # Keyed by mujoco_name
        self.current_v = {} # Keyed by mujoco_name
        self.imu_data = {}  # Store IMU data

    @Slot(str)
    def start(self, channel: str):
        print_colored("INFO", f"Starting DDS listener, channel: {channel}...")
        try:
            domain_id = 0 if channel != "lo" else 1
            ChannelFactoryInitialize(domain_id, channel)

            # State Subscribers
            self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
            self.bms_state_sub = ChannelSubscriber("rt/lf/bmsstate", BmsState_)
            self.low_state_sub.Init(self._low_state_handler, 10)
            self.bms_state_sub.Init(self._bms_state_handler, 10)

            # Hand State Subscribers (lf and hf for robustness)
            self.left_hand_state_sub_lf = ChannelSubscriber("rt/lf/dex3/left/state", HandState_)
            self.left_hand_state_sub_hf = ChannelSubscriber("rt/dex3/left/state", HandState_)
            self.right_hand_state_sub_lf = ChannelSubscriber("rt/lf/dex3/right/state", HandState_)
            self.right_hand_state_sub_hf = ChannelSubscriber("rt/dex3/right/state", HandState_)
            self.left_hand_state_sub_lf.Init(self._left_hand_state_handler, 10)
            self.left_hand_state_sub_hf.Init(self._left_hand_state_handler, 10)
            self.right_hand_state_sub_lf.Init(self._right_hand_state_handler, 10)
            self.right_hand_state_sub_hf.Init(self._right_hand_state_handler, 10)

            # Vicon Subscribers
            self.vicon_pose_sub = ChannelSubscriber(TOPIC_VICON_POSE, PoseStamped_)
            self.vicon_twist_sub = ChannelSubscriber(TOPIC_VICON_TWIST, TwistStamped_)
            self.vicon_pose_sub.Init(self._vicon_pose_handler, 10)
            self.vicon_twist_sub.Init(self._vicon_twist_handler, 10)

            # Command Subscribers
            self.low_cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
            self.low_cmd_sub.Init(self._low_cmd_handler, 10)

            # Hand Command Subscribers
            self.left_hand_cmd_sub_lf = ChannelSubscriber("rt/lf/dex3/left/cmd", HandCmd_)
            self.left_hand_cmd_sub_hf = ChannelSubscriber("rt/dex3/left/cmd", HandCmd_)
            self.right_hand_cmd_sub_lf = ChannelSubscriber("rt/lf/dex3/right/cmd", HandCmd_)
            self.right_hand_cmd_sub_hf = ChannelSubscriber("rt/dex3/right/cmd", HandCmd_)
            self.left_hand_cmd_sub_lf.Init(self._left_hand_cmd_handler, 10)
            self.left_hand_cmd_sub_hf.Init(self._left_hand_cmd_handler, 10)
            self.right_hand_cmd_sub_lf.Init(self._right_hand_cmd_handler, 10)
            self.right_hand_cmd_sub_hf.Init(self._right_hand_cmd_handler, 10)
            
            self.running = True
            print_colored("SUCCEED", "DDS listener started successfully.")
        except Exception as e:
            print_colored("ERROR", f"DDS initialization failed: {e}", file=sys.stderr)
            self.connectionStatusChanged.emit(False)

    def _low_state_handler(self, msg: LowState_):
        self.last_low_state = msg
        if not self.is_connected:
            self.is_connected = True
            self.connectionStatusChanged.emit(True)
        self._process_state()

    def _vicon_pose_handler(self, msg: PoseStamped_):
        self.last_vicon_pose = msg
        self.vicon_last_update_time = time.time()
        self._process_state()

    def _vicon_twist_handler(self, msg: TwistStamped_):
        self.last_vicon_twist = msg
        # Twist data is secondary, no need to trigger a full state update from here
        # It will be picked up when pose or low_state triggers an update

    def _left_hand_state_handler(self, msg: HandState_):
        self.last_left_hand_state = msg
        self._process_state()

    def _right_hand_state_handler(self, msg: HandState_):
        self.last_right_hand_state = msg
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

    def _left_hand_cmd_handler(self, msg: HandCmd_):
        """Handles incoming Left HandCmd messages."""
        if not self.running: return
        cmd_data = {'left_hand': {}}
        if len(msg.motor_cmd) >= NUM_HAND_DDS_MOTORS:
            for mj_idx, dds_idx in LEFT_HAND_MUJOCO_TO_DDS.items():
                mj_name = MUJOCO_JOINT_NAMES[mj_idx]
                motor_cmd = msg.motor_cmd[dds_idx]
                cmd_data['left_hand'][mj_name] = {
                    'q': motor_cmd.q, 'kp': motor_cmd.kp,
                    'dq': motor_cmd.dq, 'kd': motor_cmd.kd,
                    'tau': motor_cmd.tau
                }
        self.commandReceived.emit(cmd_data)

    def _right_hand_cmd_handler(self, msg: HandCmd_):
        """Handles incoming Right HandCmd messages."""
        if not self.running: return
        cmd_data = {'right_hand': {}}
        if len(msg.motor_cmd) >= NUM_HAND_DDS_MOTORS:
            for mj_idx, dds_idx in RIGHT_HAND_MUJOCO_TO_DDS.items():
                mj_name = MUJOCO_JOINT_NAMES[mj_idx]
                motor_cmd = msg.motor_cmd[dds_idx]
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
        
        state_data = {'body': {}, 'left_hand': {}, 'right_hand': {}, 'imu': {}, 'bms': {}, 'vicon': {}}
        
        # Process Vicon data
        vicon_is_active = (time.time() - self.vicon_last_update_time) < 0.5
        state_data['vicon']['is_active'] = vicon_is_active
        if vicon_is_active and self.last_vicon_pose:
            pose = self.last_vicon_pose.pose
            p = np.array([pose.position.x, pose.position.y, pose.position.z])
            q = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
            state_data['vicon']['p'] = p
            state_data['vicon']['q'] = q
        
        if vicon_is_active and self.last_vicon_twist:
            twist = self.last_vicon_twist.twist
            v = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
            w = np.array([twist.angular.x, twist.angular.y, twist.angular.z])
            state_data['vicon']['v'] = v
            state_data['vicon']['w'] = w

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

        # Process left hand state
        if self.last_left_hand_state:
            if len(self.last_left_hand_state.motor_state) >= NUM_HAND_DDS_MOTORS:
                for mj_idx, dds_idx in LEFT_HAND_MUJOCO_TO_DDS.items():
                    mj_name = MUJOCO_JOINT_NAMES[mj_idx]
                    motor = self.last_left_hand_state.motor_state[dds_idx]
                    state_data['left_hand'][mj_name] = {'q': motor.q, 'dq': motor.dq}
        
        # Process right hand state
        if self.last_right_hand_state:
            if len(self.last_right_hand_state.motor_state) >= NUM_HAND_DDS_MOTORS:
                for mj_idx, dds_idx in RIGHT_HAND_MUJOCO_TO_DDS.items():
                    mj_name = MUJOCO_JOINT_NAMES[mj_idx]
                    motor = self.last_right_hand_state.motor_state[dds_idx]
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
        print_colored("INFO", "Stopping DDS listener...")

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

        # UI update throttling
        self.last_ui_update_time = 0
        self.ui_update_interval = 1.0 / 30.0 # 30 FPS

        # Setup DDS backend
        self.dds_thread = QThread()
        self.dds_receiver = DDSReceiver()
        self.dds_receiver.moveToThread(self.dds_thread)
        
        # Setup MuJoCo visualizer
        self.mujoco_visualizer = None

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

        self.connection_status_label = QLabel("Status: Disconnected")
        self.connection_status_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        # Channel selection
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Network Interface:"))
        self.channel_combo = QComboBox()
        
        # Populate with common interfaces and set default from config
        default_interface = load_default_interface_from_config()
        common_interfaces = [default_interface, "lo", "eth0", "wlan0"]
        # Remove duplicates by converting to a dict and back to a list
        unique_interfaces = list(dict.fromkeys(common_interfaces))
        self.channel_combo.addItems(unique_interfaces)
        self.channel_combo.setCurrentText(default_interface)

        channel_layout.addWidget(self.channel_combo)

        self.connect_button = QPushButton("Connect")
        self.show_3d_button = QPushButton("Show 3D Visualization")
        if mujoco is None:
            self.show_3d_button.setEnabled(False)
            self.show_3d_button.setText("MuJoCo Not Found")

        self.hide_ui_checkbox = QCheckBox("Hide UI for 3D Perf")
        self.hide_ui_checkbox.setChecked(True)

        top_layout.addWidget(self.connection_status_label)
        top_layout.addWidget(channel_layout)
        top_layout.addWidget(self.connect_button)
        top_layout.addWidget(self.show_3d_button)
        top_layout.addWidget(self.hide_ui_checkbox)
        top_layout.addStretch()
        main_layout.addWidget(top_panel)
        
        # --- Details Container ---
        self.details_container = QWidget()
        details_layout = QVBoxLayout(self.details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.details_container)

        # --- Status Panel (IMU and Battery) ---
        status_panel = QFrame()
        status_layout = QHBoxLayout(status_panel)

        self.imu_widget = IMUStateWidget("IMU State")
        self.battery_widget = BatteryStateWidget("Battery State")
        
        status_layout.addWidget(self.imu_widget)
        status_layout.addWidget(self.battery_widget)
        details_layout.addWidget(status_panel)

        # --- Scroll Area for Joints ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        details_layout.addWidget(scroll_area)
        
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
        self.show_3d_button.clicked.connect(self.toggle_3d_visualization)
        
        # Window closing event
        self.destroyed.connect(self.dds_thread.quit)
        self.destroyed.connect(self.dds_thread.wait)

    @Slot()
    def toggle_3d_visualization(self):
        if not self.mujoco_visualizer and mujoco is not None:
            print_colored("INFO", "Starting 3D visualization...")
            
            if self.hide_ui_checkbox.isChecked():
                self.details_container.setVisible(False)

            self.mujoco_visualizer = MujocoVisualizer(self)
            self.dds_receiver.newStateReceived.connect(self.mujoco_visualizer.update_robot_state)
            self.mujoco_visualizer.finished.connect(self.on_3d_viz_finished)
            self.mujoco_visualizer.start()
            
            self.show_3d_button.setText("3D Viz Running")
            self.show_3d_button.setEnabled(False)
            self.hide_ui_checkbox.setEnabled(False)

    @Slot()
    def on_3d_viz_finished(self):
        """Called when the MuJoCo viewer window is closed."""
        print_colored("INFO", "3D visualization window closed.")
        self.details_container.setVisible(True)
        
        self.show_3d_button.setText("Show 3D Visualization")
        self.show_3d_button.setEnabled(True)
        self.hide_ui_checkbox.setEnabled(True)
        if self.mujoco_visualizer:
            self.dds_receiver.newStateReceived.disconnect(self.mujoco_visualizer.update_robot_state)
        self.mujoco_visualizer = None

    @Slot(bool)
    def on_connection_status_changed(self, is_connected):
        if is_connected:
            self.connection_status_label.setText("Status: Connected")
            self.connection_status_label.setStyleSheet("color: green")
            self.connect_button.setEnabled(False)
            self.channel_combo.setEnabled(False)
        else:
            self.connection_status_label.setText("Status: Disconnected")
            self.connection_status_label.setStyleSheet("color: red")
            self.connect_button.setEnabled(True)
            self.channel_combo.setEnabled(True)

    @Slot(dict)
    def on_new_state_received(self, state_data):
        # Optimization: Don't update UI elements if they are hidden for performance
        if not self.details_container.isVisible():
            return

        # Throttle UI updates to save resources
        current_time = time.time()
        if current_time - self.last_ui_update_time < self.ui_update_interval:
            return
        self.last_ui_update_time = current_time

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
        # Optimization: Don't update UI elements if they are hidden for performance
        if not self.details_container.isVisible():
            return
            
        all_cmds = {**cmd_data.get('body',{}), **cmd_data.get('left_hand',{}), **cmd_data.get('right_hand',{})}
        for name, widget in self.joint_widgets.items():
            if name in all_cmds:
                widget.update_command_state(all_cmds[name])

    def closeEvent(self, event):
        if self.mujoco_visualizer:
            self.mujoco_visualizer.stop()
        self.dds_receiver.stop()
        self.dds_thread.quit()
        self.dds_thread.wait()
        super().closeEvent(event)

# --- Main Execution ---

def load_default_interface_from_config():
    """Loads the default network interface from the global configuration file."""
    try:
        with open("global_config.json", "r") as f:
            config = json.load(f)
        return config.get("default_network_interface", "lo")
    except (FileNotFoundError, json.JSONDecodeError):
        return "lo"

def main():
    # Graceful shutdown on Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    window = G1VisualizerUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 