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

# --- 全局配置加载 ---
try:
    with open("global_config.json", "r") as f:
        GLOBAL_CONFIG = json.load(f)
    VICON_Z_OFFSET = GLOBAL_CONFIG.get("vicon_z_offset", 0.0)
    print(f"✅ 从 global_config.json 加载配置, VICON_Z_OFFSET={VICON_Z_OFFSET}")
except FileNotFoundError:
    print("⚠️ global_config.json 未找到, 使用默认值。")
    VICON_Z_OFFSET = 0.0
except json.JSONDecodeError:
    print("❌ global_config.json 解析失败, 使用默认值。")
    VICON_Z_OFFSET = 0.0

# MuJoCo and Vicon/ROS2 imports
try:
    import mujoco
    import mujoco.viewer
except ImportError as e:
    print(f"⚠️ MuJoCo import failed: {e}. 3D visualization will be disabled.", file=sys.stderr)
    mujoco = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from vicon_receiver.msg import Position
    ROS2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ROS2_AVAILABLE = False
    print("⚠️ ROS2/vicon_receiver not found. Vicon visualization will use dummy data.", file=sys.stderr)


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

# --- Vicon Subscriber (adapted from zmq_dds_bridge.py) ---

if ROS2_AVAILABLE:
    class ViconSubscriber(Node):
        """通过ROS2订阅Vicon数据，并计算速度"""
        def __init__(self):
            super().__init__('g1_visualizer_vicon_seubscriber')
            self.lock = Lock()
            
            self.p = np.zeros(3)
            self.q = np.array([1.,0.,0.,0.]) # (w, x, y, z)
            self.v = np.zeros(3)
            self.w = np.zeros(3)
            
            self.last_update_time = 0
            self.is_active = False

            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.subscription = self.create_subscription(
                Position,
                '/vicon/G1/G1',
                self.listener_callback,
                qos_profile)
            print("✅ Vicon ROS2 订阅器创建成功，话题: /vicon/G1/G1")

        def listener_callback(self, msg: Position):
            """处理传入的Vicon消息"""
            with self.lock:
                current_time = time.time()
                self.p = np.array([msg.x_trans, msg.y_trans, msg.z_trans]) / 1000.0
                self.p[2] += VICON_Z_OFFSET # 应用Z轴偏移
                self.q = np.array([msg.w, msg.x_rot, msg.y_rot, msg.z_rot])
                
                self.is_active = True
                self.last_update_time = current_time

        def get_state(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            """获取最新的位姿和速度"""
            with self.lock:
                if time.time() - self.last_update_time > 0.5:
                    self.is_active = False
                    return None, None, None, None
                
                # For visualization, we only need position and orientation
                return self.p.copy(), self.q.copy(), self.v.copy(), self.w.copy()
        
        def start(self):
            """在后台线程中启动ROS2节点"""
            from threading import Thread
            self.thread = Thread(target=self.run_node, daemon=True)
            self.thread.start()

        def run_node(self):
            """运行rclpy.spin()"""
            print("Vicon subscriber thread started.")
            try:
                # rclpy.init() has been called outside
                rclpy.spin(self)
            except Exception as e:
                print(f"RCLPY spin failed: {e}")

else: # Dummy ViconSubscriber if ROS2 is not available
    class ViconSubscriber:
        def __init__(self): pass
        def start(self): pass
        def get_state(self) -> Tuple[None, None, None, None]:
            return None, None, None, None
        def destroy_node(self): pass

# --- MuJoCo Visualization Thread ---

class MujocoVisualizer(QThread):
    finished = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.mj_model = None
        self.mj_data = None
        self.viewer = None
        self.vicon_subscriber = None
        
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

        if ROS2_AVAILABLE:
            rclpy.init()
            self.vicon_subscriber = ViconSubscriber()
            self.vicon_subscriber.start()
        else:
            self.vicon_subscriber = ViconSubscriber()

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
            print(f"❌ Failed to load MuJoCo model: {e}", file=sys.stderr)
            self.running = False

        if self.running:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        while self.running and self.viewer and self.viewer.is_running():
            loop_start_time = time.time()
            
            # --- Vicon Data ---
            p, q, _, _ = self.vicon_subscriber.get_state()
            if p is None or q is None:
                # This will be true if ROS is unavailable or if data stream stops
                print(" Vicon data not available. Halting visualization update. ", end='\r', flush=True)
                time.sleep(0.1) # Prevent busy-waiting
                continue
            
            # Set the free joint pose from Vicon data
            qpos_addr = self.mj_model.joint('floating_base_joint').qposadr[0]
            self.mj_data.qpos[qpos_addr:qpos_addr+3] = p
            self.mj_data.qpos[qpos_addr+3:qpos_addr+7] = q

            # --- Robot DDS State Data ---
            with self.state_lock:
                local_state_data = self.state_data
                state_age = time.time() - self.last_state_time
            
            if not local_state_data or state_age > 0.5:
                print(" DDS robot state not available. Halting visualization update. ", end='\r', flush=True)
                time.sleep(0.1) # Prevent busy-waiting
                continue

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
        if ROS2_AVAILABLE and rclpy.ok():
            self.vicon_subscriber.destroy_node()
            rclpy.shutdown()
        print("⏹️ 3D Visualizer stopped.")
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
            print("🚀 Starting 3D visualization...")
            
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
        print("3D visualization window closed.")
        self.details_container.setVisible(True)
        
        self.show_3d_button.setText("Show 3D Visualization")
        self.show_3d_button.setEnabled(True)
        self.hide_ui_checkbox.setEnabled(True)
        self.mujoco_visualizer = None

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

def main():
    # Graceful shutdown on Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    window = G1VisualizerUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 