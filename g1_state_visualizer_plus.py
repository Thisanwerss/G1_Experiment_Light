#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G1 Robot State Visualizer Plus
==============================

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
3. Run: `python g1_state_visualizer_plus.py`
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
    QFrame, QGridLayout, QPushButton, QProgressBar, QTextEdit, QTabWidget
)
from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread
from PySide6.QtGui import QFont, QColor

# DDS and robot specific imports
try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelSubscriber
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        LowState_, HandState_, BmsState_, LowCmd_, HandCmd_, IMUState_
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
NUM_BODY_DDS_MOTORS = 35
NUM_HAND_PRESSURE_SENSORS = 12

# --- Error Dictionaries ---
DEVICE_ERROR_CODES = {
    0x01: "上层控制命令错误",
    0x02: "下层反馈数据超时",
    0x04: "imu反馈数据超时",
    0x08: "电机反馈数据超时",
    0x10: "电池反馈数据超时",
    0x20: "宇树实体遥控器反馈数据超时",
    0x40: "电池型号错误",
    0x80: "软启动错误",
    0x100: "电机状态错误",
    0x200: "电机过流保护，触发低限位保护",
    0x400: "电机欠压保护，触发高限位保护",
    0x800: "电机过流保护，触发高限位保护",
    0x1000: "软急停开关被按下",
    0x2000: "SN错误",
    0x4000: "上层机型错误",
    0x8000: "下层机型错误",
    0x10000: "USB设备错误",
    0x40000: "胯部IMU数据超时",
    0x80000: "主板判断电池欠压保护错误",
    0x100000: "主板判断电机欠压保护错误",
}

MOTOR_ERROR_CODES = {
    0x01: "过流",
    0x02: "瞬态过压",
    0x04: "持续过压",
    0x08: "瞬态欠压",
    0x10: "芯片过热",
    0x20: "MOS过热/过冷",
    0x40: "MOS温度异常",
    0x80: "壳体过热/过冷",
    0x100: "壳体温度异常",
    0x200: "绕组过热",
    0x400: "转子编码器1错误",
    0x800: "转子编码器2错误",
    0x1000: "输出编码器错误",
    0x2000: "标定/BOOT数据错误",
    0x4000: "异常复位",
    0x8000: "电机锁定，主控认证错误",
    0x10000: "芯片验证错误",
    0x20000: "标定模式警告",
    0x40000: "通信校验错误",
    0x80000: "驱动版本过低",
    0x40000000: "电机端判断，PC连接超时",
    0x80000000: "PC端判读，电机断联超时",
}


# --- DDS Communication Backend ---

class DDSReceiver(QObject):
    """Handles all DDS state subscription in a background thread."""
    connectionStatusChanged = Signal(bool)
    newStateReceived = Signal(dict)

    def __init__(self):
        super().__init__()
        self.running = False
        self.subscribers = {}
        self.last_states = {
            'low': None, 'left_hand': None, 'right_hand': None,
            'bms': None, 'odom': None, 'secondary_imu': None
        }
        self.is_connected = False

    @Slot(str)
    def start(self, channel: str):
        print(f"🚀 正在启动DDS监听器，通道: {channel}...")
        try:
            domain_id = 0 if channel != "lo" else 1
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
            print("✅ DDS监听器启动成功。")
        except Exception as e:
            print(f"❌ DDS初始化失败: {e}", file=sys.stderr)
            self.connectionStatusChanged.emit(False)

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

# --- Main Application Window ---
class G1VisualizerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("G1 Robot State Visualizer Plus")
        self.setGeometry(100, 100, 1200, 950)

        # Setup DDS backend
        self.dds_thread = QThread()
        self.dds_receiver = DDSReceiver()
        self.dds_receiver.moveToThread(self.dds_thread)

        # UI elements
        self.body_motor_widgets = {} # Keyed by mujoco_name
        self.left_hand_motor_widgets = {}
        self.right_hand_motor_widgets = {}

        self._init_ui()
        self._create_motor_widgets()
        self._connect_signals()

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
        self.channel_combo.addItems(["enp7s0", "lo", "eth0", "wlan0"])
        self.connect_button = QPushButton("Connect")
        top_layout.addWidget(self.connection_status_label)
        top_layout.addWidget(QLabel("Network Interface:"))
        top_layout.addWidget(self.channel_combo)
        top_layout.addWidget(self.connect_button)
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
        self.connect_button.clicked.connect(
            lambda: self.dds_receiver.start(self.channel_combo.currentText())
        )
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
def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    window = G1VisualizerUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 