import mujoco
import numpy as np
import pygame
import sys
import struct

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.utils.thread import RecurrentThread

import config_g1 as config

# G1 使用 unitree_hg 消息
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_ as HandState_default

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"
TOPIC_HANDCMD = "rt/handcmd"
TOPIC_HANDSTATE = "rt/handstate"

MOTOR_SENSOR_NUM = 3
NUM_MOTOR_BODY = 29  # G1 身体电机数量
NUM_MOTOR_HAND = 7   # 每只手的电机数量


class UnitreeSdk2BridgeG1:
    """G1 机器人的 SDK2 桥接器"""

    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data

        # G1 的实际关节数量
        self.num_motor = min(self.mj_model.nu, 41)  # 最多 41 个关节（27 身体 + 14 手指）
        self.num_body_motor = 27  # 实际身体电机数（不包括 waist_roll 和 waist_pitch）
        self.num_hand_motor = 14  # 两只手的电机总数
        
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.have_imu = False
        self.have_frame_sensor = False
        self.dt = self.mj_model.opt.timestep

        self.joystick = None

        # 检查传感器
        for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name == "imu_quat" or name == "imu_in_torso_quat":
                self.have_imu_ = True
            if name == "frame_pos":
                self.have_frame_sensor_ = True

        # Unitree SDK2 消息
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()
        self.lowStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishLowState, name="sim_lowstate"
        )
        self.lowStateThread.Start()
        
        # 手部状态
        self.hand_state = HandState_default()
        self.hand_state_puber = ChannelPublisher(TOPIC_HANDSTATE, HandState_)
        self.hand_state_puber.Init()
        self.handStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishHandState, name="sim_handstate"
        )
        self.handStateThread.Start()

        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.HighStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishHighState, name="sim_highstate"
        )
        self.HighStateThread.Start()

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            TOPIC_WIRELESS_CONTROLLER, WirelessController_
        )
        self.wireless_controller_puber.Init()
        self.WirelessControllerThread = RecurrentThread(
            interval=0.01,
            target=self.PublishWirelessController,
            name="sim_wireless_controller",
        )
        self.WirelessControllerThread.Start()

        # 订阅控制命令
        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)
        
        self.hand_cmd_suber = ChannelSubscriber(TOPIC_HANDCMD, HandCmd_)
        self.hand_cmd_suber.Init(self.HandCmdHandler, 10)

        # 手柄按键映射
        self.key_map = {
            "R1": 0, "L1": 1, "start": 2, "select": 3,
            "R2": 4, "L2": 5, "F1": 6, "F2": 7,
            "A": 8, "B": 9, "X": 10, "Y": 11,
            "up": 12, "right": 13, "down": 14, "left": 15,
        }

    def LowCmdHandler(self, msg: LowCmd_):
        """处理身体电机命令"""
        if self.mj_data != None:
            # 只处理前 27 个身体关节
            for i in range(min(self.num_body_motor, NUM_MOTOR_BODY)):
                # 跳过 DDS 索引 13 和 14（waist_roll 和 waist_pitch）
                dds_idx = i
                
                if dds_idx == 13 or dds_idx == 14:
                    continue
                
                # 映射到 MuJoCo 关节索引
                mj_idx = self._dds_to_mujoco_body(dds_idx)
                if mj_idx < 0 or mj_idx >= self.num_motor:
                    continue
                
                # PD 控制
                self.mj_data.ctrl[mj_idx] = (
                    msg.motor_cmd[dds_idx].tau
                    + msg.motor_cmd[dds_idx].kp
                    * (msg.motor_cmd[dds_idx].q - self.mj_data.sensordata[mj_idx])
                    + msg.motor_cmd[dds_idx].kd
                    * (
                        msg.motor_cmd[dds_idx].dq
                        - self.mj_data.sensordata[mj_idx + self.num_motor]
                    )
                )
    
    def HandCmdHandler(self, msg: HandCmd_):
        """处理手部电机命令"""
        if self.mj_data != None:
            # 左手
            for i in range(NUM_MOTOR_HAND):
                mj_idx = self._dds_to_mujoco_left_hand(i)
                if mj_idx >= 0 and mj_idx < self.num_motor:
                    self.mj_data.ctrl[mj_idx] = (
                        msg.left_hand_cmd.motor_cmd[i].tau
                        + msg.left_hand_cmd.motor_cmd[i].kp
                        * (msg.left_hand_cmd.motor_cmd[i].q - self.mj_data.sensordata[mj_idx])
                        + msg.left_hand_cmd.motor_cmd[i].kd
                        * (
                            msg.left_hand_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[mj_idx + self.num_motor]
                        )
                    )
            
            # 右手
            for i in range(NUM_MOTOR_HAND):
                mj_idx = self._dds_to_mujoco_right_hand(i)
                if mj_idx >= 0 and mj_idx < self.num_motor:
                    self.mj_data.ctrl[mj_idx] = (
                        msg.right_hand_cmd.motor_cmd[i].tau
                        + msg.right_hand_cmd.motor_cmd[i].kp
                        * (msg.right_hand_cmd.motor_cmd[i].q - self.mj_data.sensordata[mj_idx])
                        + msg.right_hand_cmd.motor_cmd[i].kd
                        * (
                            msg.right_hand_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[mj_idx + self.num_motor]
                        )
                    )

    def _dds_to_mujoco_body(self, dds_idx):
        """DDS 身体关节索引到 MuJoCo 索引的映射"""
        # 反向映射表
        dds_to_mj = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,      # 左腿
            6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,  # 右腿
            12: 12,  # waist_yaw
            # 13, 14 跳过 (waist_roll, waist_pitch)
            15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,  # 左臂
            22: 20, 23: 21, 24: 22, 25: 23, 26: 24, 27: 25, 28: 26,  # 右臂
        }
        return dds_to_mj.get(dds_idx, -1)
    
    def _dds_to_mujoco_left_hand(self, hand_idx):
        """DDS 左手索引到 MuJoCo 索引的映射"""
        hand_to_mj = {
            0: 27, 1: 28, 2: 29,  # thumb
            3: 32, 4: 33,         # index
            5: 30, 6: 31,         # middle
        }
        return hand_to_mj.get(hand_idx, -1)
    
    def _dds_to_mujoco_right_hand(self, hand_idx):
        """DDS 右手索引到 MuJoCo 索引的映射"""
        hand_to_mj = {
            0: 34, 1: 35, 2: 36,  # thumb
            3: 37, 4: 38,         # index
            5: 39, 6: 40,         # middle
        }
        return hand_to_mj.get(hand_idx, -1)

    def PublishLowState(self):
        """发布身体电机状态"""
        if self.mj_data != None:
            # 发布身体关节状态
            for dds_idx in range(NUM_MOTOR_BODY):
                if dds_idx == 13 or dds_idx == 14:
                    # waist_roll 和 waist_pitch 设置为 0
                    self.low_state.motor_state[dds_idx].q = 0.0
                    self.low_state.motor_state[dds_idx].dq = 0.0
                    self.low_state.motor_state[dds_idx].tau_est = 0.0
                    continue
                
                mj_idx = self._dds_to_mujoco_body(dds_idx)
                if mj_idx >= 0 and mj_idx < self.num_motor:
                    self.low_state.motor_state[dds_idx].q = self.mj_data.sensordata[mj_idx]
                    self.low_state.motor_state[dds_idx].dq = self.mj_data.sensordata[
                        mj_idx + self.num_motor
                    ]
                    self.low_state.motor_state[dds_idx].tau_est = self.mj_data.sensordata[
                        mj_idx + 2 * self.num_motor
                    ]

            # IMU 数据
            if self.have_frame_sensor_:
                self.low_state.imu_state.quaternion[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 0
                ]
                self.low_state.imu_state.quaternion[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 1
                ]
                self.low_state.imu_state.quaternion[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 2
                ]
                self.low_state.imu_state.quaternion[3] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 3
                ]

                self.low_state.imu_state.gyroscope[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 4
                ]
                self.low_state.imu_state.gyroscope[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 5
                ]
                self.low_state.imu_state.gyroscope[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 6
                ]

                self.low_state.imu_state.accelerometer[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 7
                ]
                self.low_state.imu_state.accelerometer[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 8
                ]
                self.low_state.imu_state.accelerometer[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 9
                ]

            self.low_state_puber.Write(self.low_state)
    
    def PublishHandState(self):
        """发布手部状态"""
        if self.mj_data != None:
            # 左手状态
            for i in range(NUM_MOTOR_HAND):
                mj_idx = self._dds_to_mujoco_left_hand(i)
                if mj_idx >= 0 and mj_idx < self.num_motor:
                    self.hand_state.left_hand_state.motor_state[i].q = self.mj_data.sensordata[mj_idx]
                    self.hand_state.left_hand_state.motor_state[i].dq = self.mj_data.sensordata[
                        mj_idx + self.num_motor
                    ]
                    self.hand_state.left_hand_state.motor_state[i].tau_est = self.mj_data.sensordata[
                        mj_idx + 2 * self.num_motor
                    ]
            
            # 右手状态
            for i in range(NUM_MOTOR_HAND):
                mj_idx = self._dds_to_mujoco_right_hand(i)
                if mj_idx >= 0 and mj_idx < self.num_motor:
                    self.hand_state.right_hand_state.motor_state[i].q = self.mj_data.sensordata[mj_idx]
                    self.hand_state.right_hand_state.motor_state[i].dq = self.mj_data.sensordata[
                        mj_idx + self.num_motor
                    ]
                    self.hand_state.right_hand_state.motor_state[i].tau_est = self.mj_data.sensordata[
                        mj_idx + 2 * self.num_motor
                    ]
            
            self.hand_state_puber.Write(self.hand_state)

    def PublishHighState(self):
        """发布高级状态"""
        if self.mj_data != None:
            # 基座位置和速度
            self.high_state.position[0] = self.mj_data.qpos[0]
            self.high_state.position[1] = self.mj_data.qpos[1]
            self.high_state.position[2] = self.mj_data.qpos[2]

            self.high_state.velocity[0] = self.mj_data.qvel[0]
            self.high_state.velocity[1] = self.mj_data.qvel[1]
            self.high_state.velocity[2] = self.mj_data.qvel[2]
            
            # IMU 数据（如果有）
            if self.have_frame_sensor_:
                self.high_state.imu_state.quaternion = self.low_state.imu_state.quaternion
                self.high_state.imu_state.gyroscope = self.low_state.imu_state.gyroscope
                self.high_state.imu_state.accelerometer = self.low_state.imu_state.accelerometer

            self.high_state_puber.Write(self.high_state)

    def PublishWirelessController(self):
        """发布手柄控制信号"""
        if self.joystick != None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(
                self.button_id["RB"]
            )
            key_state[self.key_map["L1"]] = self.joystick.get_button(
                self.button_id["LB"]
            )
            key_state[self.key_map["start"]] = self.joystick.get_button(
                self.button_id["START"]
            )
            key_state[self.key_map["select"]] = self.joystick.get_button(
                self.button_id["SELECT"]
            )
            key_state[self.key_map["R2"]] = (
                self.joystick.get_axis(self.axis_id["RT"]) > 0
            )
            key_state[self.key_map["L2"]] = (
                self.joystick.get_axis(self.axis_id["LT"]) > 0
            )
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        """设置手柄"""
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("未检测到手柄")
            return

        if js_type == "xbox":
            self.axis_id = {
                "LX": 0, "LY": 1, "RX": 3, "RY": 4,
                "LT": 2, "RT": 5, "DX": 6, "DY": 7,
            }
            self.button_id = {
                "X": 2, "Y": 3, "B": 1, "A": 0,
                "LB": 4, "RB": 5, "SELECT": 6, "START": 7,
            }
        elif js_type == "switch":
            self.axis_id = {
                "LX": 0, "LY": 1, "RX": 2, "RY": 3,
                "LT": 5, "RT": 4, "DX": 6, "DY": 7,
            }
            self.button_id = {
                "X": 3, "Y": 4, "B": 1, "A": 0,
                "LB": 6, "RB": 7, "SELECT": 10, "START": 11,
            }
        else:
            print("不支持的手柄类型")

    def PrintSceneInformation(self):
        """打印场景信息"""
        print("=============== 场景信息 ===============")
        print(f"仿真时间步长: {self.mj_model.opt.timestep}")
        print(f"控制维度: {self.mj_model.nu}")
        print(f"广义坐标维度: {self.mj_model.nq}")
        print(f"广义速度维度: {self.mj_model.nv}")
        print(f"身体电机数: {self.num_body_motor}")
        print(f"手部电机数: {self.num_hand_motor}")
        print(f"传感器数: {self.mj_model.nsensor}")
        
        print("\n关节列表:")
        for i in range(self.mj_model.njnt):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"  [{i}] {joint_name}")
        
        print("\n执行器列表:")
        for i in range(self.mj_model.nu):
            actuator_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  [{i}] {actuator_name}")


# 导出类
UnitreeSdk2Bridge = UnitreeSdk2BridgeG1
