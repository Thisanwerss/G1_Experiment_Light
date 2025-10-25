#!/usr/bin/env python3
"""
Dummy DDS 控制器
=====================
该脚本用于直接通过DDS向G1机器人发送一个固定的控制指令，
使其保持在一个微屈膝的站立姿态。

它不依赖于ZMQ或任何外部策略，主要用于测试DDS通信链路和机器人对PD指令的响应。

使用方式:
1. 本地回环测试 (lo模式):
   python dummy_dds_controller.py --channel lo
2. 控制真实机器人 (需要Vicon):
   python dummy_dds_controller.py --channel <network_interface>

启动Vicon的命令:
ros2 launch vicon_receiver client.launch.py
"""
import argparse
import time
import struct
from typing import Dict, Any, Optional, Tuple
from threading import Thread, Event, Lock
import signal
import sys
import numpy as np
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


# 真实机器人相关导入
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from sdk_controller.robots.G1 import *
from sdk_controller.abstract_biped import HGSDKController
from typing import Dict, Any

# Vicon/ROS2相关导入 - 已被移除，使用DDS Vicon订阅


class CEMSDKController(HGSDKController):
    """CEM控制器 - 通过ZMQ接收外部策略的PD目标，专用于G1机器人"""
    
    def __init__(self, simulate: bool = False, robot_config=None, xml_path: str = "", vicon_required: bool = True, lo_mode: bool = False, kp_scale_factor: float = 1.0, safety_profile: str = "default"):
        """
        初始化CEM控制器
        
        Args:
            simulate: 是否仿真模式
            robot_config: 机器人配置
            xml_path: URDF/XML路径
            vicon_required: 是否需要Vicon定位
            lo_mode: 是否lo模式
            kp_scale_factor: Kp增益缩放因子
            safety_profile: 安全配置文件名称
        """
        print(f"🤖 初始化CEMSDKController (DDS Vicon模式)")
        print(f"   仿真模式: {simulate}")
        print(f"   需要Vicon: {vicon_required}")
        print(f"   lo模式: {lo_mode}")
        
        # 初始化HGSDKController
        super().__init__(
            simulate=simulate,
            robot_config=robot_config,
            xml_path=xml_path,
            vicon_required=vicon_required,
            lo_mode=lo_mode,
            kp_scale_factor=kp_scale_factor,
            safety_profile=safety_profile
        )
        
        # CEM控制相关状态
        self.current_pd_targets = None
        self.waiting_for_targets = True
        self.safety_emergency_stop = False
        
        # Vicon状态缓存 - 父类HGSDKController会自动处理DDS订阅，此处无需操作
        if vicon_required:
            print("   DDS Vicon 订阅器已由父类自动初始化。")
        
        print("🎯 CEMSDKController初始化完成")
        
    def update_motor_cmd(self, time: float):
        """实现抽象方法 - CEMSDKController主要通过外部PD目标控制"""
        # 当使用外部PD目标时，这个方法通常不会被调用
        # 保留为占位符或紧急情况处理
        if self.safety_emergency_stop:
            print("🛑 安全紧急停止：切换到阻尼模式")
            self.damping_motor_cmd()
        else:
            # 如果没有外部目标，使用默认站立姿态
            if self.current_pd_targets is None:
                print("⚠️ 无外部PD目标，使用默认站立姿态")
                self.update_motor_cmd_from_pd_targets(STAND_UP_JOINT_POS)
    
    def get_robot_state(self) -> Dict[str, Any]:
        """获取G1机器人状态 - 兼容ZMQ桥接格式"""
        if self.lo_mode:
            # lo模式：返回dummy状态（固定站立姿态）
            return self._get_dummy_state_for_cem()

        # 更新DDS的关节状态
        self.update_q_v_from_lowstate()
        self.update_hand_q_v_from_handstate()
        
        # 初始化mocap值为默认值
        mocap_pos_to_send = np.zeros(3)
        mocap_quat_to_send = np.array([1, 0, 0, 0])

        # 从Vicon更新基座状态 (通过DDS)
        if self.vicon_required:
            p, q, v, w = None, None, None, None
            
            # 检查Vicon DDS消息是否超时
            current_time = time.time()
            vicon_timeout = 0.5 # 秒

            if self.last_vicon_pose is not None:
                pose_timestamp = self.last_vicon_pose.header.stamp.sec + self.last_vicon_pose.header.stamp.nanosec * 1e-9
                if current_time - pose_timestamp < vicon_timeout:
                    p = np.array([
                        self.last_vicon_pose.pose.position.x,
                        self.last_vicon_pose.pose.position.y,
                        self.last_vicon_pose.pose.position.z,
                    ])
                    q = np.array([
                        self.last_vicon_pose.pose.orientation.w,
                        self.last_vicon_pose.pose.orientation.x,
                        self.last_vicon_pose.pose.orientation.y,
                        self.last_vicon_pose.pose.orientation.z,
                    ])
            
            if self.last_vicon_twist is not None:
                twist_timestamp = self.last_vicon_twist.header.stamp.sec + self.last_vicon_twist.header.stamp.nanosec * 1e-9
                if current_time - twist_timestamp < vicon_timeout:
                    v = np.array([
                        self.last_vicon_twist.twist.linear.x,
                        self.last_vicon_twist.twist.linear.y,
                        self.last_vicon_twist.twist.linear.z,
                    ])
                    w = np.array([
                        self.last_vicon_twist.twist.angular.x,
                        self.last_vicon_twist.twist.angular.y,
                        self.last_vicon_twist.twist.angular.z,
                    ])
            
            # 如果Vicon数据有效，则更新基座状态
            if p is not None and q is not None and v is not None and w is not None:
                self._q[0:3] = p
                self._q[3:7] = q  # (w, x, y, z)
                self._v[0:3] = v
                self._v[3:6] = w
                # 同样用vicon数据填充mocap字段，以对齐sim
                mocap_pos_to_send = p.copy()
                mocap_quat_to_send = q.copy()
            else:
                # Vicon数据无效或超时
                print("❌ get_robot_state: 无效的Vicon数据 (DDS超时或未接收)，返回None", flush=True)
                return None
        
        # 检查DDS数据是否有效（一个简单的完整性检查）
        # 7: 之后是关节qpos。如果它们都是零，很可能意味着没有收到DDS数据。
        if np.all(self._q[7:] == 0):
            print("❌ get_robot_state: 关节数据全为零，可能未收到DDS数据。返回None。", flush=True)
            return None

        # 返回ZMQ兼容格式
        return {
            'qpos': self._q.copy(),
            'qvel': self._v.copy(),
            'mocap_pos': mocap_pos_to_send,
            'mocap_quat': mocap_quat_to_send,
            'time': time.time()
        }

    def _get_dummy_state_for_cem(self) -> Dict[str, Any]:
        """为lo模式生成dummy状态"""
        dummy_qpos = np.zeros(48)
        dummy_qvel = np.zeros(47)
        dummy_qpos[2] = 0.75  # z
        dummy_qpos[3] = 1.0   # qw
        return {
            'qpos': dummy_qpos, 'qvel': dummy_qvel,
            'mocap_pos': np.zeros(3), 'mocap_quat': np.array([1,0,0,0]),
            'time': time.time()
        }
    
    def send_motor_command(self, time: float, pd_targets: Optional[np.ndarray] = None):
        """带安全检查的电机控制命令发送"""
        if pd_targets is not None:
            self.current_pd_targets = pd_targets.copy()
        
        # 调用父类方法
        super().send_motor_command(time, pd_targets)


class DummyController:
    """一个简单的DDS控制器，用于发送固定的站立指令"""
    def __init__(
        self,
        channel: str = "lo",
        control_frequency: float = 50.0,
        kp_scale_factor: float = 1.0,
        safety_profile: str = "default"
    ):
        self.channel = channel
        self.control_frequency = control_frequency
        self.kp_scale_factor = kp_scale_factor
        self.safety_profile = safety_profile
        self.control_dt = 1.0 / self.control_frequency
        
        self.running = Event()
        
        print(f"🚀 初始化 Dummy DDS 控制器")
        print(f"   模式: 真实机器人/lo模式 (通道: {channel})")
        print(f"   控制频率: {control_frequency} Hz")

        # 1. 初始化DDS
        self._setup_dds()

        # 2. 初始化CEM控制器后端
        self.cem_controller = self._setup_cem_controller()
            
        # 3. 定义目标姿态
        self.target_pos = self._define_target_pose()
        
        print("✅ Dummy控制器初始化完成")

    def _setup_dds(self):
        """设置DDS通信"""
        if self.channel == "lo":
            print("   使用lo接口 (domain_id=1)")
            ChannelFactoryInitialize(1, "lo")
        else:
            print(f"   使用真实网络接口: {self.channel} (domain_id=0)")
            ChannelFactoryInitialize(0, self.channel)

    def _setup_cem_controller(self):
        """设置并返回一个CEM控制器实例"""
        print(f"🤖 设置 CEM 控制模式 (通道: {self.channel})...")
        controller = CEMSDKController(
            simulate=False,
            robot_config=None,
            xml_path="g1_model/g1_lab.xml",
            vicon_required=(self.channel != "lo"),
            lo_mode=(self.channel == "lo"),
            kp_scale_factor=self.kp_scale_factor,
            safety_profile=self.safety_profile
        )
        print("✅ CEM控制器设置完成")
        return controller
    
    def _define_target_pose(self) -> np.ndarray:
        """定义并返回目标关节位置"""
        # 创建一个包含27个主动身体关节的目标数组
        target_q = np.zeros(NUM_ACTIVE_BODY_JOINTS)
        
        # 根据G1.py中的mujoco_index设置膝关节微屈
        # left_knee_joint (mujoco_index: 3)
        # right_knee_joint (mujoco_index: 9)
        target_q[3] = 0.1  # 左膝
        target_q[9] = 0.1  # 右膝

        print(f"🎯 目标姿态已设定 (双膝微屈0.1 rad)")
        return target_q

    def run(self):
        """运行主控制循环"""
        print(f"🎬 启动Dummy控制器主循环")
        self.running.set()
        
        # 在启动前，等待有效的机器人状态，确保DDS和Vicon已连接
        print("🔄 等待有效的初始机器人状态...")
        initial_state = None
        while initial_state is None and self.running.is_set():
            initial_state = self.cem_controller.get_robot_state()
            if initial_state is None:
                if not self.running.is_set(): break
                print("  ...仍在等待, 0.5s后重试...")
                time.sleep(0.5)

        if not self.running.is_set():
            self.stop()
            return
            
        print("✅ 成功获取初始状态，开始发送控制指令...")

        try:
            while self.running.is_set():
                # 检查机器人状态是否有效
                state = self.cem_controller.get_robot_state()
                if state is None:
                    print("❌ 失去机器人状态(Vicon或DDS超时)。为安全起见，正在停止。")
                    print("   发送阻尼命令...")
                    for _ in range(5):
                        self.cem_controller.damping_motor_cmd()
                        time.sleep(0.01)
                    self.stop()
                    break

                # 以固定频率发送目标姿态指令
                self.cem_controller.send_motor_command(
                    time=time.time(), 
                    pd_targets=self.target_pos
                )
                
                time.sleep(self.control_dt)

        except KeyboardInterrupt:
            print("\n🛑 收到中断信号...")
        
        finally:
            print("   发送最终阻尼命令...")
            for _ in range(5):
                self.cem_controller.damping_motor_cmd()
                time.sleep(0.01)
            self.stop()

    def stop(self):
        """停止控制器"""
        print("🛑 停止 Dummy 控制器...")
        self.running.clear()
        # cem_controller会在父进程退出时自动清理DDS资源
        print("✅ Dummy控制器已停止")


def main():
    parser = argparse.ArgumentParser(description="Dummy DDS G1 控制器")
    parser.add_argument(
        "--channel",
        type=str,
        default="lo",
        help="DDS通道：'lo'表示本地回环，其他值为网络接口名"
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=100.0,
        help="控制频率 (Hz)"
    )
    parser.add_argument(
        "--kp_scale",
        type=float,
        default=1.0,
        help="全局Kp增益缩放因子 (0.0-1.0)"
    )
    parser.add_argument(
        "--safety_profile",
        type=str,
        default="default",
        choices=["default", "conservative"],
        help="选择安全层配置文件 ('default' 或 'conservative')"
    )
    
    args = parser.parse_args()
    
    controller = None
    def signal_handler(sig, frame):
        print("\n🛑 收到停止信号")
        if controller:
            controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    controller = DummyController(
        channel=args.channel,
        control_frequency=args.frequency,
        kp_scale_factor=args.kp_scale,
        safety_profile=args.safety_profile
    )
    
    controller.run()


if __name__ == "__main__":
    main()
