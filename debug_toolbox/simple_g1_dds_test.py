#!/usr/bin/env python3
"""
简化的 G1 DDS 直接测试 - 无中间层
直接模拟策略控制器与SDK的DDS通信

架构：
模拟策略 ←→ DDS ←→ 模拟机器人
（单进程，直接通信）
"""

import time
import numpy as np
import threading
from typing import Optional

# G1 DDS imports
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_, unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC


class SimpleG1PolicyController:
    """简化的G1策略控制器 - 直接DDS通信"""
    
    def __init__(self, control_freq: float = 100.0, init_dds: bool = True):
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        
        print("🤖 初始化简化 G1 策略控制器")
        print(f"   控制频率: {control_freq} Hz")
        
        # 初始化DDS（如果需要）
        if init_dds:
            ChannelFactoryInitialize(1, "lo")
        
        # CRC计算器
        self.crc = CRC()
        
        # 状态订阅
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self._state_handler, 10)
        
        # 命令发布
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()
        
        # 状态管理
        self.last_state = None
        self.state_count = 0
        self.cmd_count = 0
        self.running = False
        
        # 初始化命令
        self._init_command()
        
        print("✅ G1 策略控制器初始化完成")
    
    def _init_command(self):
        """初始化控制命令"""
        self.cmd.mode_pr = 0
        self.cmd.mode_machine = 0
        
        # 站立姿态的关节目标
        stand_pos = np.array([
            # 左腿
            0.0, 0.0, 0.0, -0.3, 0.3, 0.0,
            # 右腿  
            0.0, 0.0, 0.0, -0.3, 0.3, 0.0,
            # 腰部
            0.0,
            # 左臂
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # 右臂
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        
        # PD增益
        kp_values = [60, 60, 60, 100, 40, 40] * 2 + [60] + [40] * 14  # 29个关节
        kd_values = [1, 1, 1, 2, 1, 1] * 2 + [1] + [1] * 14
        
        # 设置所有电机命令
        for i in range(29):
            if i == 13 or i == 14:  # 不存在的腰部关节
                self.cmd.motor_cmd[i].mode = 1
                self.cmd.motor_cmd[i].q = 0.0
                self.cmd.motor_cmd[i].kp = 0.0
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kd = 0.0
                self.cmd.motor_cmd[i].tau = 0.0
            else:
                joint_idx = i if i < 13 else i - 2  # 跳过不存在的关节
                if joint_idx < len(stand_pos):
                    self.cmd.motor_cmd[i].mode = 1
                    self.cmd.motor_cmd[i].q = stand_pos[joint_idx]
                    self.cmd.motor_cmd[i].kp = kp_values[i] if i < len(kp_values) else 40.0
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kd = kd_values[i] if i < len(kd_values) else 1.0
                    self.cmd.motor_cmd[i].tau = 0.0
    
    def _state_handler(self, msg: LowState_):
        """处理接收到的状态"""
        self.last_state = msg
        self.state_count += 1
        
        if self.state_count % 100 == 0:  # 每秒打印一次
            print(f"📥 已接收 {self.state_count} 个状态消息")
    
    def _update_control(self):
        """更新控制命令（可以在这里添加策略逻辑）"""
        if self.last_state is None:
            return
        
        # 简单的站立控制 + 轻微扰动
        time_factor = time.time() * 0.5
        
        for i in range(29):
            if i == 13 or i == 14:  # 跳过不存在的关节
                continue
                
            # 添加轻微的正弦扰动
            if i < 12:  # 腿部关节
                disturbance = 0.05 * np.sin(time_factor + i * 0.1)
                self.cmd.motor_cmd[i].q += disturbance
    
    def _send_command(self):
        """发送控制命令"""
        self._update_control()
        
        # 计算CRC并发送
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.cmd_pub.Write(self.cmd)
        
        self.cmd_count += 1
        
        if self.cmd_count % 100 == 0:  # 每秒打印一次
            print(f"📤 已发送 {self.cmd_count} 个控制命令")
    
    def run(self, duration: float = 10.0):
        """运行控制循环"""
        print(f"🚀 开始运行 G1 控制循环")
        print(f"   持续时间: {duration} 秒")
        print(f"   控制频率: {self.control_freq} Hz")
        
        self.running = True
        start_time = time.time()
        next_control = start_time
        
        try:
            while time.time() - start_time < duration and self.running:
                current_time = time.time()
                
                # 按频率发送控制命令
                if current_time >= next_control:
                    self._send_command()
                    next_control += self.dt
                
                # 短暂休眠
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n🛑 接收到中断信号")
        finally:
            self.stop()
    
    def stop(self):
        """停止控制器"""
        self.running = False
        print(f"\n📊 === 简化 G1 控制测试结果 ===")
        print(f"📥 接收状态数: {self.state_count}")
        print(f"📤 发送命令数: {self.cmd_count}")
        
        if self.state_count > 0 and self.cmd_count > 0:
            print("✅ DDS 双向通信成功!")
            print(f"📈 状态接收频率: {self.state_count/10:.1f} Hz")
            print(f"📈 命令发送频率: {self.cmd_count/10:.1f} Hz")
        else:
            print("❌ DDS 通信失败")


class SimpleG1Robot:
    """简化的G1机器人模拟器 - 直接DDS通信"""
    
    def __init__(self, publish_freq: float = 100.0, init_dds: bool = True):
        self.publish_freq = publish_freq
        self.dt = 1.0 / publish_freq
        
        print("🤖 初始化简化 G1 机器人模拟器")
        
        # 初始化DDS工厂（如果需要）
        if init_dds:
            ChannelFactoryInitialize(1, "lo")
        
        # 状态发布
        self.state = unitree_hg_msg_dds__LowState_()
        self.state_pub = ChannelPublisher("rt/lowstate", LowState_)
        self.state_pub.Init()
        
        # 命令订阅
        self.cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.cmd_sub.Init(self._cmd_handler, 10)
        
        # 状态管理
        self.publish_count = 0
        self.cmd_received_count = 0
        self.running = False
        
        # 初始化状态
        self._init_state()
        
        print("✅ G1 机器人模拟器初始化完成")
    
    def _init_state(self):
        """初始化机器人状态"""
        self.state.tick = 0
        self.state.mode_pr = 0
        self.state.mode_machine = 0
        
        # 设置电机状态（只设置基本字段，避免复杂字段）
        for i in range(35):  # G1 有35个电机状态槽位
            self.state.motor_state[i].q = 0.0
            self.state.motor_state[i].dq = 0.0
            self.state.motor_state[i].tau_est = 0.0
            self.state.motor_state[i].mode = 1
            # 不设置 temperature 字段，让它保持默认值
    
    def _cmd_handler(self, msg: LowCmd_):
        """处理接收到的命令"""
        self.cmd_received_count += 1
        
        if self.cmd_received_count % 100 == 0:
            print(f"📥 机器人已接收 {self.cmd_received_count} 个控制命令")
        
        # 简单模拟：将命令作为状态反馈（只更新基本字段）
        for i in range(min(29, len(msg.motor_cmd))):
            try:
                self.state.motor_state[i].q = msg.motor_cmd[i].q
                # 添加一些噪声
                self.state.motor_state[i].q += np.random.normal(0, 0.001)
            except:
                pass
    
    def _publish_state(self):
        """发布机器人状态"""
        self.state.tick = int(time.time() * 1000) % 1000000
        self.state_pub.Write(self.state)
        self.publish_count += 1
        
        if self.publish_count % 100 == 0:
            print(f"📤 机器人已发布 {self.publish_count} 个状态")
    
    def run(self, duration: float = 10.0):
        """运行机器人模拟"""
        print(f"🚀 开始运行 G1 机器人模拟")
        
        self.running = True
        start_time = time.time()
        next_publish = start_time
        
        try:
            while time.time() - start_time < duration and self.running:
                current_time = time.time()
                
                # 按频率发布状态
                if current_time >= next_publish:
                    self._publish_state()
                    next_publish += self.dt
                
                # 短暂休眠
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n🛑 机器人模拟中断")
        finally:
            self.stop()
    
    def stop(self):
        """停止机器人模拟"""
        self.running = False
        print(f"\n📊 === G1 机器人模拟结果 ===")
        print(f"📤 发布状态数: {self.publish_count}")
        print(f"📥 接收命令数: {self.cmd_received_count}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="简化的 G1 DDS 直接测试")
    parser.add_argument("--mode", choices=["controller", "robot", "both"], default="both",
                        help="运行模式: controller(策略), robot(机器人), both(双向)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="测试持续时间（秒）")
    parser.add_argument("--frequency", type=float, default=100.0,
                        help="运行频率（Hz）")
    
    args = parser.parse_args()
    
    print("🧪 简化 G1 DDS 直接测试")
    print(f"   模式: {args.mode}")
    print(f"   持续时间: {args.duration} 秒")
    print(f"   频率: {args.frequency} Hz")
    print("=" * 50)
    
    if args.mode == "controller":
        controller = SimpleG1PolicyController(args.frequency)
        controller.run(args.duration)
        
    elif args.mode == "robot":
        robot = SimpleG1Robot(args.frequency)
        robot.run(args.duration)
        
    elif args.mode == "both":
        # 双向测试：在不同线程中运行
        # 先初始化DDS工厂（只需要一次）
        ChannelFactoryInitialize(1, "lo")
        
        # 创建时跳过DDS初始化（因为已经初始化过了）
        controller = SimpleG1PolicyController(args.frequency, init_dds=False)
        robot = SimpleG1Robot(args.frequency, init_dds=False)
        
        # 启动机器人线程
        robot_thread = threading.Thread(target=robot.run, args=(args.duration,))
        robot_thread.start()
        
        # 等待一点时间让机器人启动
        time.sleep(0.5)
        
        # 运行控制器
        controller.run(args.duration)
        
        # 等待机器人线程结束
        robot.stop()
        robot_thread.join()
        
        print("\n🎉 === 双向测试完成 ===")


if __name__ == "__main__":
    main() 