#!/usr/bin/env python3
"""
DDS 消息监听器和验证器
专用于G1机器人DDS通信测试和调试

使用方式:
python dds_listener.py --channel lo
python dds_listener.py --channel <network_interface>
"""

import time
import argparse
import signal
import sys
import threading
from threading import Thread, Event
from collections import deque
from typing import Dict, List, Optional
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_, unitree_hg_msg_dds__HandState_

from sdk_controller.robots.G1 import *


# HG系列DDS主题
HG_TOPIC_LOWCMD = "rt/lowcmd"
HG_TOPIC_LOWSTATE = "rt/lowstate"
HG_TOPIC_HANDCMD = "rt/handcmd"
HG_TOPIC_HANDSTATE = "rt/handstate"


class DDSMessageStats:
    """DDS消息统计信息"""
    
    def __init__(self, topic_name: str, expected_frequency: float = 100.0):
        self.topic_name = topic_name
        self.expected_frequency = expected_frequency
        
        # 消息计数和时间戳
        self.message_count = 0
        self.last_timestamps = deque(maxlen=100)  # 保存最近100个时间戳
        self.last_message_time = 0.0
        
        # 频率统计
        self.current_frequency = 0.0
        self.avg_frequency = 0.0
        self.frequency_deviation = 0.0
        
        # 消息验证统计
        self.valid_messages = 0
        self.invalid_messages = 0
        self.validation_errors = []
        
        # 线程安全锁
        self.lock = threading.Lock()
    
    def update_message(self, timestamp: float, is_valid: bool = True, error_msg: str = ""):
        """更新消息统计"""
        with self.lock:
            self.message_count += 1
            self.last_timestamps.append(timestamp)
            self.last_message_time = timestamp
            
            if is_valid:
                self.valid_messages += 1
            else:
                self.invalid_messages += 1
                if error_msg:
                    self.validation_errors.append(f"[{timestamp:.3f}] {error_msg}")
                    # 只保留最近50个错误
                    if len(self.validation_errors) > 50:
                        self.validation_errors.pop(0)
            
            # 计算频率
            self._calculate_frequency()
    
    def _calculate_frequency(self):
        """计算消息频率"""
        if len(self.last_timestamps) < 2:
            return
        
        # 计算当前频率（基于最近两条消息）
        if len(self.last_timestamps) >= 2:
            dt = self.last_timestamps[-1] - self.last_timestamps[-2]
            if dt > 0:
                self.current_frequency = 1.0 / dt
        
        # 计算平均频率（基于最近的所有消息）
        if len(self.last_timestamps) >= 10:
            time_span = self.last_timestamps[-1] - self.last_timestamps[0]
            if time_span > 0:
                self.avg_frequency = (len(self.last_timestamps) - 1) / time_span
                self.frequency_deviation = abs(self.avg_frequency - self.expected_frequency)
    
    def get_stats_dict(self) -> Dict:
        """获取统计信息字典"""
        with self.lock:
            return {
                'topic': self.topic_name,
                'message_count': self.message_count,
                'valid_messages': self.valid_messages,
                'invalid_messages': self.invalid_messages,
                'current_frequency': self.current_frequency,
                'avg_frequency': self.avg_frequency,
                'expected_frequency': self.expected_frequency,
                'frequency_deviation': self.frequency_deviation,
                'last_message_time': self.last_message_time,
                'recent_errors': self.validation_errors[-5:] if self.validation_errors else []
            }


class G1DDSValidator:
    """G1 DDS消息验证器"""
    
    @staticmethod
    def validate_low_cmd(msg: LowCmd_) -> tuple:
        """验证LowCmd消息"""
        try:
            # 检查消息基本结构
            if not hasattr(msg, 'motor_cmd'):
                return False, "缺少motor_cmd字段"
            
            # 检查电机命令数量
            if len(msg.motor_cmd) != 35:
                return False, f"motor_cmd长度错误: {len(msg.motor_cmd)} != 35"
            
            # 检查关键身体关节的命令
            active_joint_count = 0
            for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
                if mj_idx < NUM_ACTIVE_BODY_JOINTS and dds_idx < len(msg.motor_cmd):
                    motor_cmd = msg.motor_cmd[dds_idx]
                    
                    # 检查关节命令有效性
                    if (hasattr(motor_cmd, 'q') and hasattr(motor_cmd, 'kp') and 
                        hasattr(motor_cmd, 'dq') and hasattr(motor_cmd, 'kd')):
                        
                        # 检查数值范围
                        if abs(motor_cmd.q) > 10.0:  # 关节位置不应超过±10弧度
                            return False, f"关节{dds_idx}位置超限: {motor_cmd.q}"
                        
                        if motor_cmd.kp < 0 or motor_cmd.kp > 500.0:  # Kp增益范围检查
                            return False, f"关节{dds_idx} Kp增益异常: {motor_cmd.kp}"
                        
                        active_joint_count += 1
                    else:
                        return False, f"关节{dds_idx}命令字段不完整"
            
            if active_joint_count != NUM_ACTIVE_BODY_JOINTS:
                return False, f"活动关节数量错误: {active_joint_count} != {NUM_ACTIVE_BODY_JOINTS}"
            
            return True, ""
            
        except Exception as e:
            return False, f"验证异常: {str(e)}"
    
    @staticmethod
    def validate_low_state(msg: LowState_) -> tuple:
        """验证LowState消息"""
        try:
            # 检查消息基本结构
            if not hasattr(msg, 'motor_state'):
                return False, "缺少motor_state字段"
            
            # 检查电机状态数量
            if len(msg.motor_state) != 35:
                return False, f"motor_state长度错误: {len(msg.motor_state)} != 35"
            
            # 检查IMU数据
            if hasattr(msg, 'imu_state'):
                if (hasattr(msg.imu_state, 'quaternion') and 
                    len(msg.imu_state.quaternion) == 4):
                    # 检查四元数的模长
                    q = np.array(msg.imu_state.quaternion)
                    norm = np.linalg.norm(q)
                    if abs(norm - 1.0) > 0.1:  # 四元数模长应接近1
                        return False, f"四元数模长异常: {norm}"
            
            # 检查关键身体关节的状态
            active_joint_count = 0
            for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
                if mj_idx < NUM_ACTIVE_BODY_JOINTS and dds_idx < len(msg.motor_state):
                    motor_state = msg.motor_state[dds_idx]
                    
                    # 检查关节状态有效性
                    if (hasattr(motor_state, 'q') and hasattr(motor_state, 'dq') and 
                        hasattr(motor_state, 'tau_est')):
                        
                        # 检查数值合理性
                        if abs(motor_state.q) > 10.0:
                            return False, f"关节{dds_idx}位置超限: {motor_state.q}"
                        
                        if abs(motor_state.dq) > 50.0:  # 关节速度不应超过±50 rad/s
                            return False, f"关节{dds_idx}速度超限: {motor_state.dq}"
                        
                        active_joint_count += 1
                    else:
                        return False, f"关节{dds_idx}状态字段不完整"
            
            return True, ""
            
        except Exception as e:
            return False, f"验证异常: {str(e)}"
    
    @staticmethod
    def validate_hand_cmd(msg: HandCmd_) -> tuple:
        """验证HandCmd消息"""
        try:
            # 检查消息基本结构
            if not hasattr(msg, 'motor_cmd'):
                return False, "缺少motor_cmd字段"
            
            # 手部控制命令应该有14个（左右手各7个关节）
            if len(msg.motor_cmd) != 14:
                return False, f"hand motor_cmd长度错误: {len(msg.motor_cmd)} != 14"
            
            return True, ""
            
        except Exception as e:
            return False, f"验证异常: {str(e)}"
    
    @staticmethod
    def validate_hand_state(msg: HandState_) -> tuple:
        """验证HandState消息"""
        try:
            # 检查消息基本结构
            if not hasattr(msg, 'motor_state'):
                return False, "缺少motor_state字段"
            
            # 手部状态应该有14个（左右手各7个关节）
            if len(msg.motor_state) != 14:
                return False, f"hand motor_state长度错误: {len(msg.motor_state)} != 14"
            
            return True, ""
            
        except Exception as e:
            return False, f"验证异常: {str(e)}"


class G1DDSListener:
    """G1 DDS消息监听器"""
    
    def __init__(self, channel: str = "lo", domain_id: int = 1):
        self.channel = channel
        self.domain_id = domain_id
        self.running = Event()
        
        # 统计对象
        self.stats = {
            'lowcmd': DDSMessageStats(HG_TOPIC_LOWCMD, 100.0),
            'lowstate': DDSMessageStats(HG_TOPIC_LOWSTATE, 100.0),
            'handcmd': DDSMessageStats(HG_TOPIC_HANDCMD, 100.0),
            'handstate': DDSMessageStats(HG_TOPIC_HANDSTATE, 100.0)
        }
        
        self.validator = G1DDSValidator()
        
        print(f"🎧 初始化G1 DDS监听器")
        print(f"   通道: {channel}")
        print(f"   域ID: {domain_id}")
        
        # 初始化DDS
        self._setup_dds()
        
        # 启动统计线程
        self.stats_thread = Thread(target=self._stats_reporter, daemon=True)
        self.stats_thread.start()
        
        print("✅ G1 DDS监听器初始化完成")
    
    def _setup_dds(self):
        """设置DDS连接"""
        print("🌐 初始化DDS连接...")
        
        # 根据通道决定domain_id
        if self.channel == "lo":
            ChannelFactoryInitialize(1, "lo")
            print("   使用lo接口 (domain_id=1)")
        else:
            ChannelFactoryInitialize(0, self.channel)
            print(f"   使用真实网络接口: {self.channel} (domain_id=0)")
        
        # 创建DDS订阅者
        self.lowcmd_sub = ChannelSubscriber(HG_TOPIC_LOWCMD, LowCmd_)
        self.lowstate_sub = ChannelSubscriber(HG_TOPIC_LOWSTATE, LowState_)
        self.handcmd_sub = ChannelSubscriber(HG_TOPIC_HANDCMD, HandCmd_)
        self.handstate_sub = ChannelSubscriber(HG_TOPIC_HANDSTATE, HandState_)
        
        # 初始化订阅者回调
        self.lowcmd_sub.Init(self._lowcmd_handler, 10)
        self.lowstate_sub.Init(self._lowstate_handler, 10)
        self.handcmd_sub.Init(self._handcmd_handler, 10)
        self.handstate_sub.Init(self._handstate_handler, 10)
        
        print("📡 DDS订阅者已设置")
    
    def _lowcmd_handler(self, msg: LowCmd_):
        """LowCmd消息处理器"""
        timestamp = time.time()
        is_valid, error_msg = self.validator.validate_low_cmd(msg)
        self.stats['lowcmd'].update_message(timestamp, is_valid, error_msg)
        
        if not is_valid:
            print(f"❌ LowCmd验证失败: {error_msg}")
    
    def _lowstate_handler(self, msg: LowState_):
        """LowState消息处理器"""
        timestamp = time.time()
        is_valid, error_msg = self.validator.validate_low_state(msg)
        self.stats['lowstate'].update_message(timestamp, is_valid, error_msg)
        
        if not is_valid:
            print(f"❌ LowState验证失败: {error_msg}")
    
    def _handcmd_handler(self, msg: HandCmd_):
        """HandCmd消息处理器"""
        timestamp = time.time()
        is_valid, error_msg = self.validator.validate_hand_cmd(msg)
        self.stats['handcmd'].update_message(timestamp, is_valid, error_msg)
        
        if not is_valid:
            print(f"❌ HandCmd验证失败: {error_msg}")
    
    def _handstate_handler(self, msg: HandState_):
        """HandState消息处理器"""
        timestamp = time.time()
        is_valid, error_msg = self.validator.validate_hand_state(msg)
        self.stats['handstate'].update_message(timestamp, is_valid, error_msg)
        
        if not is_valid:
            print(f"❌ HandState验证失败: {error_msg}")
    
    def _stats_reporter(self):
        """统计信息报告线程"""
        while self.running.is_set():
            time.sleep(5.0)  # 每5秒报告一次
            self._print_stats()
    
    def _print_stats(self):
        """打印统计信息"""
        print("\n" + "="*80)
        print(f"📊 G1 DDS消息统计 [{time.strftime('%H:%M:%S')}]")
        print("="*80)
        
        for name, stats in self.stats.items():
            stats_dict = stats.get_stats_dict()
            
            # 计算频率状态指示器
            freq_status = "🟢"  # 绿色：正常
            if abs(stats_dict['frequency_deviation']) > 10.0:
                freq_status = "🟡"  # 黄色：轻微偏差
            if abs(stats_dict['frequency_deviation']) > 20.0:
                freq_status = "🔴"  # 红色：严重偏差
            
            # 计算错误率
            total_msgs = stats_dict['message_count']
            error_rate = 0.0
            if total_msgs > 0:
                error_rate = (stats_dict['invalid_messages'] / total_msgs) * 100
            
            print(f"{freq_status} {stats_dict['topic'].upper():<12} | "
                  f"消息: {total_msgs:>6} | "
                  f"频率: {stats_dict['avg_frequency']:>6.1f}Hz "
                  f"(目标: {stats_dict['expected_frequency']:.0f}Hz) | "
                  f"错误率: {error_rate:>5.1f}% | "
                  f"最近: {time.time() - stats_dict['last_message_time']:>5.1f}s前")
            
            # 显示最近的错误
            if stats_dict['recent_errors']:
                for error in stats_dict['recent_errors']:
                    print(f"     ⚠️  {error}")
        
        print("="*80)
    
    def run(self):
        """运行监听器"""
        print("🎬 启动DDS监听器...")
        print("💡 开始监听G1 DDS消息，按Ctrl+C停止")
        
        self.running.set()
        
        try:
            while self.running.is_set():
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n🛑 收到中断信号...")
        
        finally:
            self.stop()
    
    def stop(self):
        """停止监听器"""
        print("🛑 停止DDS监听器...")
        self.running.clear()
        
        # 打印最终统计
        print("\n📈 最终统计报告:")
        self._print_stats()
        
        print("✅ DDS监听器已停止")


def signal_handler(sig, frame):
    """信号处理器"""
    print("\n🛑 收到停止信号")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="G1 DDS消息监听器和验证器")
    parser.add_argument(
        "--channel",
        type=str,
        default="lo",
        help="DDS通道：'lo'表示本地回环，其他值为网络接口名"
    )
    parser.add_argument(
        "--domain_id",
        type=int,
        default=1,
        help="DDS 域 ID (已弃用，自动根据channel确定)"
    )
    
    args = parser.parse_args()
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 创建并运行监听器
    listener = G1DDSListener(
        channel=args.channel,
        domain_id=args.domain_id
    )
    
    listener.run()


if __name__ == "__main__":
    main() 