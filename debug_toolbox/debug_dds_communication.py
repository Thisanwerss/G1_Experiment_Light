#!/usr/bin/env python3
"""
DDS 通信调试脚本 - G1 机器人专用版本
用于快速诊断 DDS 通信问题和队列拥塞
与 run_policy_sdk_bridge.py 兼容，使用正确的消息类型
"""

import time
import numpy as np
from typing import Optional

# DDS 相关导入 - 使用 G1 专用的消息类型
try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
    # G1 使用 unitree_hg 消息
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_, unitree_hg_msg_dds__LowCmd_
    print("✅ G1 DDS 导入成功")
except ImportError as e:
    print(f"❌ DDS 导入失败: {e}")
    exit(1)


class DDSCommunicationTester:
    """DDS 通信测试器 - G1 专用版本"""
    
    def __init__(self):
        print("🔧 初始化 G1 DDS 通信测试器...")
        self.setup_dds()
        
        # 统计
        self.publish_count = 0
        self.receive_count = 0
        self.error_count = 0
        
    def setup_dds(self):
        """设置 DDS 通信"""
        try:
            # 初始化 DDS
            print("📡 初始化 DDS 工厂...")
            ChannelFactoryInitialize(1, "lo")
            
            # 状态发布者 - 发布 G1 格式的状态
            print("📤 设置 G1 状态发布者...")
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.low_state_pub = ChannelPublisher("rt/lowstate", LowState_)
            self.low_state_pub.Init()
            
            # 控制命令订阅者 - 接收 G1 格式的命令
            print("📥 设置 G1 控制命令订阅者...")
            self.received_cmd = None
            self.last_cmd_time = 0.0
            self.low_cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
            self.low_cmd_sub.Init(self._cmd_handler, 10)
            
            print("✅ G1 DDS 通信设置完成")
            
        except Exception as e:
            print(f"❌ DDS 设置失败: {e}")
            raise
    
    def _cmd_handler(self, msg: LowCmd_):
        """处理控制命令"""
        self.received_cmd = msg
        self.last_cmd_time = time.time()
        self.receive_count += 1
        
        # 打印详细信息
        print(f"📥 收到 G1 控制命令 #{self.receive_count}")
        print(f"   时间戳: {self.last_cmd_time:.3f}")
        print(f"   命令头: {msg.head if hasattr(msg, 'head') else 'N/A'}")
        print(f"   电机命令数量: {len(msg.motor_cmd) if hasattr(msg, 'motor_cmd') else 0}")
        
        # 显示前几个关节的控制命令
        if hasattr(msg, 'motor_cmd') and len(msg.motor_cmd) > 0:
            print("   前5个关节目标位置:")
            for i in range(min(5, len(msg.motor_cmd))):
                print(f"     关节{i}: q={msg.motor_cmd[i].q:.3f}, kp={msg.motor_cmd[i].kp:.1f}")
    
    def publish_realistic_g1_state(self) -> bool:
        """发布更真实的 G1 状态"""
        try:
            # 设置基本字段
            self.low_state.tick = int(time.time() * 1000) % 1000000
            
            # 设置模式
            self.low_state.mode_pr = 0
            self.low_state.mode_machine = 0
            
            # 设置身体电机状态（按照G1标准：29个DDS索引）
            for dds_idx in range(29):
                try:
                    if dds_idx == 13 or dds_idx == 14:
                        # waist_roll 和 waist_pitch 在G1中不存在，设置为0
                        self.low_state.motor_state[dds_idx].q = 0.0
                        self.low_state.motor_state[dds_idx].dq = 0.0
                        self.low_state.motor_state[dds_idx].tau_est = 0.0
                        self.low_state.motor_state[dds_idx].mode = 1
                        self.low_state.motor_state[dds_idx].temperature = 25
                        continue
                    
                    # 模拟站立姿态的关节位置
                    if dds_idx < 6:  # 左腿
                        if dds_idx == 3:  # 左膝盖
                            self.low_state.motor_state[dds_idx].q = -0.3
                        elif dds_idx == 4:  # 左脚踝俯仰
                            self.low_state.motor_state[dds_idx].q = 0.3
                        else:
                            self.low_state.motor_state[dds_idx].q = 0.0
                    elif dds_idx < 12:  # 右腿
                        if dds_idx == 9:  # 右膝盖
                            self.low_state.motor_state[dds_idx].q = -0.3
                        elif dds_idx == 10:  # 右脚踝俯仰
                            self.low_state.motor_state[dds_idx].q = 0.3
                        else:
                            self.low_state.motor_state[dds_idx].q = 0.0
                    else:  # 腰部和手臂
                        self.low_state.motor_state[dds_idx].q = 0.0
                    
                    # 基本的状态信息
                    self.low_state.motor_state[dds_idx].dq = 0.0
                    self.low_state.motor_state[dds_idx].tau_est = 0.0
                    self.low_state.motor_state[dds_idx].mode = 1
                    self.low_state.motor_state[dds_idx].temperature = 25
                    
                except (IndexError, AttributeError) as e:
                    print(f"⚠️ 跳过电机索引 {dds_idx}: {e}")
                    break
            
            # 发布状态
            self.low_state_pub.Write(self.low_state)
            self.publish_count += 1
            return True
            
        except Exception as e:
            print(f"❌ G1 状态发布失败: {e}")
            self.error_count += 1
            return False
    
    def check_received_cmd(self) -> Optional[dict]:
        """检查接收到的命令"""
        if self.received_cmd is not None:
            # 提取基本信息
            cmd_info = {
                'received_at': self.last_cmd_time,
                'age': time.time() - self.last_cmd_time,
                'motor_cmd_count': len(self.received_cmd.motor_cmd) if hasattr(self.received_cmd, 'motor_cmd') else 0,
                'head': list(self.received_cmd.head) if hasattr(self.received_cmd, 'head') else None,
                'level_flag': self.received_cmd.level_flag if hasattr(self.received_cmd, 'level_flag') else None
            }
            
            # 清除命令（避免重复处理）
            self.received_cmd = None
            return cmd_info
        
        return None
    
    def run_basic_test(self, duration: float = 10.0, frequency: float = 10.0):
        """运行基本通信测试"""
        print(f"🚀 开始 G1 DDS 通信测试")
        print(f"   持续时间: {duration}s")
        print(f"   发布频率: {frequency} Hz")
        print(f"   测试模式: 发布 G1 LowState → 接收 G1 LowCmd")
        
        period = 1.0 / frequency
        start_time = time.time()
        next_publish = start_time
        
        try:
            while time.time() - start_time < duration:
                current_time = time.time()
                
                # 发布状态
                if current_time >= next_publish:
                    success = self.publish_realistic_g1_state()
                    if success:
                        print(f"📤 G1 状态发布 #{self.publish_count} (时间: {current_time:.3f})")
                    else:
                        print(f"❌ G1 状态发布失败")
                    
                    next_publish += period
                
                # 检查接收到的命令（非阻塞）
                cmd_info = self.check_received_cmd()
                if cmd_info:
                    print(f"🎉 成功接收到 G1 控制命令! #{self.receive_count}")
                    print(f"   详细信息: {cmd_info}")
                
                # 短暂休眠避免占用过多 CPU
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n🛑 测试中断")
        
        finally:
            self.print_test_results()
    
    def print_test_results(self):
        """打印测试结果"""
        print(f"\n📊 === G1 DDS 通信测试结果 ===")
        print(f"📤 发布次数: {self.publish_count}")
        print(f"📥 接收次数: {self.receive_count}")
        print(f"❌ 错误次数: {self.error_count}")
        print(f"📈 发布成功率: {(self.publish_count/(self.publish_count+self.error_count)*100):.1f}%" if (self.publish_count+self.error_count) > 0 else "N/A")
        print(f"🔄 通信成功: {'是' if self.receive_count > 0 else '否'}")
        
        if self.receive_count > 0:
            print("✅ DDS 双向通信正常 - G1 消息格式兼容!")
        else:
            print("❌ 没有接收到控制命令回传")
            print("   可能原因:")
            print("   1. SDK 桥接器未启动")
            print("   2. 消息类型不匹配") 
            print("   3. DDS 通道配置问题")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="G1 DDS 通信调试器")
    parser.add_argument("--duration", type=float, default=10.0, help="测试持续时间（秒）")
    parser.add_argument("--frequency", type=float, default=5.0, help="测试频率（Hz）")
    
    args = parser.parse_args()
    
    print("🧪 G1 DDS 通信调试器启动")
    print("此脚本专门用于调试与 G1 机器人的 DDS 通信")
    print("确保 run_policy_sdk_bridge.py 正在运行")
    print(f"测试参数: 持续={args.duration}s, 频率={args.frequency}Hz")
    
    try:
        tester = DDSCommunicationTester()
        tester.run_basic_test(duration=args.duration, frequency=args.frequency)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 