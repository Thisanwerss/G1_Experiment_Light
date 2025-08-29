#!/usr/bin/env python3
"""
策略 ZeroMQ 中继器 - 重新设计版本
承担变频功能：50Hz barrier 同步 + 100Hz 控制发送

功能：
1. 以 50Hz 的 lock-step barrier 与策略通信（发送状态、接收控制）
2. 将 50Hz 策略控制插值为 100Hz 控制序列
3. 以 100Hz 向 SDK 桥接器发送单个控制目标
4. 从 DDS 获取状态并以 50Hz 发送回策略
"""

import argparse
import time
import pickle
import struct
from typing import Optional, Dict, Any, List
import numpy as np
import zmq

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


class PolicyZMQRelay:
    """策略 ZeroMQ 中继器 - 变频版本"""
    
    def __init__(
        self,
        zmq_policy_recv_port: int = 5556,  # 从策略接收控制
        zmq_policy_send_port: int = 5555,  # 向策略发送状态
        zmq_bridge_port: int = 5557,       # 向 SDK 桥接器发送控制目标
        use_dds_state: bool = True,        # 是否从 DDS 获取状态
        policy_freq: float = 50.0,         # 策略频率 (50Hz)
        control_freq: float = 100.0        # 控制频率 (100Hz)
    ):
        """
        初始化中继器
        
        Args:
            zmq_policy_recv_port: 策略控制接收端口（PULL）
            zmq_policy_send_port: 策略状态发送端口（PUSH）
            zmq_bridge_port: 桥接器命令发送端口（PUSH）
            use_dds_state: 是否从 DDS 获取状态
            policy_freq: 策略频率 (50Hz)
            control_freq: 控制频率 (100Hz)
        """
        self.zmq_policy_recv_port = zmq_policy_recv_port
        self.zmq_policy_send_port = zmq_policy_send_port
        self.zmq_bridge_port = zmq_bridge_port
        self.use_dds_state = use_dds_state
        self.policy_freq = policy_freq
        self.control_freq = control_freq
        
        # 计算变频参数
        self.control_steps_per_policy = int(control_freq / policy_freq)  # 2
        self.policy_period = 1.0 / policy_freq  # 0.02s
        self.control_period = 1.0 / control_freq  # 0.01s
        
        print(f"🔄 初始化策略 ZeroMQ 中继器 - 变频版本")
        print(f"   策略接收端口: {zmq_policy_recv_port} (策略 PUSH → 中继 PULL)")
        print(f"   策略发送端口: {zmq_policy_send_port} (中继 PUSH → 策略 PULL)")
        print(f"   桥接器端口: {zmq_bridge_port} (中继 PUSH → 桥接器 PULL)")
        print(f"   策略频率: {policy_freq}Hz, 控制频率: {control_freq}Hz")
        print(f"   每个策略周期发送 {self.control_steps_per_policy} 个控制目标")
        print(f"   使用 DDS 状态: {use_dds_state}")
        
        # 设置 ZeroMQ
        self.context = zmq.Context()
        
        # 从策略接收控制的 socket (PULL)
        self.socket_policy_recv = self.context.socket(zmq.PULL)
        self.socket_policy_recv.setsockopt(zmq.RCVHWM, 10)
        self.socket_policy_recv.setsockopt(zmq.RCVBUF, 1048576)
        self.socket_policy_recv.setsockopt(zmq.LINGER, 0)
        # 连接到策略进程，而不是绑定
        self.socket_policy_recv.connect(f"tcp://localhost:{zmq_policy_recv_port}")
        
        # 向策略发送状态的 socket (PUSH)
        self.socket_policy_send = self.context.socket(zmq.PUSH)
        self.socket_policy_send.setsockopt(zmq.SNDHWM, 10)
        self.socket_policy_send.setsockopt(zmq.SNDBUF, 1048576)
        self.socket_policy_send.setsockopt(zmq.LINGER, 0)
        # 连接到策略进程，而不是绑定
        self.socket_policy_send.connect(f"tcp://localhost:{zmq_policy_send_port}")
        
        # 向桥接器发送控制目标的 socket (PUSH)
        self.socket_bridge = self.context.socket(zmq.PUSH)
        self.socket_bridge.setsockopt(zmq.SNDHWM, 1)  # 减少队列大小
        self.socket_bridge.setsockopt(zmq.SNDBUF, 65536)  # 减少缓冲区
        self.socket_bridge.setsockopt(zmq.LINGER, 0)
        self.socket_bridge.bind(f"tcp://*:{zmq_bridge_port}")
        
        # Poller 设置
        self.poller = zmq.Poller()
        self.poller.register(self.socket_policy_recv, zmq.POLLIN)
        
        # DDS 状态订阅（如果启用）
        self.low_state = None
        self.high_state = None
        
        if use_dds_state:
            print("📡 设置 DDS 状态订阅...")
            # 低级状态
            self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
            self.low_state_sub.Init(self._low_state_handler, 10)
            
            # 高级状态
            self.high_state_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
            self.high_state_sub.Init(self._high_state_handler, 10)
        
        # 状态管理
        self.running = False
        self.cycle_id = 0
        
        # 控制插值状态
        self.current_controls = None  # 当前策略控制序列 (n_steps, n_joints)
        self.control_buffer = []      # 100Hz 控制目标缓冲区
        self.control_send_index = 0   # 当前发送的控制索引
        
        # 设置默认控制
        self.set_default_controls()
        
        # 统计
        self.recv_count = 0
        self.send_count = 0
        self.state_send_count = 0
        self.control_send_count = 0
        
        print("✅ 策略 ZeroMQ 中继器初始化完成")
    
    def _low_state_handler(self, msg: LowState_):
        """处理低级状态"""
        self.low_state = msg
    
    def _high_state_handler(self, msg: SportModeState_):
        """处理高级状态"""
        self.high_state = msg
    
    def set_default_controls(self):
        """设置默认控制序列（站立姿态）"""
        # G1 站立姿态
        default_target = np.zeros(41)
        
        # 身体关节默认角度（站立姿态）
        # 腰部 (3)
        default_target[0:3] = 0.0
        
        # 左臂 (6)
        default_target[3:9] = 0.0
        
        # 右臂 (6)
        default_target[9:15] = 0.0
        
        # 左腿 (6) - 轻微弯曲
        default_target[15] = 0.0      # hip roll
        default_target[16] = 0.0      # hip yaw
        default_target[17] = -0.3     # hip pitch
        default_target[18] = 0.6      # knee
        default_target[19] = -0.3     # ankle pitch
        default_target[20] = 0.0      # ankle roll
        
        # 右腿 (6) - 轻微弯曲
        default_target[21] = 0.0      # hip roll
        default_target[22] = 0.0      # hip yaw
        default_target[23] = -0.3     # hip pitch
        default_target[24] = 0.6      # knee
        default_target[25] = -0.3     # ankle pitch
        default_target[26] = 0.0      # ankle roll
        
        # 手部关节 (14) - 全部为0
        default_target[27:41] = 0.0
        
        # 创建默认控制序列（重复同样的目标）
        self.current_controls = np.tile(default_target, (self.control_steps_per_policy, 1)).astype(np.float32)
        self.prepare_control_buffer()
    
    def recv_policy_controls(self, timeout_ms: int = 1000) -> Optional[Dict[str, Any]]:
        """从策略接收控制命令（带超时的同步接收）"""
        try:
            # 同步等待控制命令
            socks = dict(self.poller.poll(timeout_ms))
            
            if self.socket_policy_recv in socks:
                # 接收多部分消息 [cycle_id, controls_bytes]
                parts = self.socket_policy_recv.recv_multipart(zmq.NOBLOCK)
                
                if len(parts) != 2:
                    print(f"⚠️ 接收到无效的控制消息格式，parts={len(parts)}")
                    return None
                
                # 解析 cycle_id
                recv_cycle_id = struct.unpack('I', parts[0])[0]
                
                # 解析控制命令
                response = pickle.loads(parts[1])
                
                self.recv_count += 1
                return {
                    'cycle_id': recv_cycle_id,
                    'controls': response['controls'],
                    'timing': response.get('timing', {})
                }
            
            return None
            
        except zmq.Again:
            return None
        except Exception as e:
            print(f"❌ 接收策略控制错误: {e}")
            return None
    
    def send_state_to_policy(self, cycle_id: int) -> bool:
        """发送状态到策略（同步发送）"""
        try:
            # 构建状态数据
            state_data = self._build_state_data()
            
            if state_data is None:
                return False
            
            # 准备消息
            state_bytes = pickle.dumps(state_data, protocol=pickle.HIGHEST_PROTOCOL)
            cycle_id_bytes = struct.pack('I', cycle_id)
            
            # 发送多部分消息 [cycle_id, state_bytes]
            self.socket_policy_send.send_multipart([cycle_id_bytes, state_bytes], zmq.NOBLOCK)
            
            self.state_send_count += 1
            return True
            
        except zmq.Again:
            return False
        except Exception as e:
            print(f"❌ 发送状态到策略错误: {e}")
            return False
    
    def prepare_control_buffer(self):
        """准备 100Hz 控制缓冲区"""
        if self.current_controls is None:
            return
            
        # 如果策略提供的控制序列长度正好等于 control_steps_per_policy
        if len(self.current_controls) >= self.control_steps_per_policy:
            # 直接使用前 control_steps_per_policy 个控制目标
            self.control_buffer = [self.current_controls[i] for i in range(self.control_steps_per_policy)]
        elif len(self.current_controls) == 1:
            # 如果只有一个控制目标，重复使用
            self.control_buffer = [self.current_controls[0] for _ in range(self.control_steps_per_policy)]
        else:
            # 线性插值（如果长度不匹配）
            self.control_buffer = []
            for i in range(self.control_steps_per_policy):
                alpha = i / (self.control_steps_per_policy - 1) if self.control_steps_per_policy > 1 else 0.0
                source_idx = min(int(alpha * len(self.current_controls)), len(self.current_controls) - 1)
                self.control_buffer.append(self.current_controls[source_idx])
        
        self.control_send_index = 0
    
    def send_control_to_bridge(self) -> bool:
        """发送单个控制目标到桥接器（100Hz 调用）"""
        try:
            if self.control_send_index >= len(self.control_buffer):
                # 如果缓冲区用完，使用最后一个控制目标
                control_target = self.control_buffer[-1] if self.control_buffer else np.zeros(41)
            else:
                control_target = self.control_buffer[self.control_send_index]
            
            # 准备消息
            message = {
                'control_target': control_target,
                'cycle_id': self.cycle_id,
                'control_index': self.control_send_index,
                'timestamp': time.time()
            }
            
            # 发送到桥接器
            message_bytes = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
            self.socket_bridge.send(message_bytes, zmq.NOBLOCK)
            
            self.control_send_count += 1
            self.control_send_index += 1
            return True
            
        except zmq.Again:
            print(f"⚠️ 控制目标发送队列已满")
            return False
        except Exception as e:
            print(f"❌ 发送控制目标错误: {e}")
            return False
    
    def _build_state_data(self) -> Optional[Dict[str, Any]]:
        """构建状态数据（从 DDS 或默认）"""
        if self.use_dds_state and self.low_state is not None:
            # 从 DDS 低级状态构建
            # G1 有 41 个关节（27 身体 + 14 手部）
            # qpos: 7 (浮动基座: x,y,z,qw,qx,qy,qz) + 41 (关节)
            # qvel: 6 (浮动基座: vx,vy,vz,wx,wy,wz) + 41 (关节速度)
            qpos = np.zeros(48)  
            qvel = np.zeros(47)  
            
            # 基座状态（如果有高级状态）
            if self.high_state is not None:
                qpos[0:3] = self.high_state.position
                # IMU 四元数 (如果有的话)
                if hasattr(self.high_state, 'imu_state') and hasattr(self.high_state.imu_state, 'quaternion'):
                    qpos[3:7] = self.high_state.imu_state.quaternion  # qw, qx, qy, qz
                else:
                    qpos[3] = 1.0  # 默认四元数 w
                    qpos[4:7] = 0.0  # qx, qy, qz
                
                # 基座速度
                if hasattr(self.high_state, 'velocity'):
                    qvel[0:3] = self.high_state.velocity
                if hasattr(self.high_state, 'angular_velocity'):
                    qvel[3:6] = self.high_state.angular_velocity
            else:
                # 默认基座状态
                qpos[2] = 0.75  # 默认高度
                qpos[3] = 1.0   # 四元数 w
                qpos[4:7] = 0.0  # qx, qy, qz
            
            # 身体关节状态（G1 有 27 个身体关节）
            num_body_joints = min(27, len(self.low_state.motor_state))
            for i in range(num_body_joints):
                qpos[7 + i] = self.low_state.motor_state[i].q
                qvel[6 + i] = self.low_state.motor_state[i].dq
            
            # 手部关节状态（14 个手部关节）
            # 如果有手部状态的话
            # 左手 7 个关节
            if hasattr(self, 'hand_state') and self.hand_state is not None:
                if hasattr(self.hand_state, 'left_hand_position'):
                    for i in range(7):
                        qpos[7 + 27 + i] = self.hand_state.left_hand_position[i]
                        qvel[6 + 27 + i] = self.hand_state.left_hand_velocity[i]
                
                # 右手 7 个关节
                if hasattr(self.hand_state, 'right_hand_position'):
                    for i in range(7):
                        qpos[7 + 27 + 7 + i] = self.hand_state.right_hand_position[i]
                        qvel[6 + 27 + 7 + i] = self.hand_state.right_hand_velocity[i]
            
            # 添加 mocap_pos 和 mocap_quat（策略需要）
            mocap_pos = np.zeros(3)
            mocap_quat = np.array([1.0, 0.0, 0.0, 0.0])
            
            return {
                'qpos': qpos.astype(np.float32),
                'qvel': qvel.astype(np.float32),
                'mocap_pos': mocap_pos.astype(np.float32),
                'mocap_quat': mocap_quat.astype(np.float32),
                'time': time.time()
            }
        else:
            # 使用默认状态
            return self._get_default_state()
    
    def _get_default_state(self) -> Dict[str, Any]:
        """获取默认机器人状态"""
        # G1 默认站立状态
        qpos = np.zeros(48)
        qvel = np.zeros(47)
        
        # 基座位置
        qpos[0] = 0.0   # x
        qpos[1] = 0.0   # y  
        qpos[2] = 0.75  # z
        qpos[3] = 1.0   # qw
        qpos[4] = 0.0   # qx
        qpos[5] = 0.0   # qy
        qpos[6] = 0.0   # qz
        
        # 身体关节默认角度（站立姿态）
        # 腰部 (3)
        qpos[7:10] = 0.0
        
        # 左臂 (6)
        qpos[10:16] = 0.0
        
        # 右臂 (6)
        qpos[16:22] = 0.0
        
        # 左腿 (6) - 轻微弯曲
        qpos[22] = 0.0      # hip roll
        qpos[23] = 0.0      # hip yaw
        qpos[24] = -0.3     # hip pitch
        qpos[25] = 0.6      # knee
        qpos[26] = -0.3     # ankle pitch
        qpos[27] = 0.0      # ankle roll
        
        # 右腿 (6) - 轻微弯曲
        qpos[28] = 0.0      # hip roll
        qpos[29] = 0.0      # hip yaw
        qpos[30] = -0.3     # hip pitch
        qpos[31] = 0.6      # knee
        qpos[32] = -0.3     # ankle pitch
        qpos[33] = 0.0      # ankle roll
        
        # 手部关节 (14) - 全部为0
        qpos[34:48] = 0.0
        
        # mocap
        mocap_pos = np.zeros(3)
        mocap_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        return {
            'qpos': qpos.astype(np.float32),
            'qvel': qvel.astype(np.float32),
            'mocap_pos': mocap_pos.astype(np.float32),
            'mocap_quat': mocap_quat.astype(np.float32),
            'time': time.time()
        }
    
    def run_policy_barrier_loop(self):
        """运行 50Hz 策略 barrier 循环"""
        print("🚀 启动 50Hz 策略 barrier 循环...")
        print("📡 等待来自 run_policy_pruned.py 的控制命令...")
        
        self.running = True
        
        # 主循环 - 50Hz barrier 同步（与 isolated_simulation.py 保持一致）
        while self.running:
            try:
                barrier_start_time = time.time()
                
                # ========== Barrier A: 接收策略控制（同步等待）==========
                control_data = self.recv_policy_controls(timeout_ms=int(self.policy_period * 1000 * 1.5))  # 1.5倍超时
                
                if control_data is not None:
                    # 收到控制命令
                    controls = control_data['controls']
                    cycle_id = control_data['cycle_id']
                    
                    if self.recv_count == 1:
                        print(f"✅ 首次收到策略控制，cycle_id={cycle_id}")
                    
                    # 更新控制序列
                    self.current_controls = controls
                    self.prepare_control_buffer()
                    
                    if self.recv_count % 50 == 0:  # 每秒打印一次
                        print(f"📊 Barrier 状态 - 接收控制: {self.recv_count}, 发送状态: {self.state_send_count}")
                else:
                    print(f"⚠️ Cycle #{self.cycle_id}: 策略控制接收超时")
                
                # ========== Barrier B: 发送状态到策略 ==========
                if not self.send_state_to_policy(self.cycle_id):
                    print(f"❌ Cycle #{self.cycle_id}: 状态发送失败")
                    continue
                
                if self.cycle_id == 0:
                    print(f"✅ 首次发送状态到策略，cycle_id={self.cycle_id}")
                
                # ========== 推进到下一个周期 ==========
                self.cycle_id += 1
                
                # ========== 频率控制 ==========
                barrier_elapsed = time.time() - barrier_start_time
                sleep_time = self.policy_period - barrier_elapsed
                
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
                elif sleep_time < -0.005:
                    if self.cycle_id % 50 == 0:
                        print(f"⚠️ Barrier #{self.cycle_id}: 延迟 {-sleep_time*1000:.1f}ms")
                
            except KeyboardInterrupt:
                print("\n🛑 收到中断信号，停止 barrier 循环...")
                break
            except Exception as e:
                print(f"❌ Barrier 循环错误: {e}")
                time.sleep(0.1)
    
    def run_control_sender_loop(self):
        """运行 100Hz 控制发送循环"""
        print("🎮 启动 100Hz 控制发送循环...")
        
        # 主循环 - 100Hz 控制发送
        while self.running:
            try:
                control_start_time = time.time()
                
                # 只有在有有效控制缓冲区时才发送
                if len(self.control_buffer) > 0:
                    self.send_control_to_bridge()
                    
                    if self.control_send_count % 100 == 0:  # 每秒打印一次
                        print(f"🎮 控制发送状态 - 已发送: {self.control_send_count}")
                
                # ========== 频率控制 ==========
                control_elapsed = time.time() - control_start_time
                sleep_time = self.control_period - control_elapsed
                
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
                elif sleep_time < -0.005:
                    if self.control_send_count % 100 == 0:
                        print(f"⚠️ 控制发送延迟: {-sleep_time*1000:.1f}ms")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 控制发送循环错误: {e}")
                time.sleep(0.1)
    
    def run(self):
        """主运行方法"""
        import threading
        
        print("🚀 启动策略 ZeroMQ 中继服务...")
        print("💡 运行模式:")
        print("   - 50Hz lock-step barrier 与策略通信")
        print("   - 100Hz 控制目标发送到 SDK 桥接器")
        print("   - 变频功能：50Hz → 100Hz")
        
        self.running = True
        
        try:
            # 启动 100Hz 控制发送线程
            control_thread = threading.Thread(target=self.run_control_sender_loop, daemon=True)
            control_thread.start()
            
            # 主线程运行 50Hz barrier 循环
            self.run_policy_barrier_loop()
            
        except KeyboardInterrupt:
            print("\n🛑 收到中断信号，停止所有循环...")
        
        self.stop()
    
    def stop(self):
        """停止服务"""
        self.running = False
        
        # 关闭 ZeroMQ
        self.socket_policy_recv.close()
        self.socket_policy_send.close()
        self.socket_bridge.close()
        self.context.term()
        
        print(f"\n📊 === 最终统计 ===")
        print(f"   接收控制命令: {self.recv_count}")
        print(f"   发送状态反馈: {self.state_send_count}")
        print(f"   发送控制目标: {self.control_send_count}")
        print("✅ 策略 ZeroMQ 中继已停止")


def main():
    parser = argparse.ArgumentParser(
        description="策略 ZeroMQ 中继器 - 变频版本，承担 50Hz→100Hz 变频功能"
    )
    parser.add_argument(
        "--zmq_policy_recv_port",
        type=int,
        default=5556,
        help="策略控制接收端口"
    )
    parser.add_argument(
        "--zmq_policy_send_port",
        type=int,
        default=5555,
        help="策略状态发送端口"
    )
    parser.add_argument(
        "--zmq_bridge_port",
        type=int,
        default=5557,
        help="桥接器命令发送端口"
    )
    parser.add_argument(
        "--no_dds_state",
        action="store_true",
        help="不使用 DDS 状态反馈"
    )
    parser.add_argument(
        "--policy_freq",
        type=float,
        default=50.0,
        help="策略频率"
    )
    parser.add_argument(
        "--control_freq",
        type=float,
        default=100.0,
        help="控制频率"
    )
    
    args = parser.parse_args()
    
    # 初始化 DDS（如果需要）
    if not args.no_dds_state:
        print("📡 初始化 DDS 通信...")
        ChannelFactoryInitialize(1, "lo")
    
    # 创建并运行中继器
    relay = PolicyZMQRelay(
        zmq_policy_recv_port=args.zmq_policy_recv_port,
        zmq_policy_send_port=args.zmq_policy_send_port,
        zmq_bridge_port=args.zmq_bridge_port,
        use_dds_state=not args.no_dds_state,
        policy_freq=args.policy_freq,
        control_freq=args.control_freq
    )
    
    relay.run()


if __name__ == "__main__":
    main() 