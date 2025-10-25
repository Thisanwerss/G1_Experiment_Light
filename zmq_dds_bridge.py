#!/usr/bin/env python3
"""
ZeroMQ-DDS 通信桥接器
=====================
该脚本作为连接 `run_policy_pruned.py` 控制节点和真实G1机器人的桥梁。

它通过ZeroMQ从控制器接收PD（Proportional-Derivative）控制目标，
并通过DDS（Data Distribution Service）将这些指令发送给机器人硬件。
同时，它会收集机器人的状态数据（通过DDS和Vicon）并将其发送回控制节点。

使用方式:
1. CEM+lo模式 (用于测试DDS通信): 
   python zmq_dds_bridge.py --channel lo 
2. CEM+真实机器人 (需要Vicon): 
   python zmq_dds_bridge.py --channel <network_interface>

启动Vicon的命令:
ros2 launch vicon_receiver client.launch.py 
"""

import sys, os
print("[dbg] sys.executable =", sys.executable)
print("[dbg] first 3 sys.path =", sys.path[:3])
import importlib
m = importlib.import_module("sdk_controller")
print("[dbg] sdk_controller file =", getattr(m, "__file__", None))





import argparse
import time
import pickle
import struct
from typing import Dict, Any, Optional, Tuple
from threading import Thread, Event, Lock
import signal
import sys
import numpy as np
import zmq
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
from sdk_controller.vicon_hg_publisher import ViconPosePublisherHG
from typing import Dict, Any


# Vicon/ROS2相关导入 - 如果失败则定义一个假的ViconSubscriber
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from vicon_receiver.msg import Position

    class ViconSubscriber(Node):
        """通过ROS2订阅Vicon数据，并计算速度"""
        def __init__(self):
            super().__init__('vicon_subscriber_node')
            self.lock = Lock()
            
            # 用于速度计算的数据历史
            self.t, self.prev_t, self.prev_prev_t = 0., 0., 0.
            self.p, self.prev_p, self.prev_prev_p = np.zeros(3), np.zeros(3), np.zeros(3)
            # q: (w, x, y, z)
            self.q, self.prev_q, self.prev_prev_q = np.array([1.,0.,0.,0.]), np.array([1.,0.,0.,0.]), np.array([1.,0.,0.,0.])

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
                # 时间戳更新
                self.prev_prev_t = self.prev_t
                self.prev_t = self.t
                self.t = current_time

                # 位置更新 (mm -> m)
                self.prev_prev_p = self.prev_p
                self.prev_p = self.p
                self.p = np.array([msg.x_trans, msg.y_trans, msg.z_trans]) / 1000.0
                self.p[2] += VICON_Z_OFFSET # 应用Z轴偏移

                # 四元数更新 (w, x, y, z)
                self.prev_prev_q = self.prev_q
                self.prev_q = self.q
                self.q = np.array([msg.w, msg.x_rot, msg.y_rot, msg.z_rot])
                
                self.is_active = True
                self.last_update_time = current_time

        def get_state(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            """获取最新的位姿和速度"""
            with self.lock:
                # 如果超过0.5秒没有收到数据，则认为Vicon失效
                if time.time() - self.last_update_time > 0.5:
                    self.is_active = False
                    print("⚠️ Vicon 数据超时", flush=True)
                    return None, None, None, None

                # 如果历史数据不足，则不计算速度
                if self.prev_prev_t == 0:
                    return self.p.copy(), self.q.copy(), np.zeros(3), np.zeros(3)

                # --- 计算线速度 (全局坐标系) ---
                # 参考 vicon_publisher.py 使用二阶后向差分
                avg_dt = (self.t - self.prev_prev_t) / 2.0
                if avg_dt > 1e-6:
                    self.v = (3 * self.p - 4 * self.prev_p + self.prev_prev_p) / (2 * avg_dt)

                # --- 计算角速度 (机身坐标系) ---
                # 参考 vicon_publisher.py 实现
                dt_w = self.t - self.prev_prev_t
                if dt_w > 1e-6:
                    # q_prev_prev_conj * q_curr ~= [cos(theta/2), sin(theta/2)*axis]
                    # 以下公式是其线性近似，用于小角度变化
                    self.w = (2.0 / dt_w) * np.array([
                        self.prev_prev_q[0]*self.q[1] - self.prev_prev_q[1]*self.q[0] - self.prev_prev_q[2]*self.q[3] + self.prev_prev_q[3]*self.q[2],
                        self.prev_prev_q[0]*self.q[2] + self.prev_prev_q[1]*self.q[3] - self.prev_prev_q[2]*self.q[0] - self.prev_prev_q[3]*self.q[1],
                        self.prev_prev_q[0]*self.q[3] - self.prev_prev_q[1]*self.q[2] + self.prev_prev_q[2]*self.q[1] - self.prev_prev_q[3]*self.q[0]
                    ])
                    # 噪声过滤
                    self.w[np.abs(self.w) < 0.04] = 0.0
                
                return self.p.copy(), self.q.copy(), self.v.copy(), self.w.copy()
        
        def start(self):
            """在后台线程中启动ROS2节点"""
            self.thread = Thread(target=self.run_node, daemon=True)
            self.thread.start()

        def run_node(self):
            """运行rclpy.spin()"""
            print("Vicon subscriber thread started.")
            try:
                rclpy.init()
                rclpy.spin(self)
            except Exception as e:
                print(f"RCLPY spin failed: {e}")
            finally:
                self.destroy_node()
                rclpy.shutdown()
                print("Vicon subscriber thread stopped.")

except (ImportError, ModuleNotFoundError):
    print("⚠️ ROS2 或 vicon_receiver 未找到，Vicon功能将被禁用。")

    class ViconSubscriber:
        """当ROS2不可用时的虚拟ViconSubscriber"""
        def __init__(self):
            pass
        
        def start(self):
            print("   (虚拟ViconSubscriber已启动，无实际操作)")

        def get_state(self) -> Tuple[None, None, None, None]:
            # 总是返回None，模拟Vicon未激活状态
            return None, None, None, None


class ZMQDDSBridge:
    """ZeroMQ 到 DDS 的通信桥接器"""
    
    def __init__(
        self,
        channel: str = "lo",
        domain_id: int = 1,
        zmq_state_port: int = 5555,
        zmq_ctrl_port: int = 5556,
        control_frequency: float = 50.0,
        kp_scale_factor: float = 1.0,
        safety_profile: str = "default"
    ):
        self.channel = channel
        self.domain_id = domain_id
        self.zmq_state_port = zmq_state_port
        self.zmq_ctrl_port = zmq_ctrl_port
        self.control_frequency = control_frequency
        self.kp_scale_factor = kp_scale_factor
        self.safety_profile = safety_profile
        self.control_dt = 1.0 / self.control_frequency
        
        # 状态管理
        self.running = Event()
        self.cycle_id = 0
        self.current_controls = None
        
        print(f"🚀 初始化 ZeroMQ-DDS 桥接器")
        print(f"   模式: 真实机器人/lo模式 (通道: {channel})")
        print(f"   控制频率: {control_frequency} Hz")
        
        # 1. 初始化 ZeroMQ 连接
        self._setup_zmq()
        
        # 2. 初始化CEM控制器后端
        self._setup_cem_controller()
            
        print("✅ 桥接器初始化完成")
    
    def _setup_zmq(self):
        """设置 ZeroMQ 连接"""
        print("🌐 设置 ZeroMQ 连接...")
        
        self.context = zmq.Context()
        
        # 状态发送端 (连接到控制节点的 PULL 端口)
        self.socket_state = self.context.socket(zmq.PUSH)
        self.socket_state.connect(f"tcp://localhost:{self.zmq_state_port}")
        
        # 控制接收端 (连接到控制节点的 PUSH 端口) 
        self.socket_ctrl = self.context.socket(zmq.PULL)
        self.socket_ctrl.connect(f"tcp://localhost:{self.zmq_ctrl_port}")
        
        # 设置非阻塞轮询
        self.poller = zmq.Poller()
        self.poller.register(self.socket_ctrl, zmq.POLLIN)
        
    def _setup_cem_controller(self):
        """设置CEM控制器模式"""
        print(f"🤖 设置 CEM 控制模式 (通道: {self.channel})...")
        
        # 初始化DDS - 根据通道决定domain_id
        if self.channel == "lo":
            print("   使用lo接口 (domain_id=1)")
            ChannelFactoryInitialize(1, "lo")
        else:
            print(f"   使用真实网络接口: {self.channel} (domain_id=0)")
            ChannelFactoryInitialize(0, self.channel)
        
        # 创建CEM控制器
        self.cem_controller = CEMSDKController(
            simulate=False,
            robot_config=None,  # 使用G1默认配置
            xml_path="g1_model/g1_lab.xml",
            vicon_required=(self.channel != "lo"),  # lo模式不需要vicon
            lo_mode=(self.channel == "lo"),  # 传递lo模式标志
            kp_scale_factor=self.kp_scale_factor,
            safety_profile=self.safety_profile
        )
        
        print("✅ CEM控制器设置完成")
    
    def get_robot_state(self) -> Dict[str, Any]:
        """获取机器人状态"""
        # 真实机器人或lo模式：使用CEM控制器获取状态
        return self.cem_controller.get_robot_state()
    
    def execute_robot_control(self, controls: np.ndarray):
        """执行G1机器人控制 - 将控制序列插值到1000Hz发送"""
        if len(controls) > 0 and self.cem_controller is not None:
            # 策略(e.g., CEM)以50Hz提供PD目标(每0.02s提供1个点)，机器人控制器期望1000Hz
            # 因此，每个PD目标需要保持20ms (发送20次，每次间隔1ms)
            num_sends = int((1.0 / self.control_frequency) / 0.001)
            
            for pd_targets in controls:
                # 在一个控制周期内，以1000Hz重复发送同一个PD目标
                for _ in range(num_sends):
                    self.cem_controller.send_motor_command(
                        time=time.time(), 
                        pd_targets=pd_targets
                    )
                    # 1000Hz 控制频率
                    time.sleep(0.001)
    
    def send_state_to_control(self, state: Dict[str, Any]) -> bool:
        """向控制节点发送状态"""
        try:
            # 转换numpy数组为list避免序列化兼容性问题
            safe_state = {}
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    safe_state[key] = value.tolist()
                else:
                    safe_state[key] = value
            
            # 序列化状态
            state_bytes = pickle.dumps(safe_state, protocol=pickle.HIGHEST_PROTOCOL)
            cycle_id_bytes = struct.pack('I', self.cycle_id)
            
            # 发送多部分消息
            self.socket_state.send_multipart([cycle_id_bytes, state_bytes], zmq.NOBLOCK)
            return True
            
        except zmq.Again:
            print(f"⚠️ 状态发送队列满，cycle_id={self.cycle_id}")
            return False
        except Exception as e:
            print(f"❌ 状态发送错误: {e}")
            return False
    
    def recv_controls_from_control_blocking(self) -> Optional[np.ndarray]:
        """从控制节点接收控制命令 - 阻塞等待"""
        try:
            # 阻塞等待控制命令
            socks = dict(self.poller.poll(100))  # 100ms 超时
            
            if self.socket_ctrl in socks:
                # 接收多部分消息
                parts = self.socket_ctrl.recv_multipart(zmq.NOBLOCK)
                
                if len(parts) != 2:
                    print(f"⚠️ 接收到无效控制消息格式")
                    return None
                
                # 解析 cycle_id
                recv_cycle_id = struct.unpack('I', parts[0])[0]
                
                # 反序列化控制命令
                response = pickle.loads(parts[1])
                controls = response['controls']
                
                # 转换回numpy数组
                if isinstance(controls, list):
                    controls = np.array(controls, dtype=np.float32)
                
                return controls
            
            return None
            
        except zmq.Again:
            return None
        except Exception as e:
            print(f"❌ 控制接收错误: {e}")
            return None
    
    def run(self):
        """运行主循环"""
        print(f"🎬 启动桥接器主循环")
        
        self.running.set()

        # --- STAGE 1: HANDSHAKE ---
        print("\n--- STAGE 1: Handshake ---")
        
        initial_state = None
        print("🔄 等待有效的初始机器人状态 (Vicon+DDS)...")
        while initial_state is None and self.running.is_set():
            initial_state = self.get_robot_state()
            if initial_state is None:
                if not self.running.is_set(): break
                print("  ...仍在等待初始状态, 0.5s后重试...")
                time.sleep(0.5)
                
        if not self.running.is_set() or initial_state is None:
            print("❌ 未能获取初始状态，桥接器正在停止。")
            self.stop()
            return

        try:
            initial_state_msg = {
                'type': 'init',
                'qpos': initial_state['qpos']
            }
            print("📤 正在发送初始状态给策略节点...")
            self.socket_state.send(pickle.dumps(initial_state_msg))
            
            print("🔄 正在等待策略节点返回握手确认...")
            response_bytes = self.socket_ctrl.recv() # 阻塞接收
            response = pickle.loads(response_bytes)
            
            if response.get('type') == 'aligned_trajectory':
                print("✅ 握手完成。")
            else:
                raise ValueError("从策略节点收到无效的握手响应")

        except Exception as e:
            print(f"❌ 握手失败: {e}")
            self.stop()
            return

        # --- STAGE 2: MAIN CONTROL LOOP ---
        print("\n--- STAGE 2: 主控制循环 ---")
        
        # 发送第一个真实状态以启动控制循环
        first_state = self.get_robot_state()
        if first_state is None:
            print("❌ 握手后立即失去机器人状态。为安全起见，正在停止。")
            self.stop()
            return
            
        self.send_state_to_control(first_state)
        print("📤 已发送初始状态，等待第一个控制命令...")
        
        try:
            while self.running.is_set():
                # ========== 锁步屏障：等待新的控制命令 ==========
                print(f"🔒 Cycle #{self.cycle_id}: 等待控制命令...")
                new_controls = None
                
                # 阻塞等待控制命令
                while new_controls is None and self.running.is_set():
                    new_controls = self.recv_controls_from_control_blocking()
                    if new_controls is None:
                        time.sleep(0.001)  # 短暂等待避免CPU占用过高
                
                if not self.running.is_set():
                    break
                    
                self.current_controls = new_controls
                print(f"✅ Cycle #{self.cycle_id}: 收到控制命令 shape={new_controls.shape}")
                
                # ========== 同步交换：发送当前状态 ==========
                state = self.get_robot_state()
                if state is None:
                    print("❌ 失去机器人状态(Vicon或DDS超时)。为安全起见，正在停止。")
                    print("   发送阻尼命令...")
                    for _ in range(5): # 发送几次以确保机器人收到
                        self.cem_controller.damping_motor_cmd()
                        time.sleep(0.01)
                    self.stop()
                    break # 退出循环
                
                if not self.send_state_to_control(state):
                    print(f"❌ Cycle #{self.cycle_id}: 状态发送失败")
                    continue
                
                # ========== 执行机器人控制 ==========
                self.execute_robot_control(self.current_controls)
                
                # ========== 周期完成 ==========
                self.cycle_id += 1
                        
        except KeyboardInterrupt:
            print("\n🛑 收到中断信号...")
        
        finally:
            self.stop()
    
    def stop(self):
        """停止桥接器"""
        print("🛑 停止桥接器...")
        self.running.clear()
        
        # 关闭 ZeroMQ 连接
        self.socket_state.close()
        self.socket_ctrl.close()
        self.context.term()
        
        print("✅ 桥接器已停止")


def signal_handler(sig, frame):
    """信号处理器"""
    print("\n🛑 收到停止信号")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="ZeroMQ-DDS 通信桥接器")
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
    parser.add_argument(
        "--zmq_state_port",
        type=int,
        default=5555,
        help="ZeroMQ 状态端口"
    )
    parser.add_argument(
        "--zmq_ctrl_port",
        type=int,
        default=5556,
        help="ZeroMQ 控制端口"
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
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
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 创建并运行桥接器
    bridge = ZMQDDSBridge(
        channel=args.channel,
        domain_id=args.domain_id,
        zmq_state_port=args.zmq_state_port,
        zmq_ctrl_port=args.zmq_ctrl_port,
        control_frequency=args.frequency,
        kp_scale_factor=args.kp_scale,
        safety_profile=args.safety_profile
    )
    
    bridge.run()


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
        print(f"🤖 初始化CEMSDKController")
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
        
        # Vicon状态缓存
        if vicon_required:
            print("   启动 Vicon Subscriber...")
            self.vicon_subscriber = ViconSubscriber()
            self.vicon_subscriber.start()
        else:
            self.vicon_subscriber = None
        
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

        # 从Vicon更新基座状态
        if self.vicon_required and self.vicon_subscriber:
            p, q, v, w = self.vicon_subscriber.get_state()
            
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
                # Vicon数据无效，可能导致上层策略出问题，返回None来中断当前周期
                print("❌ get_robot_state: 无效的Vicon数据，返回None", flush=True)
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


if __name__ == "__main__":
    main() 