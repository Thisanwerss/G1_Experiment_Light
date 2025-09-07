#!/usr/bin/env python3
"""
ZeroMQ-DDS Simulation Bridge
============================

This script runs a MuJoCo simulation that acts as a digital twin for the G1 robot.
It communicates with the `run_policy_pruned.py` control node via ZeroMQ,
receiving PD targets and applying them in a simulated environment.

Key Features:
- **Lock-step Simulation**: Mirrors the real robot's control loop for high-fidelity testing.
- **Dynamic Initialization**:
    - Attempts to initialize from the real robot's state (via DDS and Vicon).
    - If unavailable, starts from a standing pose with randomized base position/yaw.
- **No DDS Command Forwarding**: Receives control commands but does NOT send them to the
  real robot, making it a safe environment for policy testing.
- **Control Logging**: Logs received PD control targets for analysis.

Usage:
    python zmq_dds_sim.py
    # Optional: specify DDS channel for initialization attempts
    python zmq_dds_sim.py --init_channel enp7s0
"""

import sys, os
print("[dbg] sys.executable =", sys.executable)
print("[dbg] first 3 sys.path =", sys.path[:3])
import importlib
m = importlib.import_module("sdk_controller")
print("[dbg] sdk_controller file =", getattr(m, "__file__", None))




import sys
import os
# Ensure the project root is in the Python path to allow finding the 'sdk_controller' module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import pickle
import struct
from typing import Dict, Any, Optional, Tuple
from threading import Thread, Event, Lock
import signal
import numpy as np
import zmq
import json
from scipy.spatial.transform import Rotation as R


VICON_Z_OFFSET = 0.0 # for simulation no need to offset

# --- MuJoCo Simulation Imports ---
import mujoco
import mujoco.viewer

# --- Real Robot State Imports (for initialization only) ---
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from sdk_controller.robots.G1 import G1, STAND_UP_JOINT_POS, NUM_ACTIVE_BODY_JOINTS
    from sdk_controller.abstract_biped import HGSDKController, HGSafetyLayer
except (ImportError, ModuleNotFoundError) as e:
    print(f"❌ 捕获到导入错误: {e}")
    import traceback
    traceback.print_exc()
    print("❌ 关键的 'sdk_controller' 或其依赖项未找到。")
    print("   请确保 ATARI_NMPC 的根目录在您的 PYTHONPATH 中，并已安装所有依赖项。")
    sys.exit(1)

# Vicon/ROS2 related imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from vicon_receiver.msg import Position

    class ViconSubscriber(Node):
        """通过ROS2订阅Vicon数据，仅用于初始化"""
        def __init__(self):
            super().__init__('vicon_subscriber_init_node')
            self.lock = Lock()
            self.p = None
            self.q = None
            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.subscription = self.create_subscription(
                Position, '/vicon/G1/G1', self.listener_callback, qos_profile
            )
            self.get_logger().info("Vicon subscriber for initialization created.")

        def listener_callback(self, msg: Position):
            with self.lock:
                self.p = np.array([msg.x_trans, msg.y_trans, msg.z_trans]) / 1000.0
                self.p[2] += VICON_Z_OFFSET
                self.q = np.array([msg.w, msg.x_rot, msg.y_rot, msg.z_rot])
        
        def get_state(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            with self.lock:
                if self.p is None or self.q is None:
                    return None, None
                return self.p.copy(), self.q.copy()

except (ImportError, ModuleNotFoundError):
    print("⚠️ ROS2 或 vicon_receiver 未找到，无法从Vicon获取初始状态。")
    ViconSubscriber = None
    rclpy = None


class StateInitController(HGSDKController):
    """一个临时的控制器，仅用于从DDS获取一次关节状态"""
    def __init__(self, xml_path):
        # 使用一个假的机器人配置进行初始化
        robot_config = G1()
        robot_config.motor_wait_posture = None
        robot_config.motor_init_fsm_state = FSM_State.PASSIVE
        
        super().__init__(
            simulate=False,
            robot_config=robot_config,
            xml_path=xml_path,
            vicon_required=False,
            lo_mode=False
        )
        self.state_received = False

    def update_motor_cmd(self, time: float):
        # 不需要实现，因为我们不发送命令
        pass

    def get_joint_states(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取一次关节状态"""
        self.update_q_v_from_lowstate()
        self.update_hand_q_v_from_handstate()
        
        # 简单检查是否收到了数据 (例如，检查髋关节是否非零)
        if np.any(self._q[7:] != 0):
             self.state_received = True
             return self._q.copy(), self._v.copy()
        return None, None


class ZMQSimulationBridge:
    """ZeroMQ 驱动的 MuJoCo 仿真器"""
    
    def __init__(
        self,
        init_channel: Optional[str],
        zmq_state_port: int = 5555,
        zmq_ctrl_port: int = 5556,
        control_frequency: float = 50.0,
        seeref: bool = False,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_yaw: float = 0.0,
        log: bool = False
    ):
        self.init_channel = init_channel
        self.zmq_state_port = zmq_state_port
        self.zmq_ctrl_port = zmq_ctrl_port
        self.control_frequency = control_frequency
        self.control_dt = 1.0 / control_frequency
        self.seeref = seeref
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_yaw = offset_yaw # in degrees
        self.log_enabled = log
        if self.log_enabled:
            self.log_data = {}
        
        # 状态管理
        self.running = Event()
        self.cycle_id = 0
        self.current_controls = None
        self.emergency_stop_activated = False
        self.current_target_qpos = None # 用于存储幻影目标姿态
        
        print(f"🚀 初始化 ZeroMQ 仿真桥接器")
        if self.seeref:
            print(f"   模式: 查看参考轨迹 (See Reference Trajectory)")
        else:
            print(f"   模式: MuJoCo仿真")
        print(f"   控制频率: {control_frequency} Hz")
        print(f"   日志记录: {'启用' if self.log_enabled else '禁用'}")
        
        # 1. 初始化 ZeroMQ 连接
        self._setup_zmq()
        
        # 2. 初始化仿真环境
        self._setup_simulation()
        
        # 3. 初始化仿真状态 (关键步骤)
        self._initialize_sim_state()

        # 4. 在seeref模式下应用透明度
        if self.seeref:
            print("   Applying transparency for seeref mode.")
            # 保存原始颜色以便退出时恢复
            self.original_geom_rgba = self.mj_model.geom_rgba.copy()
            # 设置所有几何体的alpha通道为0.4 (半透明)
            self.mj_model.geom_rgba[:, 3] = 0.4
        
        # 5. 启动查看器
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        
        print("✅ 仿真桥接器初始化完成")
    
    def _setup_zmq(self):
        """设置 ZeroMQ 连接"""
        print("🌐 设置 ZeroMQ 连接...")
        self.context = zmq.Context()
        self.socket_state = self.context.socket(zmq.PUSH)
        self.socket_state.connect(f"tcp://localhost:{self.zmq_state_port}")
        self.socket_ctrl = self.context.socket(zmq.PULL)
        self.socket_ctrl.connect(f"tcp://localhost:{self.zmq_ctrl_port}")
        self.poller = zmq.Poller()
        self.poller.register(self.socket_ctrl, zmq.POLLIN)
        
    def _setup_simulation(self):
        """设置仿真后端"""
        print("🎮 设置 MuJoCo 仿真...")
        self.mj_model = mujoco.MjModel.from_xml_path("g1_model/scene.xml")
        
        # 配置 MuJoCo 参数
        self.mj_model.opt.timestep = 0.01
        self.mj_model.opt.iterations = 10
        self.mj_model.opt.ls_iterations = 50
        self.mj_model.opt.noslip_iterations = 2
        self.mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
        self.mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
        
        self.mj_data = mujoco.MjData(self.mj_model)

        # --- 移除幻影参考模型 ---
        self.ref_data = None
        
        # 计算仿真步数
        replan_period = 1.0 / self.control_frequency
        sim_steps_per_replan = int(replan_period / self.mj_model.opt.timestep)
        self.sim_steps_per_replan = max(sim_steps_per_replan, 1)
        self.actual_step_dt = self.sim_steps_per_replan * self.mj_model.opt.timestep
        
        # # 初始化安全层 (已禁用)
        # self.safety_layer = HGSafetyLayer(self.mj_model, conservative_safety=False)
        # # 估算一个用于安全检查的Kp增益数组
        # self.kp_gains_for_safety = self._estimate_kp_gains()
        
        print(f"   MuJoCo 时间步: {self.mj_model.opt.timestep:.4f}s")
        print(f"   MuJoCo 每控制周期步数: {self.sim_steps_per_replan}")
        print(f"   实际控制周期: {self.actual_step_dt:.4f}s")

    # def _estimate_kp_gains(self) -> np.ndarray:
    #     """估算一个近似的Kp增益数组用于安全检查，模仿HGSDKController的行为"""
    #     gains = np.zeros(NUM_ACTIVE_BODY_JOINTS)
    #     for mj_idx in range(NUM_ACTIVE_BODY_JOINTS):
    #         if mj_idx < 12:  # leg joints
    #             if mj_idx % 6 in [0, 1, 2]: gains[mj_idx] = 90.0   # hip
    #             elif mj_idx % 6 == 3: gains[mj_idx] = 150.0 # knee
    #             else: gains[mj_idx] = 60.0              # ankle
    #         elif mj_idx == 12: gains[mj_idx] = 90.0 # waist
    #         else:  # arm joints
    #             if mj_idx <= 19 or (mj_idx >= 22 and mj_idx <=25) : gains[mj_idx] = 60.0 # shoulder/elbow
    #             else: gains[mj_idx] = 30.0 # wrist
    #     return gains

    def _get_real_robot_initial_state(self, timeout=2.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        独立尝试从DDS和Vicon获取初始状态。
        返回一个元组 (base_qpos, joint_qpos)，其中任何一个都可能为None。
        """
        if self.init_channel is None:
            print("   --init_channel 未提供，跳过从真实机器人初始化。")
            return None, None
            
        print(f"   正在尝试连接到通道 '{self.init_channel}' 以获取初始状态...")
        
        base_qpos, joint_qpos = None, None
        
        # 初始化ROS2 (如果可用)
        vicon_sub = None
        executor = None
        if rclpy and ViconSubscriber:
            rclpy.init()
            vicon_sub = ViconSubscriber()
            executor = rclpy.executors.SingleThreadedExecutor()
            executor.add_node(vicon_sub)
            ros_thread = Thread(target=executor.spin, daemon=True)
            ros_thread.start()
        
        # 初始化DDS
        state_initializer = None
        try:
            domain_id = 0 if self.init_channel != "lo" else 1
            ChannelFactoryInitialize(domain_id, self.init_channel)
            state_initializer = StateInitController(xml_path="g1_model/g1_lab.xml")
        except Exception as e:
            print(f"   ❌ DDS 初始化失败: {e}")
            # 即使DDS失败，也继续尝试Vicon

        start_time = time.time()
        vicon_ok, dds_ok = False, False

        while time.time() - start_time < timeout and not (vicon_ok and dds_ok):
            # 尝试获取关节状态
            if state_initializer and not dds_ok:
                q, _ = state_initializer.get_joint_states()
                if q is not None:
                    joint_qpos = q[7:] # 只取关节部分
                    dds_ok = True
                    print("   ✅ 已从DDS获取关节状态。")

            # 尝试获取基座状态
            if vicon_sub and not vicon_ok:
                p, q_base = vicon_sub.get_state()
                if p is not None:
                    base_qpos = np.concatenate([p, q_base])
                    vicon_ok = True
                    print("   ✅ 已从Vicon获取基座状态。")
            
            # 如果Vicon不可用，则认为Vicon部分完成
            if not vicon_sub:
                vicon_ok = True

            time.sleep(0.1)

        if time.time() - start_time >= timeout:
            print("   ⚠️ 获取初始状态超时。")

        if executor:
            executor.shutdown()
        if rclpy and rclpy.ok():
            rclpy.shutdown()
            
        return base_qpos, joint_qpos

    def _initialize_sim_state(self):
        """初始化仿真器的状态，独立处理基座和关节。"""
        # 1. 定义一个已知稳定的基础站立姿态
        # 基座部分 (x, y, z, qw, qx, qy, qz)
        base_qpos = np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])
        # 关节部分
        joint_qpos = STAND_UP_JOINT_POS.copy()

        # 2. 应用用户指定的静态偏移量
        base_qpos[0] += self.offset_x
        base_qpos[1] += self.offset_y
        
        # 以与run_policy.py相同的方式应用yaw偏转
        yaw_offset_rad = np.deg2rad(self.offset_yaw)
        initial_quat_wxyz = base_qpos[3:7]
        initial_rotation = R.from_quat([initial_quat_wxyz[1], initial_quat_wxyz[2], initial_quat_wxyz[3], initial_quat_wxyz[0]])
        yaw_rotation = R.from_euler('z', yaw_offset_rad)
        new_rotation = yaw_rotation * initial_rotation
        new_quat_xyzw = new_rotation.as_quat()
        base_qpos[3:7] = np.array([new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]])

        print("--- Simulation Initial State ---")
        print(f"   Mode: Using fixed stand pose with offsets.")
        print(f"   Offset X: {self.offset_x:.2f} m")
        print(f"   Offset Y: {self.offset_y:.2f} m")
        print(f"   Offset Yaw: {self.offset_yaw:.2f} deg")
        print("--------------------------------")

        # 3. 组合并设置到mj_data
        full_qpos = np.concatenate([base_qpos, joint_qpos])
        num_joints_to_copy = min(len(full_qpos), len(self.mj_data.qpos))
        self.mj_data.qpos[:num_joints_to_copy] = full_qpos[:num_joints_to_copy]
        
        # 确保初始状态是物理合法的
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def get_robot_state(self) -> Dict[str, Any]:
        """获取仿真机器人状态"""
        return {
            'qpos': self.mj_data.qpos.copy(),
            'qvel': self.mj_data.qvel.copy(),
            'mocap_pos': self.mj_data.mocap_pos.copy(),
            'mocap_quat': self.mj_data.mocap_quat.copy(),
            'time': self.mj_data.time
        }
    
    def execute_simulation_steps(self, controls: np.ndarray):
        """执行仿真步骤 (已移除安全层检查)"""
        for i in range(self.sim_steps_per_replan):
            # 直接应用控制指令，不进行安全检查
            if i < len(controls):
                self.mj_data.ctrl[:] = controls[i]
            
            mujoco.mj_step(self.mj_model, self.mj_data)
            
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()
    
    def _update_ghost_visualization(self):
        """更新幻影模型的可视化状态"""
        pass
    
    def send_state_to_control(self, state: Dict[str, Any]) -> bool:
        """向控制节点发送状态"""
        try:
            safe_state = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in state.items()}
            state_bytes = pickle.dumps(safe_state, protocol=pickle.HIGHEST_PROTOCOL)
            cycle_id_bytes = struct.pack('I', self.cycle_id)
            self.socket_state.send_multipart([cycle_id_bytes, state_bytes], zmq.NOBLOCK)
            return True
        except Exception as e:
            print(f"❌ 状态发送错误: {e}")
            return False
    
    def recv_controls_from_control_blocking(self) -> Optional[np.ndarray]:
        """从控制节点接收控制命令 - 阻塞等待"""
        try:
            socks = dict(self.poller.poll(100))
            if self.socket_ctrl in socks:
                parts = self.socket_ctrl.recv_multipart(zmq.NOBLOCK)
                if len(parts) != 2:
                    print(f"⚠️ 接收到无效控制消息格式")
                    return None
                
                response = pickle.loads(parts[1])
                controls = np.array(response['controls'], dtype=np.float32)
                
                # --- LOGGING ---
                print(f"   LOG: 收到 {controls.shape[0]} 个PD目标, "
                      f"第一个目标的前3个关节: "
                      f"[{controls[0][0]:.3f}, {controls[0][1]:.3f}, {controls[0][2]:.3f}]")
                
                return controls
            return None
        except Exception as e:
            print(f"❌ 控制接收错误: {e}")
            return None
    
    def play_reference_trajectory(self, trajectory: list):
        """在MuJoCo Viewer中播放给定的轨迹"""
        print(f"🎬 开始播放接收到的参考轨迹... (共 {len(trajectory)} 帧)")
        
        if not self.viewer or not self.viewer.is_running():
            print("❌ Viewer未运行，无法播放轨迹。")
            return
            
        # 参考轨迹的帧率为30Hz
        fps = 30.0
        dt = 1.0 / fps
        
        for i, qpos_frame in enumerate(trajectory):
            if not self.viewer.is_running():
                print(" Viewer已关闭，播放中断。")
                break
                
            start_time = time.time()
            
            self.mj_data.qpos[:] = qpos_frame
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self.viewer.sync()
            
            # 保持帧率
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
        
        print("✅ 轨迹播放完成。")
        self.stop()

    def run(self):
        """运行主循环"""
        self.running.set()
        print(f"🎬 启动仿真桥接器主循环")

        # --- STAGE 1: HANDSHAKE ---
        print("\n--- STAGE 1: Handshake ---")
        try:
            # 1. 发送初始状态给策略节点用于对齐
            initial_qpos = self.mj_data.qpos.copy()
            initial_state_msg = {
                'type': 'init',
                'qpos': initial_qpos.tolist()
            }
            print("📤 正在发送初始状态给策略节点...")
            self.socket_state.send(pickle.dumps(initial_state_msg))
            
            # 2. 等待策略节点回传对齐后的轨迹数据
            print("🔄 正在等待策略节点返回对齐后的轨迹数据...")
            response_bytes = self.socket_ctrl.recv() # Blocking receive
            response = pickle.loads(response_bytes)
            
            if response.get('type') == 'aligned_trajectory':
                self.current_target_qpos = np.array(response['ghost_qpos'])
                received_trajectory = response['trajectory']
                print("✅ 已收到对齐后的轨迹数据，握手完成。")
                print(f"   幻影基座位置 (x,y,z): {self.current_target_qpos[0]:.3f}, {self.current_target_qpos[1]:.3f}, {self.current_target_qpos[2]:.3f}")
                
                if self.log_enabled:
                    self.log_data = {
                        'metadata': {
                            'timestamp': time.strftime("%Y%m%d_%H%M%S"),
                            'control_frequency': self.control_frequency,
                            'sim_timestep': self.mj_model.opt.timestep,
                            'mode': 'simulation',
                            'init_channel': self.init_channel,
                            'initial_offsets': {'x': self.offset_x, 'y': self.offset_y, 'yaw_deg': self.offset_yaw},
                            'seeref': self.seeref
                        },
                        'initial_state': {
                            'qpos': self.mj_data.qpos.copy(),
                            'qvel': self.mj_data.qvel.copy()
                        },
                        'data_per_cycle': []
                    }

            else:
                raise ValueError("从策略节点收到无效的握手响应")

        except Exception as e:
            print(f"❌ 握手失败: {e}")
            self.stop()
            return
        
        # --- STAGE 2: 根据模式选择执行路径 ---
        if self.seeref:
            # 回放模式
            print("\n--- STAGE 2: Reference Trajectory Playback ---")
            self.play_reference_trajectory(received_trajectory)
            return # 播放完后直接退出

        # --- STAGE 2: MAIN CONTROL LOOP ---
        print("\n--- STAGE 2: Main Control Loop ---")
        # 发送第一个真实状态以启动控制循环
        print("📤 已发送初始状态，等待第一个控制命令...")
        initial_state = self.get_robot_state()
        self.send_state_to_control(initial_state)
        
        try:
            while self.running.is_set():
                if self.viewer and not self.viewer.is_running():
                    break
                
                print(f"🔒 Cycle #{self.cycle_id}: 等待控制命令...")
                new_controls = None
                while new_controls is None and self.running.is_set():
                    new_controls = self.recv_controls_from_control_blocking()
                    if new_controls is None:
                        time.sleep(0.001)
                
                if not self.running.is_set(): break
                    
                self.current_controls = new_controls
                print(f"✅ Cycle #{self.cycle_id}: 收到控制命令 shape={new_controls.shape}")
                
                state = self.get_robot_state()
                if not self.send_state_to_control(state):
                    print(f"❌ Cycle #{self.cycle_id}: 状态发送失败")
                    continue
                
                self.execute_simulation_steps(self.current_controls)

                if self.log_enabled:
                    cycle_log = {
                        'cycle_id': self.cycle_id,
                        'time': state['time'],
                        'qpos': state['qpos'],
                        'qvel': state['qvel'],
                        'pd_targets': self.current_controls.copy(),
                        'qacc': self.mj_data.qacc.copy(),
                        'actuator_force': self.mj_data.actuator_force.copy()
                    }
                    self.log_data['data_per_cycle'].append(cycle_log)

                self.cycle_id += 1
                        
        except KeyboardInterrupt:
            print("\n🛑 收到中断信号...")
        finally:
            self.stop()
    
    def stop(self):
        """停止桥接器"""
        print("🛑 停止仿真桥接器...")
        self.running.clear()

        if self.log_enabled and hasattr(self, 'log_data') and self.log_data:
            os.makedirs("logs", exist_ok=True)
            filename = f"logs/sim_log_{self.log_data['metadata']['timestamp']}.pkl"
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(self.log_data, f)
                print(f"✅ 仿真日志已保存至 {filename}")
            except Exception as e:
                print(f"❌ 保存日志文件失败: {e}")

        # 如果在seeref模式下修改了颜色，则恢复
        if self.seeref and hasattr(self, 'original_geom_rgba'):
            print("   Restoring original model colors.")
            self.mj_model.geom_rgba[:] = self.original_geom_rgba

        if self.viewer:
            self.viewer.close()
        self.socket_state.close()
        self.socket_ctrl.close()
        self.context.term()
        print("✅ 仿真桥接器已停止")

def signal_handler(sig, frame):
    print("\n🛑 收到停止信号")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="ZeroMQ-DDS 仿真桥接器")
    parser.add_argument(
        "--init_channel", type=str, default="enp7s0",
        help="用于获取机器人初始状态的DDS网络接口 (例如 'enp7s0')。如果未提供，则使用随机化初始状态。"
    )
    parser.add_argument("--zmq_state_port", type=int, default=5555, help="ZeroMQ 状态端口")
    parser.add_argument("--zmq_ctrl_port", type=int, default=5556, help="ZeroMQ 控制端口")
    parser.add_argument("--frequency", type=float, default=50.0, help="控制频率 (Hz)")
    parser.add_argument("--seeref", action="store_true", help="启动参考轨迹查看模式，不进行仿真控制")
    parser.add_argument("--offset_x", type=float, default=1.0, help="初始位置X轴偏移量 (米)")
    parser.add_argument("--offset_y", type=float, default=-1.5, help="初始位置Y轴偏移量 (米)")
    parser.add_argument("--offset_yaw", type=float, default=-20.0, help="初始朝向Yaw偏移量 (度)")
    parser.add_argument("--log", action="store_true", help="启用仿真数据日志记录")
    
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    bridge = ZMQSimulationBridge(
        init_channel=args.init_channel,
        zmq_state_port=args.zmq_state_port,
        zmq_ctrl_port=args.zmq_ctrl_port,
        control_frequency=args.frequency,
        seeref=args.seeref,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        offset_yaw=args.offset_yaw,
        log=args.log
    )
    
    bridge.run()

if __name__ == "__main__":
    main() 