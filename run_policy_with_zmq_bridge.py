#!/usr/bin/env python3
"""
修改版的策略控制节点 - 与 SDK 桥接器通信
将原本发送到仿真的控制命令改为发送到 SDK 桥接器

工作流程：
1. 从 DDS 接收机器人状态（通过 SDK 桥接器）
2. 运行 NN+CEM 策略计算
3. 将 PD 目标发送到 SDK 桥接器（而不是直接到仿真）
4. SDK 桥接器负责转换并发送到机器人/仿真
"""

import argparse
import time
import pickle
from typing import Tuple, Optional, Dict, Any, List
import struct

import numpy as np
import torch
import torch.nn as nn
import mujoco
import jax
import jax.numpy as jnp
from mujoco import mjx
import pytorch_lightning as pl
import zmq

from hydrax.algs import CEM
from hydrax.tasks.humanoid_standonly import HumanoidStand

# 导入原始策略代码中的类
import sys
sys.path.append('.')  # 确保可以导入当前目录的模块
from run_policy_pruned import MLPRegressor, load_model, predict_knots, OutlierFilteredStats


class PolicyZMQBridge:
    """策略与 SDK 桥接器之间的通信接口"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        zmq_bridge_port: int = 5557,
        zmq_state_port: int = 5558,  # 接收状态的端口
        frequency: float = 50.0,
        # CEM 参数
        num_samples: int = 500,
        num_elites: int = 20,
        sigma_start: float = 0.3,
        sigma_min: float = 0.05,
        plan_horizon: float = 0.5,
        num_knots: int = 4,
        iterations: int = 1
    ):
        """初始化策略-桥接器通信"""
        self.device = torch.device(device)
        self.frequency = frequency
        self.replan_period = 1.0 / frequency
        self.zmq_bridge_port = zmq_bridge_port
        self.zmq_state_port = zmq_state_port
        
        print(f"🚀 初始化策略-桥接器通信")
        print(f"   设备: {self.device}")
        print(f"   策略频率: {frequency} Hz")
        print(f"   桥接器端口: {zmq_bridge_port} (策略 PUSH → 桥接器 PULL)")
        print(f"   状态端口: {zmq_state_port} (桥接器 PUSH → 策略 PULL)")
        
        # 1. 加载 PyTorch 网络
        print("📦 加载神经网络...")
        self.net = load_model(model_path, self.device)
        
        # 2. 设置任务和模型（使用 g1_lab.xml）
        print("🤖 设置 G1 机器人任务...")
        # 修改为使用 g1_lab.xml
        self.task = HumanoidStand()
        # 尝试加载 g1_lab.xml
        try:
            from hydrax import ROOT
            g1_lab_path = ROOT + "/models/g1/g1_lab.xml"
            self.mj_model = mujoco.MjModel.from_xml_path(g1_lab_path)
            print(f"✅ 加载 g1_lab.xml: {g1_lab_path}")
        except:
            print("⚠️ 无法加载 g1_lab.xml，使用默认模型")
            self.mj_model = self.task.mj_model
        
        # 配置 MuJoCo 参数
        self.mj_model.opt.timestep = 0.01  # 100Hz 仿真
        self.mj_model.opt.iterations = 10
        self.mj_model.opt.ls_iterations = 50
        self.mj_model.opt.noslip_iterations = 2
        self.mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
        self.mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
        
        # 3. 设置 CEM 控制器
        print("🧠 设置 CEM 控制器...")
        self.ctrl = CEM(
            self.task,
            num_samples=num_samples, 
            num_elites=num_elites,
            sigma_start=sigma_start, 
            sigma_min=sigma_min,
            explore_fraction=0.3,
            plan_horizon=plan_horizon,
            spline_type="zero",
            num_knots=num_knots,
            iterations=iterations
        )
        
        # 4. 预计算仿真参数
        # 50Hz 策略 -> 100Hz 控制，每个策略周期需要 2 个控制目标
        self.control_steps_per_plan = 2
        self.sim_steps_per_replan = max(int(self.replan_period / self.mj_model.opt.timestep), 1)
        self.step_dt = self.sim_steps_per_replan * self.mj_model.opt.timestep
        
        print(f"   每个策略周期的仿真步数: {self.sim_steps_per_replan}")
        print(f"   规划周期: {self.step_dt:.4f}s")
        print(f"   每个策略周期的控制目标数: {self.control_steps_per_plan}")
        
        # 5. 初始化虚拟状态和 JAX 数据
        print("🎭 初始化虚拟状态...")
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)
        self.mjx_data = self.mjx_data.replace(
            mocap_pos=self.mj_data.mocap_pos, 
            mocap_quat=self.mj_data.mocap_quat
        )
        
        # 6. 初始化策略参数
        initial_knots = predict_knots(self.net, self.mj_data.qpos, self.mj_data.qvel, self.device)
        self.policy_params = self.ctrl.init_params(initial_knots=initial_knots)
        
        # 7. 预编译 JAX 函数
        print("⚡ 预编译 JAX 函数...")
        self.jit_optimize = jax.jit(self.ctrl.optimize)
        self.jit_interp_func = jax.jit(self.ctrl.interp_func)
        
        # JIT 预热
        print("🔥 预热 JIT...")
        self.policy_params, rollouts = self.jit_optimize(self.mjx_data, self.policy_params)
        tq = jnp.arange(0, self.sim_steps_per_replan) * self.mj_model.opt.timestep
        tk = self.policy_params.tk
        knots = self.policy_params.mean[None, ...]
        _ = self.jit_interp_func(tq, tk, knots)
        print("✅ JIT 预热完成")
        
        # 8. 设置 ZeroMQ 通信
        print("🌐 设置 ZeroMQ 通信...")
        self.context = zmq.Context()
        
        # 发送 PD 目标到桥接器的 socket (PUSH)
        self.socket_bridge = self.context.socket(zmq.PUSH)
        self.socket_bridge.setsockopt(zmq.SNDHWM, 10)
        self.socket_bridge.setsockopt(zmq.SNDBUF, 1048576)
        self.socket_bridge.setsockopt(zmq.LINGER, 0)
        self.socket_bridge.bind(f"tcp://*:{zmq_bridge_port}")
        
        # 接收状态的 socket (PULL) - 从 DDS 高级状态
        self.socket_state = self.context.socket(zmq.PULL)
        self.socket_state.setsockopt(zmq.RCVHWM, 10)
        self.socket_state.setsockopt(zmq.RCVBUF, 1048576)
        self.socket_state.setsockopt(zmq.LINGER, 0)
        self.socket_state.bind(f"tcp://*:{zmq_state_port}")
        
        # Poller 设置
        self.poller = zmq.Poller()
        self.poller.register(self.socket_state, zmq.POLLIN)
        
        print(f"✅ ZeroMQ 通信设置完成")
        
        # 9. 状态管理
        self.running = False
        self.current_state = None
        
        # 10. 统计信息
        self.timing_history = []
        self.compute_stats = OutlierFilteredStats()
        self.send_stats = OutlierFilteredStats()
    
    def recv_robot_state(self, timeout_ms: int = 100) -> Optional[Dict[str, Any]]:
        """接收机器人状态（从 DDS 或其他来源）"""
        try:
            # 检查是否有状态数据
            socks = dict(self.poller.poll(timeout_ms))
            
            if self.socket_state in socks:
                # 接收状态
                message = self.socket_state.recv(zmq.NOBLOCK)
                state = pickle.loads(message)
                return state
            
            return None
            
        except zmq.Again:
            return None
        except Exception as e:
            print(f"❌ 接收状态错误: {e}")
            return None
    
    def send_pd_targets(self, pd_targets: List[np.ndarray]) -> bool:
        """发送 PD 目标到桥接器"""
        try:
            # 准备消息
            message = {
                'pd_targets': pd_targets,  # 两个 PD 目标用于插值
                'timestamp': time.time()
            }
            
            # 序列化并发送
            message_bytes = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
            self.socket_bridge.send(message_bytes, zmq.NOBLOCK)
            
            return True
            
        except zmq.Again:
            print(f"⚠️ PD 目标发送队列已满")
            return False
        except Exception as e:
            print(f"❌ 发送 PD 目标错误: {e}")
            return False
    
    def compute_pd_targets(
        self, 
        qpos: np.ndarray, 
        qvel: np.ndarray,
        current_time: float = 0.0
    ) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """计算 PD 目标（核心控制逻辑）"""
        t_start = time.time()
        timing_info = {}
        
        # 1. 神经网络预测
        nn_start = time.time()
        predicted_knots = predict_knots(self.net, qpos, qvel, self.device)
        timing_info['nn_time'] = time.time() - nn_start
        
        # 2. 准备 JAX 数据
        prep_start = time.time()
        mjx_data = self.mjx_data.replace(
            qpos=jnp.array(qpos),
            qvel=jnp.array(qvel),
            time=current_time
        )
        
        # 更新策略参数
        policy_params = self.policy_params.replace(mean=predicted_knots)
        timing_info['prep_time'] = time.time() - prep_start
        
        # 3. CEM 优化
        cem_start = time.time()
        policy_params, rollouts = self.jit_optimize(mjx_data, policy_params)
        timing_info['cem_time'] = time.time() - cem_start
        
        # 4. 插值生成控制序列
        interp_start = time.time()
        tq = jnp.arange(0, self.sim_steps_per_replan) * self.mj_model.opt.timestep + current_time
        tk = policy_params.tk
        knots = policy_params.mean[None, ...]
        controls_jax = self.jit_interp_func(tq, tk, knots)[0]
        
        # 转换为 numpy
        controls = np.asarray(controls_jax)
        timing_info['interp_time'] = time.time() - interp_start
        
        # 5. 提取两个 PD 目标（用于 50Hz -> 100Hz 插值）
        # 选择第 0 和第 1 个时间步的控制目标
        pd_targets = []
        for i in range(min(self.control_steps_per_plan, len(controls))):
            pd_targets.append(controls[i].copy())
        
        # 如果不足两个，复制最后一个
        while len(pd_targets) < self.control_steps_per_plan:
            pd_targets.append(pd_targets[-1].copy())
        
        timing_info['total_time'] = time.time() - t_start
        
        # 更新内部策略参数
        self.policy_params = policy_params
        
        return pd_targets, timing_info
    
    def get_default_state(self) -> Dict[str, Any]:
        """获取默认状态（用于测试）"""
        # G1 的默认站立姿态
        qpos = np.zeros(48)  # 7 (浮动基座) + 41 (关节)
        qpos[2] = 0.75  # z 高度
        qpos[3] = 1.0   # 四元数 w
        
        # 腿部略微弯曲
        # 左腿
        qpos[10] = -0.3  # 膝盖
        qpos[11] = 0.3   # 踝关节
        # 右腿
        qpos[16] = -0.3  # 膝盖
        qpos[17] = 0.3   # 踝关节
        
        qvel = np.zeros(47)  # 6 (浮动基座) + 41 (关节)
        
        return {
            'qpos': qpos,
            'qvel': qvel,
            'time': 0.0
        }
    
    def run(self):
        """主运行循环"""
        print("🚀 启动策略-桥接器通信服务...")
        print("💡 工作模式:")
        print("   - 从 DDS/状态源接收机器人状态")
        print("   - 运行 NN+CEM 计算 PD 目标")
        print("   - 发送 PD 目标到 SDK 桥接器")
        print("   - SDK 桥接器负责 100Hz 插值和电机控制")
        
        self.running = True
        cycle_count = 0
        
        # 使用默认状态初始化
        current_state = self.get_default_state()
        
        # 预热
        print("🔥 预热控制器...")
        for i in range(10):
            pd_targets, _ = self.compute_pd_targets(
                current_state['qpos'], 
                current_state['qvel'],
                i * self.replan_period
            )
        print("✅ 预热完成")
        
        print("\n🎯 等待机器人状态或使用默认状态...")
        
        try:
            while self.running:
                cycle_start = time.time()
                
                # 尝试接收新状态
                new_state = self.recv_robot_state(timeout_ms=5)
                if new_state is not None:
                    current_state = new_state
                    if cycle_count == 0:
                        print("✅ 收到机器人状态，开始控制")
                
                # 计算 PD 目标
                t_compute_start = time.time()
                pd_targets, timing_info = self.compute_pd_targets(
                    current_state['qpos'],
                    current_state['qvel'],
                    current_state.get('time', cycle_count * self.replan_period)
                )
                t_compute_end = time.time()
                compute_time = t_compute_end - t_compute_start
                self.compute_stats.add_sample(compute_time)
                
                # 发送 PD 目标
                t_send_start = time.time()
                success = self.send_pd_targets(pd_targets)
                t_send_end = time.time()
                send_time = t_send_end - t_send_start
                self.send_stats.add_sample(send_time)
                
                if success:
                    if cycle_count % 50 == 0:  # 每秒打印一次
                        print(f"📤 周期 {cycle_count}: 发送 PD 目标")
                        print(f"   计算时间: {compute_time*1000:.1f}ms")
                        print(f"   发送时间: {send_time*1000:.1f}ms")
                
                # 周期性打印统计
                if cycle_count > 0 and cycle_count % 100 == 0:
                    self.print_stats()
                
                cycle_count += 1
                
                # 频率控制
                cycle_elapsed = time.time() - cycle_start
                sleep_time = self.replan_period - cycle_elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
                elif sleep_time < -0.005 and cycle_count % 50 == 0:
                    print(f"⚠️ 周期 {cycle_count}: 延迟 {-sleep_time*1000:.1f}ms")
                    
        except KeyboardInterrupt:
            print("\n🛑 收到中断信号，停止服务...")
        finally:
            self.stop()
    
    def print_stats(self):
        """打印统计信息"""
        print(f"\n🔍 === 策略统计报告 ===")
        
        # 计算统计
        compute_mean, compute_std, _, _ = self.compute_stats.get_filtered_stats()
        print(f"🧠 策略计算: {compute_mean*1000:.2f}±{compute_std*1000:.2f}ms")
        
        # 发送统计
        send_mean, send_std, _, _ = self.send_stats.get_filtered_stats()
        print(f"📤 目标发送: {send_mean*1000:.2f}±{send_std*1000:.2f}ms")
        
        # 总延迟
        total_mean = compute_mean + send_mean
        print(f"⏱️  总延迟: {total_mean*1000:.2f}ms")
        
        # 估计频率
        if total_mean > 0:
            estimated_freq = 1.0 / total_mean
            print(f"📈 估计频率: {estimated_freq:.1f} Hz (目标: {self.frequency:.1f} Hz)")
        
        print("=" * 40)
    
    def stop(self):
        """停止服务"""
        self.running = False
        self.socket_bridge.close()
        self.socket_state.close()
        self.context.term()
        
        # 打印最终统计
        print(f"\n🏁 === 最终统计报告 ===")
        self.print_stats()
        print("✅ 策略服务已停止")


def main():
    parser = argparse.ArgumentParser(
        description="策略-SDK 桥接器通信节点"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=False, 
        default="nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt",
        help="PyTorch 模型路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="PyTorch 设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
        help="策略频率 (Hz)"
    )
    parser.add_argument(
        "--zmq_bridge_port",
        type=int,
        default=5557,
        help="桥接器端口 (策略 PUSH → 桥接器 PULL)"
    )
    parser.add_argument(
        "--zmq_state_port",
        type=int,
        default=5558,
        help="状态端口 (桥接器 PUSH → 策略 PULL)"
    )
    
    # CEM 参数
    parser.add_argument("--num_samples", type=int, default=500, help="CEM 样本数")
    parser.add_argument("--num_elites", type=int, default=20, help="CEM 精英数")
    parser.add_argument("--sigma_start", type=float, default=0.3, help="初始标准差")
    parser.add_argument("--plan_horizon", type=float, default=0.5, help="规划时域")
    parser.add_argument("--num_knots", type=int, default=4, help="样条节点数")
    parser.add_argument("--iterations", type=int, default=1, help="CEM 迭代次数")

    args = parser.parse_args()

    # 创建并运行策略桥接器
    bridge = PolicyZMQBridge(
        model_path=args.model_path,
        device=args.device,
        frequency=args.frequency,
        zmq_bridge_port=args.zmq_bridge_port,
        zmq_state_port=args.zmq_state_port,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        sigma_start=args.sigma_start,
        plan_horizon=args.plan_horizon,
        num_knots=args.num_knots,
        iterations=args.iterations
    )
    
    bridge.run()


if __name__ == "__main__":
    main() 