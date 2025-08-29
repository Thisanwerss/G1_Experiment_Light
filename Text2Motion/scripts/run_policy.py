#!/usr/bin/env python3
"""
Run NN policy that predicts knots.
Run:
    python run_policy.py --model_path overfit_model.pth --show_reference
"""
import argparse
import os
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer
import jax
import jax.numpy as jnp
from mujoco import mjx
import pytorch_lightning as pl


from hydrax.algs import CEM
from hydrax.tasks.humanoid_standonly import HumanoidStand
from replay_episode import replay_knots

# Define the MLP regressor model
class MLPRegressor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim=1, learning_rate=1e-3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            
            nn.Linear(hidden_dim3, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def load_model(model_path, device):
    """Load the model from disk."""
    checkpoint = torch.load(model_path, map_location=device)
    
    net = MLPRegressor(95, 512, 512, 512, 164)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device).eval()

    print("Model loaded successfully")
    
    return net
    

def predict_knots(net, qpos, qvel, device):
    """
    Run NN knots prediction
    """
    with torch.no_grad():
        net.eval()  # Ensure network is in evaluation mode

        # Network inference
        state = np.concatenate([qpos, qvel], axis=0).astype(np.float32)
        state = torch.from_numpy(state).to(device).float().unsqueeze(0)

        knots = net(state)
        knots = knots.squeeze(0).cpu().numpy().reshape(4, 41)

        return knots


class FrequencyMonitor:
    """专门监控各种频率的类"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        
        # 时间戳记录
        self.system_cycle_times = []         # 系统总循环时间戳
        self.knots_update_times = []         # knots更新时间戳
        self.pd_target_update_times = []     # PD目标设置时间戳
        self.simulation_step_times = []      # 仿真步骤时间戳
        
        # 计数器
        self.total_system_cycles = 0
        self.total_knots_updates = 0
        self.total_pd_updates = 0
        self.total_sim_steps = 0
        
        # 开始时间
        self.start_time = None
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        print("🔍 === 频率监控系统启动 ===")
        print("监控内容:")
        print("  - 系统循环频率 (总控制循环)")
        print("  - Knots更新频率 (NN+CEM规划)")  
        print("  - PD目标更新频率 (控制量设置)")
        print("  - 仿真步骤频率 (MuJoCo步进)")
        print("=" * 50)
        
    def record_system_cycle(self):
        """记录系统循环"""
        current_time = time.time()
        self.system_cycle_times.append(current_time)
        self.total_system_cycles += 1
        
    def record_knots_update(self):
        """记录knots更新"""
        current_time = time.time()
        self.knots_update_times.append(current_time)
        self.total_knots_updates += 1
        
    def record_pd_target_update(self):
        """记录PD目标更新"""
        current_time = time.time()
        self.pd_target_update_times.append(current_time)
        self.total_pd_updates += 1
        
    def record_simulation_step(self):
        """记录仿真步骤"""
        current_time = time.time()
        self.simulation_step_times.append(current_time)
        self.total_sim_steps += 1
        
    def calculate_frequency(self, timestamps):
        """计算频率"""
        if len(timestamps) < 2:
            return 0.0, 0.0
            
        # 最近window_size个时间戳
        recent_times = timestamps[-self.window_size:] if len(timestamps) >= self.window_size else timestamps
        
        if len(recent_times) < 2:
            return 0.0, 0.0
            
        # 计算时间间隔
        intervals = np.diff(recent_times)
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # 转换为频率
        freq = 1.0 / avg_interval if avg_interval > 0 else 0.0
        freq_std = std_interval / (avg_interval**2) if avg_interval > 0 else 0.0
        
        return freq, freq_std
        
    def get_overall_frequencies(self):
        """获取总体平均频率"""
        if self.start_time is None:
            return {}
            
        elapsed = time.time() - self.start_time
        
        return {
            'system_cycles': self.total_system_cycles / elapsed if elapsed > 0 else 0.0,
            'knots_updates': self.total_knots_updates / elapsed if elapsed > 0 else 0.0,
            'pd_updates': self.total_pd_updates / elapsed if elapsed > 0 else 0.0,
            'sim_steps': self.total_sim_steps / elapsed if elapsed > 0 else 0.0,
        }
        
    def print_detailed_report(self, target_frequency=50.0):
        """打印详细的频率报告"""
        # 计算瞬时频率
        system_freq, system_std = self.calculate_frequency(self.system_cycle_times)
        knots_freq, knots_std = self.calculate_frequency(self.knots_update_times)
        pd_freq, pd_std = self.calculate_frequency(self.pd_target_update_times)
        sim_freq, sim_std = self.calculate_frequency(self.simulation_step_times)
        
        # 获取总体频率
        overall_freqs = self.get_overall_frequencies()
        
        print(f"\n🔍 === 频率监控报告 ===")
        print(f"📊 瞬时频率 (最近{min(len(self.system_cycle_times), self.window_size)}个样本):")
        print(f"   🔄 系统循环频率: {system_freq:.2f} ± {system_std:.2f} Hz")
        print(f"   🧠 Knots更新频率: {knots_freq:.2f} ± {knots_std:.2f} Hz")
        print(f"   🎯 PD目标更新频率: {pd_freq:.2f} ± {pd_std:.2f} Hz")
        print(f"   ⚙️  仿真步骤频率: {sim_freq:.2f} ± {sim_std:.2f} Hz")
        
        print(f"\n📈 总体平均频率:")
        print(f"   🔄 系统循环频率: {overall_freqs['system_cycles']:.2f} Hz")
        print(f"   🧠 Knots更新频率: {overall_freqs['knots_updates']:.2f} Hz")
        print(f"   🎯 PD目标更新频率: {overall_freqs['pd_updates']:.2f} Hz")
        print(f"   ⚙️  仿真步骤频率: {overall_freqs['sim_steps']:.2f} Hz")
        
        print(f"\n🎯 目标对比 (目标: {target_frequency:.1f} Hz):")
        print(f"   系统循环达成率: {system_freq/target_frequency*100:.1f}%")
        print(f"   Knots更新达成率: {knots_freq/target_frequency*100:.1f}%")
        
        # 解耦分析
        print(f"\n🔗 解耦分析:")
        mujoco_target_freq = 100.0  # MuJoCo 10ms时间步
        expected_sim_freq = system_freq * 2  # 每个系统循环2个仿真步
        
        sim_decoupling_ratio = sim_freq / expected_sim_freq if expected_sim_freq > 0 else 0.0
        
        if abs(sim_decoupling_ratio - 1.0) < 0.1:
            print(f"   ✅ 仿真与控制耦合良好: {sim_decoupling_ratio:.2f}")
        else:
            print(f"   ⚠️  仿真与控制存在异常: {sim_decoupling_ratio:.2f}")
            
        if abs(system_freq - knots_freq) < 1.0:
            print(f"   ✅ 系统循环与Knots更新同步")
        else:
            print(f"   ⚠️  系统循环与Knots更新异步: {abs(system_freq - knots_freq):.2f} Hz差异")
            
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Replay episode using NN-predicted knots instead of recorded ones."
    )
    parser.add_argument(
        "--model_path", type=str, required=False, default="nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt",
        help="Path to the overfit model checkpoint (overfit_model.pth)."
    )
    parser.add_argument(
        "--show_reference", action="store_true",
        help="Show the reference trajectory as a transparent ghost."
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Use the network to warm-start the sampling approach"
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
        help="Planning frequency.",
    )

    # CEM parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of rollouts to perform.",
    )
    parser.add_argument(
        "--num_elites",
        type=int,
        default=20,
        help="Number of elites.",
    )
    parser.add_argument(
        "--plan_horizon",
        type=float,
        default=0.5,
        help="Horizon length.",
    )
    parser.add_argument(
        "--num_knots",
        type=int,
        default=4,
        help="Number of spline knots.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Sampling iterations to perform.",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Disable viewer/rendering for performance testing.",
    )

    args = parser.parse_args()

    # Setup device and load network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    net = load_model(args.model_path, device)

    # 初始化频率监控器
    freq_monitor = FrequencyMonitor(window_size=20)
    freq_monitor.start_monitoring()

    # Setup Mujoco model and data
    task = HumanoidStand()
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.01
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 50
    mj_model.opt.noslip_iterations = 2
    mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
    mj_data = mujoco.MjData(mj_model)
    reference = task.reference if args.show_reference else None

    # Instantiate CEM controller (only for interpolation)
    ctrl = CEM(
        task,
        num_samples=args.num_samples, 
        num_elites=args.num_elites,
        sigma_start=0.3, 
        sigma_min=0.05,
        explore_fraction=0.3,
        plan_horizon=args.plan_horizon,
        spline_type="zero",
        num_knots=args.num_knots,
        iterations=args.iterations
    )
    print("Starting simulation")

    # Figure out timing
    replan_period = 1.0 / args.frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    
    print(f"⚙️  仿真参数:")
    print(f"   目标频率: {args.frequency} Hz")
    print(f"   重规划周期: {step_dt:.4f}s")
    print(f"   每周期仿真步数: {sim_steps_per_replan}")
    print(f"   MuJoCo时间步: {mj_model.opt.timestep:.4f}s")
    
    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    initial_knots = predict_knots(net, mj_data.qpos, mj_data.qvel, device) if args.hybrid else None
    policy_params = ctrl.init_params(initial_knots=initial_knots)
    jit_optimize = jax.jit(ctrl.optimize)
    jit_interp_func = jax.jit(ctrl.interp_func)

    # Warm-up JIT
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)

    # Viewer and ghost reference setup
    viewer = None
    ref_data = None
    vopt = None
    pert = None
    catmask = None
    
    if not args.no_viewer:
        # Ghost reference setup
        if reference is not None:
            ref_data = mujoco.MjData(mj_model)
            ref_data.qpos[:] = reference[0, :]
            mujoco.mj_forward(mj_model, ref_data)
            vopt = mujoco.MjvOption()
            vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            pert = mujoco.MjvPerturb()
            catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        
        # Add ghost reference geometry
        if reference is not None:
            mujoco.mjv_addGeoms(mj_model, ref_data, vopt, pert, catmask, viewer.user_scn)
    else:
        print("🚫 渲染已禁用 - 性能测试模式")

    start_time = time.time()

    # Set up termination condition
    max_cycles = 1000 if args.no_viewer else None  # 限制无viewer模式的循环次数
    
    while True:
        # 🔄 记录系统循环开始
        freq_monitor.record_system_cycle()
        cycle_start_time = time.time()
        
        # Check termination conditions
        if viewer is not None and not viewer.is_running():
            break
        if args.no_viewer and freq_monitor.total_system_cycles >= max_cycles:
            print(f"🏁 无viewer模式达到最大循环次数 ({max_cycles})")
            break
        
        # Update controller state
        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
            time=mj_data.time,
        )

        # Planning step - 🧠 记录knots更新
        plan_start = time.time()
        freq_monitor.record_knots_update()
        
        if args.hybrid:
            # Neural network prediction
            new_knots = predict_knots(net, mj_data.qpos, mj_data.qvel, device)
            
            # Update policy parameters  
            policy_params = policy_params.replace(mean=new_knots)
            
            # CEM optimization
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            
        else:
            # Neural network prediction only
            new_knots = predict_knots(net, mj_data.qpos, mj_data.qvel, device)
            
            # Update policy parameters
            policy_params = policy_params.replace(mean=new_knots)
        
        plan_time = time.time() - plan_start

        # Update ghost reference
        if viewer is not None and reference is not None:
            t_ref = mj_data.time * 30
            i_ref = min(int(t_ref), reference.shape[0] - 1)
            ref_data.qpos[:] = reference[i_ref]
            mujoco.mj_forward(mj_model, ref_data)
            mujoco.mjv_updateScene(
                mj_model, ref_data, vopt, pert, viewer.cam, catmask, viewer.user_scn
            )

        # Interpolate controls
        sim_dt = mj_model.opt.timestep
        t_curr = mj_data.time

        # Querying parameters
        us = None
        if args.hybrid:
            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]
        else:
            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt
            tk = jnp.linspace(0, 0.5, 4)
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]

        # Simulate - 详细监控每个仿真步骤和PD目标更新
        for i in range(sim_steps_per_replan):
            # 🎯 记录PD目标更新 (控制量设置)
            freq_monitor.record_pd_target_update()
            mj_data.ctrl[:] = np.array(us[i])
            
            # ⚙️ 记录仿真步骤
            freq_monitor.record_simulation_step()
            mujoco.mj_step(mj_model, mj_data)

            # Viewer sync timing
            if viewer is not None:
                viewer.sync()

        # 频率监控报告 - 每20个循环详细报告一次
        if freq_monitor.total_system_cycles % 20 == 0:
            freq_monitor.print_detailed_report(args.frequency)
            
        # 简化的即时状态显示 - 每5个循环
        if freq_monitor.total_system_cycles % 5 == 0:
            elapsed_total = time.time() - start_time
            instant_system_freq = freq_monitor.total_system_cycles / elapsed_total
            print(f"⚡ 即时状态 - 循环#{freq_monitor.total_system_cycles}: "
                  f"系统频率={instant_system_freq:.1f}Hz, "
                  f"规划耗时={plan_time*1000:.1f}ms")

        # Timing control - sleep to maintain desired frequency
        cycle_elapsed = time.time() - cycle_start_time
        target_cycle_time = 1.0 / args.frequency
        if cycle_elapsed < target_cycle_time:
            time.sleep(target_cycle_time - cycle_elapsed)

    # 最终频率监控报告
    print(f"\n🏁 === 最终频率监控报告 ===")
    freq_monitor.print_detailed_report(args.frequency)
    
    elapsed_total = time.time() - start_time
    print(f"\n📊 运行总结:")
    print(f"   总运行时间: {elapsed_total:.2f}s")
    print(f"   系统循环总数: {freq_monitor.total_system_cycles}")
    print(f"   Knots更新总数: {freq_monitor.total_knots_updates}")
    print(f"   PD目标更新总数: {freq_monitor.total_pd_updates}")
    print(f"   仿真步骤总数: {freq_monitor.total_sim_steps}")

if __name__ == "__main__":
    main()
