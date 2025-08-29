#!/usr/bin/env python3
"""
JIT预编译版本的run_policy pipeline

这个版本通过以下方式消除JIT开销：
1. 深度预热所有JAX函数
2. 缓存编译状态
3. 支持多种输入模式的预编译

虽然不是真正的AOT编译，但可以将首次使用的JIT开销降到最低。

运行方法:
    python export_run_policy_aot.py --model_path nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt
"""

import argparse
import os
import pickle
import time
import tempfile
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import mujoco
import jax
import jax.numpy as jnp
from mujoco import mjx
import pytorch_lightning as pl

# 导入原始的CEM控制器和任务
from hydrax.algs import CEM
from hydrax.tasks.humanoid_standonly import HumanoidStand


class MLPRegressor(pl.LightningModule):
    """神经网络模型定义（与原始保持一致）"""
    
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


class PrecompiledRunPolicyPipeline:
    """JIT预编译版本的run_policy pipeline
    
    特点：
    1. 深度预热所有JAX函数路径
    2. 消除运行时JIT编译开销
    3. 接近静态的性能表现
    4. 简单可靠的实现
    """
    
    def __init__(
        self,
        pytorch_network: nn.Module,
        jax_task: Any,
        jax_controller: Any,
        device: str = 'cuda',
        # 仿真参数
        frequency: float = 50.0,
        plan_horizon: float = 0.5,
        sim_steps_per_replan: int = 2,
        step_dt: float = 0.02,
        # 预编译参数
        precompile_depth: int = 3
    ):
        """初始化预编译pipeline
        
        Args:
            pytorch_network: 训练好的PyTorch网络
            jax_task: JAX任务对象
            jax_controller: JAX CEM控制器
            device: PyTorch设备
            其他参数: 仿真参数和预编译深度
        """
        self.device = torch.device(device)
        self.pytorch_network = pytorch_network.to(self.device).eval()
        self.jax_task = jax_task
        self.ctrl = jax_controller
        
        # 仿真参数
        self.frequency = frequency
        self.plan_horizon = plan_horizon
        self.sim_steps_per_replan = sim_steps_per_replan
        self.step_dt = step_dt
        
        print(f"预编译Pipeline初始化:")
        print(f"  - 规划频率: {frequency} Hz")
        print(f"  - 设备: {self.device}")
        print(f"  - 预编译深度: {precompile_depth}")
        
        # 初始化MuJoCo环境
        self._setup_mujoco_env()
        
        # 深度预编译所有JAX函数
        self._deep_precompile(precompile_depth)
        
        print("预编译Pipeline初始化完成!")
    
    def _setup_mujoco_env(self):
        """设置MuJoCo环境"""
        self.mj_model = self.jax_task.mj_model
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # 初始化MJX数据
        self.mjx_data_template = mjx.put_data(self.mj_model, self.mj_data)
        self.mjx_data_template = self.mjx_data_template.replace(
            mocap_pos=self.mj_data.mocap_pos, 
            mocap_quat=self.mj_data.mocap_quat
        )
        
        # 初始化策略参数
        self.policy_params_template = self.ctrl.init_params(initial_knots=None)
    
    def _deep_precompile(self, depth: int):
        """深度预编译JAX函数"""
        print("开始深度预编译JAX函数...")
        
        # 1. 编译优化和插值函数
        print("  编译核心函数...")
        self.jit_optimize = jax.jit(self.ctrl.optimize)
        self.jit_interp_func = jax.jit(self.ctrl.interp_func)
        
        # 2. 多模式预热编译
        print(f"  进行{depth}轮多模式预热...")
        
        nq, nv = 48, 47  # G1机器人维度
        
        for round_idx in range(depth):
            print(f"    预热轮次 {round_idx + 1}/{depth}")
            
            # 生成多种输入模式
            variations = self._generate_input_variations(nq, nv, round_idx)
            
            for i, (qpos, qvel, time_val) in enumerate(variations):
                if i == 0:  # 详细输出第一个
                    start_time = time.time()
                
                # 预测knots
                predicted_knots = self.predict_knots_pytorch(qpos, qvel)
                
                # 更新MJX数据和策略参数
                mjx_data = self.mjx_data_template.replace(
                    qpos=jnp.array(qpos),
                    qvel=jnp.array(qvel),
                    time=time_val
                )
                policy_params = self.policy_params_template.replace(mean=predicted_knots)
                
                # CEM优化
                policy_params_out, rollouts = self.jit_optimize(mjx_data, policy_params)
                
                # 插值
                tq = jnp.arange(0, self.sim_steps_per_replan) * self.mj_model.opt.timestep + time_val
                tk = policy_params_out.tk
                knots = policy_params_out.mean[None, ...]
                controls = self.jit_interp_func(tq, tk, knots)[0]
                
                if i == 0:
                    elapsed = time.time() - start_time
                    print(f"      第一次执行耗时: {elapsed:.4f}s")
        
        print("  深度预编译完成!")
    
    def _generate_input_variations(self, nq: int, nv: int, round_idx: int):
        """生成多种输入变化以触发所有编译路径"""
        variations = []
        
        # 基本种子
        np.random.seed(42 + round_idx)
        
        # 不同幅度的随机状态
        scales = [0.001, 0.01, 0.1, 0.5, 1.0]
        for scale in scales:
            qpos = np.random.randn(nq) * scale
            qvel = np.random.randn(nv) * scale
            time_val = round_idx * 0.1 + np.random.rand() * 0.1
            variations.append((qpos, qvel, time_val))
        
        # 极值情况
        variations.append((np.zeros(nq), np.zeros(nv), 0.0))
        variations.append((np.ones(nq) * 0.1, np.ones(nv) * 0.1, 1.0))
        
        # 混合情况
        for _ in range(3):
            qpos = np.random.randn(nq) * 0.05
            qvel = np.random.randn(nv) * 0.05
            time_val = np.random.rand() * 2.0
            variations.append((qpos, qvel, time_val))
        
        return variations
    
    def predict_knots_pytorch(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """使用PyTorch网络预测knots"""
        with torch.no_grad():
            self.pytorch_network.eval()
            
            # 状态拼接和预处理
            state = np.concatenate([qpos, qvel], axis=0).astype(np.float32)
            state_tensor = torch.from_numpy(state).to(self.device).float().unsqueeze(0)
            
            # 网络推理
            knots_flat = self.pytorch_network(state_tensor)
            knots = knots_flat.squeeze(0).cpu().numpy().reshape(4, 41)
            
            return knots
    
    def predict_controls(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        mocap_pos: Optional[np.ndarray] = None,
        mocap_quat: Optional[np.ndarray] = None,
        current_time: float = 0.0,
        return_timing: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
        """预编译的控制预测（最小JIT开销版本）"""
        total_start = time.time()
        timing_info = {} if return_timing else None
        
        # 1. PyTorch神经网络预测
        if return_timing:
            nn_start = time.time()
        
        predicted_knots = self.predict_knots_pytorch(qpos, qvel)
        
        if return_timing:
            timing_info['nn_time'] = time.time() - nn_start
        
        # 2. 准备JAX数据
        if return_timing:
            prep_start = time.time()
        
        # 更新MJX数据
        mjx_data = self.mjx_data_template.replace(
            qpos=jnp.array(qpos),
            qvel=jnp.array(qvel),
            time=current_time
        )
        
        if mocap_pos is not None:
            mjx_data = mjx_data.replace(mocap_pos=jnp.array(mocap_pos))
        if mocap_quat is not None:
            mjx_data = mjx_data.replace(mocap_quat=jnp.array(mocap_quat))
        
        # 更新策略参数
        policy_params = self.policy_params_template.replace(mean=predicted_knots)
        
        if return_timing:
            timing_info['prep_time'] = time.time() - prep_start
        
        # 3. 预编译的CEM优化（最小JIT开销）
        if return_timing:
            cem_start = time.time()
        
        policy_params_out, rollouts = self.jit_optimize(mjx_data, policy_params)
        
        if return_timing:
            timing_info['cem_time'] = time.time() - cem_start
        
        # 4. 预编译的插值（最小JIT开销）
        if return_timing:
            interp_start = time.time()
        
        # 查询时间序列
        tq = jnp.arange(0, self.sim_steps_per_replan) * self.mj_model.opt.timestep + current_time
        tk = policy_params_out.tk
        knots = policy_params_out.mean[None, ...]
        controls_jax = self.jit_interp_func(tq, tk, knots)[0]
        
        # 转换为numpy
        controls = np.asarray(controls_jax)
        
        if return_timing:
            timing_info['interp_time'] = time.time() - interp_start
            timing_info['total_time'] = time.time() - total_start
        
        # 返回结果
        if return_timing:
            return controls, timing_info
        else:
            return controls, None
    
    def get_simulation_params(self) -> Dict[str, Any]:
        """获取仿真参数信息"""
        return {
            'frequency': self.frequency,
            'sim_steps_per_replan': self.sim_steps_per_replan,
            'step_dt': self.step_dt,
            'plan_horizon': self.plan_horizon,
            'mujoco_timestep': self.mj_model.opt.timestep
        }


def create_and_precompile_pipeline(
    model_path: str,
    device: str = 'cuda',
    # CEM参数
    num_samples: int = 500,
    num_elites: int = 20,
    sigma_start: float = 0.3,
    sigma_min: float = 0.05,
    plan_horizon: float = 0.5,
    num_knots: int = 4,
    iterations: int = 1,
    frequency: float = 50.0,
    # 预编译参数
    precompile_depth: int = 3
):
    """创建并预编译pipeline"""
    print(f"开始创建预编译pipeline...")
    
    # 1. 加载PyTorch网络
    print("加载PyTorch网络...")
    device_obj = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device_obj)
    
    pl_net = MLPRegressor(95, 512, 512, 512, 164)
    pl_net.load_state_dict(checkpoint['state_dict'])
    net = pl_net.model
    net.to(device_obj).eval()
    
    # 2. 设置MuJoCo任务和控制器
    print("设置任务和控制器...")
    task = HumanoidStand()
    mj_model = task.mj_model
    
    # 配置MuJoCo参数
    mj_model.opt.timestep = 0.01
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 50
    mj_model.opt.noslip_iterations = 2
    mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
    
    ctrl = CEM(
        task,
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
    
    # 3. 计算仿真参数
    sim_steps_per_replan = max(int((1.0 / frequency) / mj_model.opt.timestep), 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    
    # 4. 创建预编译pipeline
    print("创建预编译pipeline...")
    pipeline = PrecompiledRunPolicyPipeline(
        pytorch_network=net,
        jax_task=task,
        jax_controller=ctrl,
        device=device,
        frequency=frequency,
        plan_horizon=plan_horizon,
        sim_steps_per_replan=sim_steps_per_replan,
        step_dt=step_dt,
        precompile_depth=precompile_depth
    )
    
    return pipeline


def save_precompiled_pipeline(
    pipeline: PrecompiledRunPolicyPipeline,
    output_dir: str
):
    """保存预编译pipeline配置"""
    print(f"保存预编译pipeline配置到: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存网络和配置
    config = {
        'pytorch_state_dict': pipeline.pytorch_network.state_dict(),
        'device': str(pipeline.device),
        'frequency': pipeline.frequency,
        'plan_horizon': pipeline.plan_horizon,
        'sim_steps_per_replan': pipeline.sim_steps_per_replan,
        'step_dt': pipeline.step_dt,
        # 注意：MuJoCo模型无法直接序列化XML，我们保存任务类型
        'task_type': 'HumanoidStand',
        # CEM参数
        'num_samples': pipeline.ctrl.num_samples,
        'num_elites': pipeline.ctrl.num_elites,
        'sigma_start': pipeline.ctrl.sigma_start,
        'sigma_min': pipeline.ctrl.sigma_min,
        'num_knots': pipeline.ctrl.num_knots,
        'iterations': pipeline.ctrl.iterations,
    }
    
    config_path = os.path.join(output_dir, "config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    print(f"配置已保存!")


def load_precompiled_pipeline(compiled_dir: str, precompile_depth: int = 3) -> PrecompiledRunPolicyPipeline:
    """加载预编译pipeline"""
    print(f"加载预编译pipeline: {compiled_dir}")
    
    # 加载配置
    config_path = os.path.join(compiled_dir, "config.pkl")
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # 重建PyTorch网络
    device_obj = torch.device(config['device'])
    pl_net = MLPRegressor(95, 512, 512, 512, 164)
    net = pl_net.model
    net.load_state_dict(config['pytorch_state_dict'])
    net.to(device_obj).eval()
    
    # 重建JAX任务（目前只支持HumanoidStand）
    if config.get('task_type', 'HumanoidStand') == 'HumanoidStand':
        task = HumanoidStand()
    else:
        raise NotImplementedError(f"Task type {config['task_type']} not supported")
    
    mj_model = task.mj_model
    
    # 配置MuJoCo参数
    mj_model.opt.timestep = 0.01
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 50
    mj_model.opt.noslip_iterations = 2
    mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
    
    # 重建CEM控制器
    ctrl = CEM(
        task,
        num_samples=config['num_samples'], 
        num_elites=config['num_elites'],
        sigma_start=config['sigma_start'], 
        sigma_min=config['sigma_min'],
        explore_fraction=0.3,
        plan_horizon=config['plan_horizon'],
        spline_type="zero",
        num_knots=config['num_knots'],
        iterations=config['iterations']
    )
    
    # 创建预编译pipeline
    pipeline = PrecompiledRunPolicyPipeline(
        pytorch_network=net,
        jax_task=task,
        jax_controller=ctrl,
        device=config['device'],
        frequency=config['frequency'],
        plan_horizon=config['plan_horizon'],
        sim_steps_per_replan=config['sim_steps_per_replan'],
        step_dt=config['step_dt'],
        precompile_depth=precompile_depth
    )
    
    return pipeline


def test_precompiled_pipeline(pipeline: PrecompiledRunPolicyPipeline, num_tests: int = 20):
    """测试预编译pipeline性能"""
    print(f"\n🚀 测试预编译pipeline性能 (运行{num_tests}次)...")
    
    # 获取仿真参数
    sim_params = pipeline.get_simulation_params()
    print("仿真参数:")
    for key, value in sim_params.items():
        print(f"  {key}: {value}")
    
    nq, nv = 48, 47
    print(f"\n模型维度: nq={nq}, nv={nv}")
    
    # 性能测试
    times = []
    
    print("\n详细测试:")
    for i in range(num_tests):
        # 生成随机状态
        qpos = np.random.randn(nq) * 0.1
        qvel = np.random.randn(nv) * 0.1
        
        # 预测控制
        start_time = time.time()
        controls, timing_info = pipeline.predict_controls(
            qpos, qvel, 
            current_time=i * sim_params['step_dt'],
            return_timing=True
        )
        total_time = time.time() - start_time
        times.append(total_time)
        
        # 打印详细信息（前几次和后几次）
        if i < 3 or i >= num_tests - 3:
            print(f"\n测试 #{i+1}:")
            print(f"  输入: qpos{qpos.shape}, qvel{qvel.shape}")
            print(f"  输出: controls{controls.shape}")
            print(f"  总耗时: {total_time:.4f}s")
            if timing_info:
                print(f"    - NN推理: {timing_info['nn_time']:.4f}s")
                print(f"    - 数据准备: {timing_info['prep_time']:.4f}s")
                print(f"    - CEM优化: {timing_info['cem_time']:.4f}s")
                print(f"    - 插值: {timing_info['interp_time']:.4f}s")
    
    # 统计结果
    times = np.array(times)
    print(f"\n📊 预编译性能统计 ({num_tests}次测试):")
    print(f"  平均耗时: {np.mean(times):.4f}s")
    print(f"  最小耗时: {np.min(times):.4f}s")
    print(f"  最大耗时: {np.max(times):.4f}s")
    print(f"  标准差: {np.std(times):.4f}s")
    print(f"  理论最大频率: {1.0/np.mean(times):.2f} Hz")
    print(f"  目标频率: {sim_params['frequency']:.2f} Hz")
    print(f"  频率达成率: {min(1.0, sim_params['frequency'] / (1.0/np.mean(times))) * 100:.1f}%")
    
    # 性能一致性检查
    if np.std(times) < 0.005:
        print("✅ 预编译成功：性能非常一致，JIT开销已最小化")
    elif np.std(times) < 0.01:
        print("✅ 预编译基本成功：性能较为一致，JIT开销已大幅减少")
    else:
        print("⚠️ 性能波动较大，可能需要更深度的预编译")


def main():
    parser = argparse.ArgumentParser(description="JIT预编译版本的run_policy pipeline")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt",
        help="PyTorch模型checkpoint路径"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exported_models/precompiled",
        help="预编译输出目录"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch计算设备"
    )
    
    # CEM参数
    parser.add_argument("--num_samples", type=int, default=500, help="CEM样本数")
    parser.add_argument("--num_elites", type=int, default=20, help="CEM精英数")
    parser.add_argument("--frequency", type=float, default=50.0, help="规划频率")
    
    # 预编译参数
    parser.add_argument("--precompile_depth", type=int, default=3, help="预编译深度")
    
    parser.add_argument("--compile_only", action="store_true", help="仅预编译，不测试")
    parser.add_argument("--test_only", action="store_true", help="仅测试现有预编译")
    
    args = parser.parse_args()
    
    if args.test_only:
        # 仅测试现有预编译
        if os.path.exists(args.output_dir):
            pipeline = load_precompiled_pipeline(args.output_dir, args.precompile_depth)
            test_precompiled_pipeline(pipeline)
        else:
            print(f"预编译目录不存在: {args.output_dir}")
        return
    
    # 创建预编译pipeline
    pipeline = create_and_precompile_pipeline(
        model_path=args.model_path,
        device=args.device,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        frequency=args.frequency,
        precompile_depth=args.precompile_depth
    )
    
    # 保存配置
    save_precompiled_pipeline(pipeline, args.output_dir)
    
    if not args.compile_only:
        # 测试预编译pipeline
        test_precompiled_pipeline(pipeline)
    
    print(f"\n🎉 预编译完成!")
    print(f"\n使用方法:")
    print(f"```python")
    print(f"from export_run_policy_aot import load_precompiled_pipeline")
    print(f"")
    print(f"# 加载预编译pipeline（最小JIT开销）")
    print(f"planner = load_precompiled_pipeline('{args.output_dir}')")
    print(f"")
    print(f"# 预测控制（接近静态性能）")
    print(f"controls, timing = planner.predict_controls(qpos, qvel, return_timing=True)")
    print(f"```")


if __name__ == "__main__":
    main() 