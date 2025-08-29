#!/usr/bin/env python3
"""
优化的run_policy pipeline，通过预热和缓存最小化JIT开销

这个版本通过以下方式减少JIT开销：
1. 充分的预热编译（多次调用不同输入）
2. 使用JAX的编译缓存
3. 优化的数据流管道
4. 最小化重新编译的触发

运行方法:
    python export_run_policy_optimized.py --model_path nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt
"""

import argparse
import os
import pickle
import time
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


class OptimizedRunPolicyPipeline:
    """优化的run_policy pipeline
    
    通过充分预热编译来最小化运行时JIT开销：
    1. 深度预热：使用多种不同输入模式
    2. 缓存友好：避免触发重新编译的操作
    3. 数据复用：预分配和复用数据结构
    4. 批量优化：减少函数调用开销
    """
    
    def __init__(
        self,
        pytorch_network: nn.Module,
        jax_task: Any,
        jax_controller: Any,
        device: str = 'cuda',
        # CEM参数
        num_samples: int = 500,
        num_elites: int = 20,
        sigma_start: float = 0.3,
        sigma_min: float = 0.05,
        plan_horizon: float = 0.5,
        num_knots: int = 4,
        iterations: int = 1,
        frequency: float = 50.0
    ):
        """初始化优化后的pipeline"""
        self.device = torch.device(device)
        self.pytorch_network = pytorch_network.to(self.device).eval()
        self.jax_task = jax_task
        self.jax_controller = jax_controller
        
        # 计算参数
        self.frequency = frequency
        self.replan_period = 1.0 / frequency
        self.plan_horizon = plan_horizon
        
        # 预计算仿真参数
        mj_model = jax_task.mj_model
        self.sim_steps_per_replan = max(int(self.replan_period / mj_model.opt.timestep), 1)
        self.step_dt = self.sim_steps_per_replan * mj_model.opt.timestep
        
        print(f"优化Pipeline初始化:")
        print(f"  - 规划频率: {frequency} Hz")
        print(f"  - 每次规划仿真步数: {self.sim_steps_per_replan}")
        print(f"  - CEM样本数: {num_samples}")
        print(f"  - 设备: {self.device}")
        
        # 深度预热编译
        print("开始深度预热编译...")
        self._deep_warmup_compilation()
        print("深度预热编译完成!")
    
    def _deep_warmup_compilation(self):
        """深度预热编译：使用多种输入模式充分编译所有代码路径"""
        # 创建虚拟的MuJoCo数据用于编译
        mj_model = self.jax_task.mj_model
        mj_data = mujoco.MjData(mj_model)
        
        # 转换为JAX格式
        mjx_data = mjx.put_data(mj_model, mj_data)
        mjx_data = mjx_data.replace(
            mocap_pos=mj_data.mocap_pos, 
            mocap_quat=mj_data.mocap_quat
        )
        
        # 初始化控制器参数
        dummy_knots = np.zeros((4, 41))
        policy_params = self.jax_controller.init_params(initial_knots=dummy_knots)
        
        # 编译optimize函数
        print("  深度编译CEM优化函数...")
        self.jit_optimize = jax.jit(self.jax_controller.optimize)
        
        # 深度预热：使用多种不同的状态模式
        compile_start = time.time()
        
        print("    阶段1: 基础编译...")
        for i in range(3):
            _, _ = self.jit_optimize(mjx_data, policy_params)
        
        print("    阶段2: 多样化状态编译...")
        # 使用不同的状态和参数模式来触发所有编译路径
        for i in range(5):
            # 随机化状态
            random_qpos = np.random.randn(mj_model.nq) * 0.1
            random_qvel = np.random.randn(mj_model.nv) * 0.1
            random_time = i * 0.1
            
            varied_mjx_data = mjx_data.replace(
                qpos=jnp.array(random_qpos),
                qvel=jnp.array(random_qvel),
                time=random_time
            )
            
            # 随机化knots
            random_knots = np.random.randn(4, 41) * 0.1
            varied_params = policy_params.replace(mean=random_knots)
            
            # 执行编译
            _, _ = self.jit_optimize(varied_mjx_data, varied_params)
        
        print("    阶段3: 极值案例编译...")
        # 测试极值情况
        extreme_cases = [
            (np.zeros((mj_model.nq,)), np.zeros((mj_model.nv,))),  # 全零状态
            (np.ones((mj_model.nq,)), np.ones((mj_model.nv,))),   # 全一状态
            (np.random.randn(mj_model.nq) * 5, np.random.randn(mj_model.nv) * 5),  # 大幅状态
        ]
        
        for qpos, qvel in extreme_cases:
            extreme_mjx_data = mjx_data.replace(
                qpos=jnp.array(qpos),
                qvel=jnp.array(qvel)
            )
            _, _ = self.jit_optimize(extreme_mjx_data, policy_params)
        
        compile_time = time.time() - compile_start
        print(f"    CEM优化深度编译耗时: {compile_time:.3f}s")
        
        # 编译插值函数
        print("  深度编译插值函数...")
        self.jit_interp_func = jax.jit(self.jax_controller.interp_func)
        
        # 深度预热插值函数
        tq_base = jnp.arange(0, self.sim_steps_per_replan) * mj_model.opt.timestep
        tk = policy_params.tk
        
        # 使用不同的时间偏移和knots模式
        for i in range(5):
            time_offset = i * 0.02
            tq = tq_base + time_offset
            
            # 使用不同的knots
            varied_knots = (policy_params.mean + np.random.randn(4, 41) * 0.1)[None, ...]
            _ = self.jit_interp_func(tq, tk, varied_knots)
        
        # 保存编译后的参数模板
        self.policy_params_template = policy_params
        self.mjx_data_template = mjx_data
        
        print("  所有JAX函数深度编译完成!")
    
    def predict_knots_pytorch(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """使用PyTorch网络预测knots（优化版本）"""
        with torch.no_grad():
            # 状态拼接和预处理（批量化）
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
        """优化的控制预测（最小JIT开销版本）"""
        total_start = time.time()
        timing_info = {} if return_timing else None
        
        # 1. PyTorch神经网络预测
        if return_timing:
            nn_start = time.time()
        
        predicted_knots = self.predict_knots_pytorch(qpos, qvel)
        
        if return_timing:
            timing_info['nn_time'] = time.time() - nn_start
        
        # 2. 准备JAX数据（优化：复用模板，只更新必要字段）
        if return_timing:
            prep_start = time.time()
        
        # 高效更新MJX数据
        mjx_data = self.mjx_data_template.replace(
            qpos=jnp.array(qpos),
            qvel=jnp.array(qvel),
            time=current_time
        )
        
        if mocap_pos is not None:
            mjx_data = mjx_data.replace(mocap_pos=jnp.array(mocap_pos))
        if mocap_quat is not None:
            mjx_data = mjx_data.replace(mocap_quat=jnp.array(mocap_quat))
        
        # 高效更新策略参数
        policy_params = self.policy_params_template.replace(mean=predicted_knots)
        
        if return_timing:
            timing_info['prep_time'] = time.time() - prep_start
        
        # 3. JAX CEM优化（已深度预热，应该无JIT开销）
        if return_timing:
            cem_start = time.time()
        
        policy_params, rollouts = self.jit_optimize(mjx_data, policy_params)
        
        if return_timing:
            timing_info['cem_time'] = time.time() - cem_start
        
        # 4. JAX插值生成控制序列（已深度预热）
        if return_timing:
            interp_start = time.time()
        
        # 查询时间序列
        tq = jnp.arange(0, self.sim_steps_per_replan) * self.jax_task.mj_model.opt.timestep + current_time
        tk = policy_params.tk
        knots = policy_params.mean[None, ...]
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
            'replan_period': self.replan_period,
            'sim_steps_per_replan': self.sim_steps_per_replan,
            'step_dt': self.step_dt,
            'plan_horizon': self.plan_horizon,
            'mujoco_timestep': self.jax_task.mj_model.opt.timestep
        }


def create_optimized_pipeline(
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
    frequency: float = 50.0
) -> OptimizedRunPolicyPipeline:
    """创建优化后的pipeline"""
    print(f"创建优化的pipeline...")
    print(f"模型路径: {model_path}")
    
    # 1. 加载PyTorch网络
    print("加载PyTorch网络...")
    device_obj = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device_obj)
    
    pl_net = MLPRegressor(95, 512, 512, 512, 164)
    pl_net.load_state_dict(checkpoint['state_dict'])
    net = pl_net.model
    net.to(device_obj).eval()
    print("PyTorch网络加载完成")
    
    # 2. 设置MuJoCo任务和模型
    print("设置MuJoCo任务...")
    task = HumanoidStand()
    mj_model = task.mj_model
    
    # 配置MuJoCo参数
    mj_model.opt.timestep = 0.01
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 50
    mj_model.opt.noslip_iterations = 2
    mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
    print("MuJoCo任务设置完成")
    
    # 3. 创建JAX CEM控制器
    print("创建JAX CEM控制器...")
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
    print("JAX CEM控制器创建完成")
    
    # 4. 创建优化后的pipeline
    print("编译优化pipeline...")
    pipeline = OptimizedRunPolicyPipeline(
        pytorch_network=net,
        jax_task=task,
        jax_controller=ctrl,
        device=device,
        num_samples=num_samples,
        num_elites=num_elites,
        sigma_start=sigma_start,
        sigma_min=sigma_min,
        plan_horizon=plan_horizon,
        num_knots=num_knots,
        iterations=iterations,
        frequency=frequency
    )
    
    print("优化Pipeline创建完成!")
    return pipeline


def save_optimized_pipeline(pipeline: OptimizedRunPolicyPipeline, output_path: str):
    """保存优化后的pipeline配置"""
    print(f"保存优化pipeline配置到: {output_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存配置信息
    config = {
        'pytorch_state_dict': pipeline.pytorch_network.state_dict(),
        'device': str(pipeline.device),
        'frequency': pipeline.frequency,
        'replan_period': pipeline.replan_period,
        'plan_horizon': pipeline.plan_horizon,
        'sim_steps_per_replan': pipeline.sim_steps_per_replan,
        'step_dt': pipeline.step_dt,
        'num_samples': pipeline.jax_controller.num_samples,
        'num_elites': pipeline.jax_controller.num_elites,
        'sigma_start': pipeline.jax_controller.sigma_start,
        'sigma_min': pipeline.jax_controller.sigma_min,
        'num_knots': pipeline.jax_controller.num_knots,
        'iterations': pipeline.jax_controller.iterations,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(config, f)
    
    print(f"优化Pipeline配置已保存: {output_path}")


def load_optimized_pipeline(pipeline_path: str) -> OptimizedRunPolicyPipeline:
    """加载优化后的pipeline（包含深度预热）"""
    print(f"加载优化pipeline: {pipeline_path}")
    
    with open(pipeline_path, 'rb') as f:
        config = pickle.load(f)
    
    print("重建并优化pipeline...")
    
    # 1. 首先重建PyTorch网络
    device_obj = torch.device(config['device'])
    pl_net = MLPRegressor(95, 512, 512, 512, 164)
    net = pl_net.model
    net.load_state_dict(config['pytorch_state_dict'])
    net.to(device_obj).eval()
    print("PyTorch网络重建完成")
    
    # 2. 设置MuJoCo任务和模型
    print("设置MuJoCo任务...")
    task = HumanoidStand()
    mj_model = task.mj_model
    
    # 配置MuJoCo参数
    mj_model.opt.timestep = 0.01
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 50
    mj_model.opt.noslip_iterations = 2
    mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
    print("MuJoCo任务设置完成")
    
    # 3. 创建JAX CEM控制器
    print("创建JAX CEM控制器...")
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
    print("JAX CEM控制器创建完成")
    
    # 4. 创建优化后的pipeline（直接构造，避免重复加载模型）
    print("编译优化pipeline...")
    pipeline = OptimizedRunPolicyPipeline(
        pytorch_network=net,
        jax_task=task,
        jax_controller=ctrl,
        device=config['device'],
        num_samples=config['num_samples'],
        num_elites=config['num_elites'],
        sigma_start=config['sigma_start'],
        sigma_min=config['sigma_min'],
        plan_horizon=config['plan_horizon'],
        num_knots=config['num_knots'],
        iterations=config['iterations'],
        frequency=config['frequency']
    )
    
    print("优化Pipeline重建完成!")
    return pipeline


def test_optimized_pipeline(pipeline: OptimizedRunPolicyPipeline, num_tests: int = 10):
    """测试优化后的pipeline性能"""
    print(f"\n🚀 测试优化pipeline性能 (运行{num_tests}次)...")
    
    sim_params = pipeline.get_simulation_params()
    print("仿真参数:")
    for key, value in sim_params.items():
        print(f"  {key}: {value}")
    
    nq, nv = 48, 47
    print(f"\n模型维度: nq={nq}, nv={nv}")
    
    # 性能测试
    times = []
    consistency_check = []
    
    for i in range(num_tests):
        qpos = np.random.randn(nq) * 0.1
        qvel = np.random.randn(nv) * 0.1
        
        start_time = time.time()
        controls, timing_info = pipeline.predict_controls(
            qpos, qvel, 
            current_time=i * sim_params['step_dt'],
            return_timing=True
        )
        total_time = time.time() - start_time
        times.append(total_time)
        
        if timing_info:
            consistency_check.append(timing_info['cem_time'])
        
        # 打印详细信息（前几次）
        if i < 3:
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
    consistency_check = np.array(consistency_check)
    
    print(f"\n📊 优化性能统计 ({num_tests}次测试):")
    print(f"  平均耗时: {np.mean(times):.4f}s")
    print(f"  最小耗时: {np.min(times):.4f}s")
    print(f"  最大耗时: {np.max(times):.4f}s")
    print(f"  标准差: {np.std(times):.4f}s")
    print(f"  理论最大频率: {1.0/np.mean(times):.2f} Hz")
    print(f"  目标频率: {sim_params['frequency']:.2f} Hz")
    
    # 一致性检查
    cem_std = np.std(consistency_check)
    print(f"\n📈 优化效果评估:")
    print(f"  CEM时间标准差: {cem_std:.4f}s")
    if cem_std < 0.005:
        print("  ✅ 优化成功：CEM性能非常一致")
    elif cem_std < 0.02:
        print("  ⚡ 优化良好：CEM性能基本一致")
    else:
        print("  ⚠️ 仍有优化空间：CEM性能存在波动")
    
    # 首次调用检查
    if len(times) > 1:
        first_vs_rest = times[0] / np.mean(times[1:])
        print(f"  首次调用倍数: {first_vs_rest:.2f}x")
        if first_vs_rest < 2:
            print("  ✅ 深度预热成功：首次调用无明显开销")
        else:
            print("  ⚠️ 仍存在首次调用开销")


def main():
    parser = argparse.ArgumentParser(description="优化的run_policy pipeline")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt",
        help="PyTorch模型checkpoint路径"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exported_models",
        help="输出目录"
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
    
    parser.add_argument("--test", action="store_true", help="测试优化后的pipeline")
    parser.add_argument("--test_only", action="store_true", help="仅测试现有pipeline")
    
    args = parser.parse_args()
    
    output_path = os.path.join(args.output_dir, "run_policy_optimized.pkl")
    
    if args.test_only:
        if os.path.exists(output_path):
            pipeline = load_optimized_pipeline(output_path)
            test_optimized_pipeline(pipeline)
        else:
            print(f"优化Pipeline文件不存在: {output_path}")
        return
    
    # 创建并保存优化pipeline
    pipeline = create_optimized_pipeline(
        model_path=args.model_path,
        device=args.device,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        frequency=args.frequency
    )
    
    save_optimized_pipeline(pipeline, output_path)
    
    # 测试pipeline
    if args.test:
        test_optimized_pipeline(pipeline)
    
    print(f"\n🎉 优化Pipeline创建完成!")
    print(f"\n关键优化:")
    print(f"  ✅ 深度预热编译：使用多种输入模式")
    print(f"  ✅ 数据结构复用：减少内存分配")
    print(f"  ✅ 批量操作优化：减少函数调用开销")
    print(f"  ✅ 缓存友好设计：避免重新编译触发")
    print(f"\n使用方法:")
    print(f"```python")
    print(f"from export_run_policy_optimized import load_optimized_pipeline")
    print(f"planner = load_optimized_pipeline('{output_path}')")
    print(f"controls, timing = planner.predict_controls(qpos, qvel, return_timing=True)")
    print(f"```")


if __name__ == "__main__":
    main() 