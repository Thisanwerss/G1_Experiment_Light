#!/usr/bin/env python3
"""
导出run_policy.py的完整hybrid pipeline为可直接调用的编译函数

这个脚本将原始的JAX+PyTorch hybrid pipeline封装成：
1. 预编译的JAX函数（JIT优化）
2. 封装好的PyTorch网络
3. 统一的调用接口
4. 零启动开销的函数调用

运行方法:
    python export_run_policy.py --model_path nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt
    
使用方法:
    from exported_run_policy import load_compiled_pipeline
    planner = load_compiled_pipeline("exported_models/run_policy_compiled.pkl")
    controls = planner.predict_controls(qpos, qvel)
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


class CompiledRunPolicyPipeline:
    """编译后的run_policy hybrid pipeline
    
    这个类封装了完整的hybrid模式流程：
    1. PyTorch神经网络预测knots
    2. JAX CEM优化
    3. JAX插值生成控制序列
    
    所有JAX函数都预先JIT编译，实现零启动开销。
    """
    
    def __init__(
        self,
        pytorch_network: nn.Module,
        jax_task: Any,
        jax_controller: Any,
        device: str = 'cuda',
        # CEM参数（与run_policy.py保持一致）
        num_samples: int = 500,
        num_elites: int = 20,
        sigma_start: float = 0.3,
        sigma_min: float = 0.05,
        plan_horizon: float = 0.5,
        num_knots: int = 4,
        iterations: int = 1,
        frequency: float = 50.0
    ):
        """初始化编译后的pipeline
        
        Args:
            pytorch_network: 训练好的PyTorch网络
            jax_task: JAX任务定义
            jax_controller: JAX CEM控制器
            device: PyTorch设备
            其他参数: CEM控制参数
        """
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
        
        print(f"编译参数:")
        print(f"  - 规划频率: {frequency} Hz")
        print(f"  - 每次规划仿真步数: {self.sim_steps_per_replan}")
        print(f"  - CEM样本数: {num_samples}")
        print(f"  - 设备: {self.device}")
        
        # 预编译JAX函数
        print("开始预编译JAX函数...")
        self._compile_jax_functions()
        print("JAX函数编译完成!")
    
    def _compile_jax_functions(self):
        """预编译所有JAX函数，消除JIT开销"""
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
        dummy_knots = np.zeros((4, 41))  # 4个knots，41维控制
        policy_params = self.jax_controller.init_params(initial_knots=dummy_knots)
        
        # 编译optimize函数
        print("  编译CEM优化函数...")
        self.jit_optimize = jax.jit(self.jax_controller.optimize)
        
        # 预热编译（执行一次以触发JIT）
        start_time = time.time()
        _, _ = self.jit_optimize(mjx_data, policy_params)
        _, _ = self.jit_optimize(mjx_data, policy_params)  # 第二次确保完全编译
        compile_time = time.time() - start_time
        print(f"    CEM优化编译耗时: {compile_time:.3f}s")
        
        # 编译插值函数
        print("  编译插值函数...")
        self.jit_interp_func = jax.jit(self.jax_controller.interp_func)
        
        # 预热插值函数
        tq = jnp.arange(0, self.sim_steps_per_replan) * mj_model.opt.timestep
        tk = policy_params.tk
        knots = policy_params.mean[None, ...]
        _ = self.jit_interp_func(tq, tk, knots)
        _ = self.jit_interp_func(tq, tk, knots)  # 第二次确保完全编译
        
        # 保存编译后的参数模板
        self.policy_params_template = policy_params
        self.mjx_data_template = mjx_data
        
        print("  所有JAX函数编译完成!")
    
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
        """完整的hybrid控制预测（原汁原味的run_policy.py逻辑）
        
        Args:
            qpos: 关节位置 (nq,)
            qvel: 关节速度 (nv,)
            mocap_pos: mocap位置 (optional)
            mocap_quat: mocap四元数 (optional)
            current_time: 当前时间
            return_timing: 是否返回详细时间统计
            
        Returns:
            controls: (sim_steps_per_replan, nu) 控制序列
            timing_info: 时间统计信息（可选）
        """
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
        
        # 更新策略参数（使用NN预测的knots）
        policy_params = self.policy_params_template.replace(mean=predicted_knots)
        
        if return_timing:
            timing_info['prep_time'] = time.time() - prep_start
        
        # 3. JAX CEM优化
        if return_timing:
            cem_start = time.time()
        
        policy_params, rollouts = self.jit_optimize(mjx_data, policy_params)
        
        if return_timing:
            timing_info['cem_time'] = time.time() - cem_start
        
        # 4. JAX插值生成控制序列
        if return_timing:
            interp_start = time.time()
        
        # 查询时间序列
        tq = jnp.arange(0, self.sim_steps_per_replan) * self.jax_task.mj_model.opt.timestep + current_time
        tk = policy_params.tk
        knots = policy_params.mean[None, ...]
        controls_jax = self.jit_interp_func(tq, tk, knots)[0]  # (sim_steps_per_replan, nu)
        
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


def create_compiled_pipeline(
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
) -> CompiledRunPolicyPipeline:
    """创建编译后的pipeline
    
    Args:
        model_path: PyTorch模型路径
        device: 计算设备
        其他参数: CEM控制参数
        
    Returns:
        编译后的pipeline对象
    """
    print(f"创建编译后的pipeline...")
    print(f"模型路径: {model_path}")
    
    # 1. 加载PyTorch网络
    print("加载PyTorch网络...")
    device_obj = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device_obj)
    
    pl_net = MLPRegressor(95, 512, 512, 512, 164)
    pl_net.load_state_dict(checkpoint['state_dict'])
    net = pl_net.model  # 提取纯PyTorch模型
    net.to(device_obj).eval()
    print("PyTorch网络加载完成")
    
    # 2. 设置MuJoCo任务和模型
    print("设置MuJoCo任务...")
    task = HumanoidStand()
    mj_model = task.mj_model
    
    # 配置MuJoCo参数（与run_policy.py保持一致）
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
    
    # 4. 创建编译后的pipeline
    print("编译pipeline...")
    pipeline = CompiledRunPolicyPipeline(
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
    
    print("Pipeline编译完成!")
    return pipeline


def save_compiled_pipeline(pipeline: CompiledRunPolicyPipeline, output_path: str):
    """保存编译后的pipeline（保存创建参数而非编译对象）"""
    print(f"保存编译后的pipeline到: {output_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存PyTorch网络的state_dict
    pytorch_state = pipeline.pytorch_network.state_dict()
    
    # 保存配置信息（用于重建pipeline）
    config = {
        'pytorch_state_dict': pytorch_state,
        'device': str(pipeline.device),
        'frequency': pipeline.frequency,
        'replan_period': pipeline.replan_period,
        'plan_horizon': pipeline.plan_horizon,
        'sim_steps_per_replan': pipeline.sim_steps_per_replan,
        'step_dt': pipeline.step_dt,
        # CEM参数（从控制器中提取）
        'num_samples': pipeline.jax_controller.num_samples,
        'num_elites': pipeline.jax_controller.num_elites,
        'sigma_start': pipeline.jax_controller.sigma_start,
        'sigma_min': pipeline.jax_controller.sigma_min,
        'num_knots': pipeline.jax_controller.num_knots,
        'iterations': pipeline.jax_controller.iterations,
    }
    
    # 保存配置到pickle文件
    with open(output_path, 'wb') as f:
        pickle.dump(config, f)
    
    print(f"Pipeline配置已保存: {output_path}")


def load_compiled_pipeline(pipeline_path: str) -> CompiledRunPolicyPipeline:
    """加载编译后的pipeline（从配置重建）"""
    print(f"加载编译后的pipeline: {pipeline_path}")
    
    # 加载配置
    with open(pipeline_path, 'rb') as f:
        config = pickle.load(f)
    
    print("重建pipeline...")
    
    # 1. 重建PyTorch网络
    device_obj = torch.device(config['device'])
    pl_net = MLPRegressor(95, 512, 512, 512, 164)
    
    # 创建纯PyTorch模型并加载权重
    net = pl_net.model
    net.load_state_dict(config['pytorch_state_dict'])
    net.to(device_obj).eval()
    
    # 2. 重建任务和控制器
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
    
    # 3. 重建pipeline（这会重新编译JAX函数）
    pipeline = CompiledRunPolicyPipeline(
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
    
    print("Pipeline重建完成!")
    return pipeline


def test_compiled_pipeline(pipeline: CompiledRunPolicyPipeline, num_tests: int = 10):
    """测试编译后的pipeline性能"""
    print(f"\n测试编译后的pipeline性能 (运行{num_tests}次)...")
    
    # 获取仿真参数
    sim_params = pipeline.get_simulation_params()
    print("仿真参数:")
    for key, value in sim_params.items():
        print(f"  {key}: {value}")
    
    # 创建测试数据
    mj_model = pipeline.jax_task.mj_model
    nq, nv, nu = mj_model.nq, mj_model.nv, mj_model.nu
    
    print(f"\n模型维度: nq={nq}, nv={nv}, nu={nu}")
    
    # 性能测试
    times = []
    
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
    print(f"\n性能统计 ({num_tests}次测试):")
    print(f"  平均耗时: {np.mean(times):.4f}s")
    print(f"  最小耗时: {np.min(times):.4f}s")
    print(f"  最大耗时: {np.max(times):.4f}s")
    print(f"  标准差: {np.std(times):.4f}s")
    print(f"  理论最大频率: {1.0/np.mean(times):.2f} Hz")
    print(f"  目标频率: {sim_params['frequency']:.2f} Hz")
    print(f"  频率达成率: {min(1.0, sim_params['frequency'] / (1.0/np.mean(times))) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="导出run_policy.py完整pipeline为编译函数")
    
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
    parser.add_argument("--sigma_start", type=float, default=0.3, help="初始标准差")
    parser.add_argument("--sigma_min", type=float, default=0.05, help="最小标准差")
    parser.add_argument("--plan_horizon", type=float, default=0.5, help="规划时间范围")
    parser.add_argument("--num_knots", type=int, default=4, help="spline节点数")
    parser.add_argument("--iterations", type=int, default=1, help="CEM迭代次数")
    parser.add_argument("--frequency", type=float, default=50.0, help="规划频率")
    
    parser.add_argument("--test", action="store_true", help="测试编译后的pipeline")
    parser.add_argument("--test_only", action="store_true", help="仅测试现有pipeline")
    
    args = parser.parse_args()
    
    output_path = os.path.join(args.output_dir, "run_policy_compiled.pkl")
    
    if args.test_only:
        # 仅测试现有pipeline
        if os.path.exists(output_path):
            pipeline = load_compiled_pipeline(output_path)
            test_compiled_pipeline(pipeline)
        else:
            print(f"Pipeline文件不存在: {output_path}")
        return
    
    # 创建并保存pipeline
    pipeline = create_compiled_pipeline(
        model_path=args.model_path,
        device=args.device,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        sigma_start=args.sigma_start,
        sigma_min=args.sigma_min,
        plan_horizon=args.plan_horizon,
        num_knots=args.num_knots,
        iterations=args.iterations,
        frequency=args.frequency
    )
    
    save_compiled_pipeline(pipeline, output_path)
    
    # 测试pipeline
    if args.test:
        test_compiled_pipeline(pipeline)
    
    print(f"\n🎉 导出完成!")
    print(f"\n使用方法:")
    print(f"```python")
    print(f"from export_run_policy import load_compiled_pipeline")
    print(f"")
    print(f"# 加载编译后的pipeline")
    print(f"planner = load_compiled_pipeline('{output_path}')")
    print(f"")
    print(f"# 预测控制序列")
    print(f"controls, timing_info = planner.predict_controls(")
    print(f"    qpos, qvel, current_time=current_time, return_timing=True")
    print(f")")
    print(f"")
    print(f"# 获取仿真参数")
    print(f"sim_params = planner.get_simulation_params()")
    print(f"```")


if __name__ == "__main__":
    main() 