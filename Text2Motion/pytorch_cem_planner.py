#!/usr/bin/env python3
"""
完整的PyTorch CEM Planner实现，用于替代JAX版本

这个模块提供了一个完整的CEM规划器，可以：
1. 使用神经网络预测初始控制序列
2. 执行在线CEM优化
3. 支持TorchScript编译以获得更高性能
4. 完全消除JAX依赖

使用方法 (DOOM环境中):
    from pytorch_cem_planner import CEMPlanner
    
    planner = CEMPlanner(model_path="exported_models/network.ts")
    controls = planner.plan(state, mj_model, mj_data)
"""

import torch
import torch.nn as nn
import numpy as np
import mujoco
from typing import Tuple, Optional, Union
import time


class MLPRegressor(nn.Module):
    """神经网络模型定义，与原始模型保持一致"""
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim=1):
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


class CEMPlanner:
    """完整的CEM规划器，支持神经网络预测和在线优化"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        network: Optional[nn.Module] = None,
        num_samples: int = 500,
        num_elites: int = 20,
        sigma_start: float = 0.3,
        sigma_min: float = 0.05,
        plan_horizon: float = 0.5,
        num_knots: int = 4,
        iterations: int = 1,
        device: str = 'auto',
        use_network_warmstart: bool = True,
        cost_weights: Optional[dict] = None
    ):
        """初始化CEM规划器
        
        Args:
            model_path: 神经网络模型路径（TorchScript或普通PyTorch）
            network: 预加载的网络模型
            num_samples: CEM采样数量
            num_elites: 精英样本数量
            sigma_start: 初始标准差
            sigma_min: 最小标准差
            plan_horizon: 规划时间范围（秒）
            num_knots: 控制样条节点数量
            iterations: CEM优化迭代次数
            device: 计算设备
            use_network_warmstart: 是否使用网络预测作为CEM初始化
            cost_weights: 成本函数权重
        """
        
        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # CEM参数
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.sigma_start = sigma_start
        self.sigma_min = sigma_min
        self.plan_horizon = plan_horizon
        self.num_knots = num_knots
        self.iterations = iterations
        self.use_network_warmstart = use_network_warmstart
        
        # 成本函数权重
        self.cost_weights = cost_weights or {
            'height': 10.0,
            'angular_vel': 0.1,
            'control': 0.01,
            'stability': 1.0
        }
        
        # 加载神经网络
        self.network = None
        if model_path is not None:
            self.load_network(model_path)
        elif network is not None:
            self.network = network.to(self.device).eval()
        
        print(f"CEM规划器初始化完成，设备: {self.device}")
    
    def load_network(self, model_path: str):
        """加载神经网络模型"""
        try:
            # 尝试加载TorchScript模型
            self.network = torch.jit.load(model_path, map_location=self.device)
            print(f"加载TorchScript模型: {model_path}")
        except:
            try:
                # 尝试加载普通PyTorch模型
                checkpoint = torch.load(model_path, map_location=self.device)
                self.network = MLPRegressor(95, 512, 512, 512, 164)
                self.network.load_state_dict(checkpoint['state_dict'])
                self.network.to(self.device).eval()
                print(f"加载PyTorch Lightning checkpoint: {model_path}")
            except Exception as e:
                print(f"无法加载模型 {model_path}: {e}")
                self.network = None
    
    def predict_initial_knots(self, state: torch.Tensor) -> torch.Tensor:
        """使用神经网络预测初始控制节点
        
        Args:
            state: (95,) 机器人状态 [qpos, qvel]
            
        Returns:
            knots: (num_knots, nu) 预测的控制节点
        """
        if self.network is None:
            # 如果没有网络，返回零初始化
            return torch.zeros(self.num_knots, 41, device=self.device)
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)  # 添加batch维度
            
            knots_flat = self.network(state)  # (1, 164)
            knots = knots_flat.view(self.num_knots, -1)  # (4, 41)
            
        return knots
    
    def sample_control_sequences(
        self, 
        mean: torch.Tensor, 
        std: torch.Tensor
    ) -> torch.Tensor:
        """采样控制序列
        
        Args:
            mean: (num_knots, nu) 均值
            std: (num_knots, nu) 标准差
            
        Returns:
            samples: (num_samples, num_knots, nu) 采样的控制序列
        """
        noise = torch.randn(
            self.num_samples, *mean.shape, 
            device=self.device, dtype=mean.dtype
        )
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        return samples
    
    def evaluate_rollouts(
        self,
        control_samples: torch.Tensor,
        mj_model,
        initial_qpos: np.ndarray,
        initial_qvel: np.ndarray,
        dt: float = 0.01
    ) -> torch.Tensor:
        """评估控制样本的成本
        
        Args:
            control_samples: (num_samples, num_knots, nu) 控制样本
            mj_model: MuJoCo模型
            initial_qpos: 初始位置
            initial_qvel: 初始速度
            dt: 仿真时间步长
            
        Returns:
            costs: (num_samples,) 每个样本的总成本
        """
        costs = torch.zeros(self.num_samples, device=self.device)
        num_steps = int(self.plan_horizon / dt)
        
        # 批量处理以提高效率
        batch_size = min(50, self.num_samples)  # 避免内存溢出
        
        for batch_start in range(0, self.num_samples, batch_size):
            batch_end = min(batch_start + batch_size, self.num_samples)
            batch_costs = []
            
            for i in range(batch_start, batch_end):
                cost = self._evaluate_single_rollout(
                    control_samples[i], 
                    mj_model, 
                    initial_qpos, 
                    initial_qvel, 
                    dt, 
                    num_steps
                )
                batch_costs.append(cost)
            
            costs[batch_start:batch_end] = torch.tensor(
                batch_costs, device=self.device
            )
        
        return costs
    
    def _evaluate_single_rollout(
        self,
        knots: torch.Tensor,
        mj_model,
        initial_qpos: np.ndarray,
        initial_qvel: np.ndarray,
        dt: float,
        num_steps: int
    ) -> float:
        """评估单个控制序列的成本"""
        # 创建独立的MuJoCo data
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = initial_qpos.copy()
        mj_data.qvel[:] = initial_qvel.copy()
        mujoco.mj_forward(mj_model, mj_data)
        
        total_cost = 0.0
        knots_np = knots.cpu().numpy()
        
        for step in range(num_steps):
            t = step * dt
            
            # 零阶插值
            knot_idx = min(
                int(t / self.plan_horizon * (self.num_knots - 1)), 
                self.num_knots - 1
            )
            control = knots_np[knot_idx]
            
            # 应用控制并仿真
            mj_data.ctrl[:] = control
            mujoco.mj_step(mj_model, mj_data)
            
            # 计算当前步骤的成本
            step_cost = self._compute_step_cost(mj_data, control)
            total_cost += step_cost
        
        return total_cost
    
    def _compute_step_cost(self, mj_data, control: np.ndarray) -> float:
        """计算单步成本"""
        # 基本成本组件
        com_height = mj_data.qpos[2]  # 质心高度
        angular_vel = np.linalg.norm(mj_data.qvel[3:6])  # 角速度
        control_norm = np.sum(control**2)  # 控制成本
        
        # 姿态稳定性（基于四元数）
        quat = mj_data.qpos[3:7]
        quat_w = quat[0]
        orientation_error = 1.0 - quat_w**2  # 偏离直立的角度
        
        # 总成本
        cost = (
            self.cost_weights['height'] * max(0, 1.0 - com_height)**2 +
            self.cost_weights['angular_vel'] * angular_vel**2 +
            self.cost_weights['control'] * control_norm +
            self.cost_weights['stability'] * orientation_error
        )
        
        return cost
    
    def update_distribution(
        self,
        samples: torch.Tensor,
        costs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新CEM分布参数
        
        Args:
            samples: (num_samples, num_knots, nu) 控制样本
            costs: (num_samples,) 成本
            
        Returns:
            new_mean: (num_knots, nu) 新均值
            new_std: (num_knots, nu) 新标准差
        """
        # 选择精英样本
        _, elite_indices = torch.topk(costs, self.num_elites, largest=False)
        elites = samples[elite_indices]
        
        # 计算新分布参数
        new_mean = torch.mean(elites, dim=0)
        new_std = torch.clamp(
            torch.std(elites, dim=0), 
            min=self.sigma_min
        )
        
        return new_mean, new_std
    
    def zero_order_interpolation(
        self,
        knots: torch.Tensor,
        query_times: torch.Tensor
    ) -> torch.Tensor:
        """零阶插值生成控制序列
        
        Args:
            knots: (num_knots, nu) 控制节点
            query_times: (T,) 查询时间点
            
        Returns:
            controls: (T, nu) 插值后的控制序列
        """
        # 归一化时间到[0, 1]
        normalized_times = query_times / self.plan_horizon
        
        # 计算对应的节点索引
        knot_indices = torch.clamp(
            (normalized_times * (self.num_knots - 1)).long(),
            0, self.num_knots - 1
        )
        
        return knots[knot_indices]
    
    def plan(
        self,
        state: Union[torch.Tensor, np.ndarray],
        mj_model,
        mj_data,
        warm_start_knots: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """执行完整的CEM规划
        
        Args:
            state: 当前机器人状态
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
            warm_start_knots: 可选的初始控制节点
            
        Returns:
            controls: 优化后的控制序列
            info: 规划信息字典
        """
        start_time = time.time()
        timing_info = {}
        
        # 状态预处理
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        
        # 神经网络推理阶段
        nn_start = time.time()
        if warm_start_knots is not None:
            mean = warm_start_knots.to(self.device)
        elif self.use_network_warmstart and self.network is not None:
            mean = self.predict_initial_knots(state)
        else:
            # 零初始化
            mean = torch.zeros(self.num_knots, mj_model.nu, device=self.device)
        timing_info['nn_time'] = time.time() - nn_start
        
        std = torch.full_like(mean, self.sigma_start)
        
        # 获取当前状态
        current_qpos = mj_data.qpos.copy()
        current_qvel = mj_data.qvel.copy()
        
        # CEM优化阶段
        cem_start = time.time()
        best_cost = float('inf')
        best_knots = mean.clone()
        
        for iteration in range(self.iterations):
            # 采样控制序列
            sampling_start = time.time()
            samples = self.sample_control_sequences(mean, std)
            sampling_time = time.time() - sampling_start
            
            # 评估样本
            evaluation_start = time.time()
            costs = self.evaluate_rollouts(
                samples, mj_model, current_qpos, current_qvel
            )
            evaluation_time = time.time() - evaluation_start
            
            # 更新最佳解
            update_start = time.time()
            min_cost_idx = torch.argmin(costs)
            if costs[min_cost_idx] < best_cost:
                best_cost = costs[min_cost_idx].item()
                best_knots = samples[min_cost_idx].clone()
            
            # 更新分布（如果还有剩余迭代）
            if iteration < self.iterations - 1:
                mean, std = self.update_distribution(samples, costs)
            update_time = time.time() - update_start
            
            # 记录第一次迭代的详细时间
            if iteration == 0:
                timing_info.update({
                    'sampling_time': sampling_time,
                    'evaluation_time': evaluation_time,
                    'update_time': update_time
                })
        
        timing_info['cem_time'] = time.time() - cem_start
        
        # 插值阶段
        interp_start = time.time()
        plan_steps = int(self.plan_horizon / mj_model.opt.timestep)
        query_times = torch.linspace(
            0, self.plan_horizon, plan_steps, device=self.device
        )
        controls = self.zero_order_interpolation(best_knots, query_times)
        timing_info['interpolation_time'] = time.time() - interp_start
        
        # 总规划信息
        total_time = time.time() - start_time
        info = {
            'planning_time': total_time,
            'best_cost': best_cost,
            'iterations': self.iterations,
            'num_samples': self.num_samples,
            'num_elites': self.num_elites,
            'best_knots': best_knots.cpu().numpy(),
            **timing_info  # 添加详细的耗时信息
        }
        
        return controls, info
    
    def get_action(
        self,
        knots: torch.Tensor,
        current_time: float
    ) -> torch.Tensor:
        """从控制节点获取当前时刻的动作
        
        Args:
            knots: (num_knots, nu) 控制节点
            current_time: 当前时间
            
        Returns:
            action: (nu,) 当前动作
        """
        t_normalized = min(current_time / self.plan_horizon, 1.0)
        knot_idx = min(
            int(t_normalized * (self.num_knots - 1)), 
            self.num_knots - 1
        )
        return knots[knot_idx]


class FastCEMPlanner(CEMPlanner):
    """高性能版本的CEM规划器，针对速度优化
    
    主要优化：
    1. 复用MjData对象，避免重复创建
    2. 批量处理以减少Python循环开销
    3. 优化GPU-CPU数据传输
    4. 使用预分配的内存池
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 预分配MjData池，避免运行时创建
        self._mj_data_pool = []
        self._pool_size = min(50, self.num_samples)  # 限制内存使用
        
    def _initialize_data_pool(self, mj_model):
        """初始化MjData对象池"""
        if not self._mj_data_pool:
            print(f"初始化MjData池，大小: {self._pool_size}")
            for _ in range(self._pool_size):
                self._mj_data_pool.append(mujoco.MjData(mj_model))
    
    def evaluate_samples_fast(
        self,
        control_samples: torch.Tensor,
        mj_model,
        initial_qpos: np.ndarray,
        initial_qvel: np.ndarray
    ) -> torch.Tensor:
        """快速评估控制样本（优化版本）"""
        self._initialize_data_pool(mj_model)
        
        costs = torch.zeros(self.num_samples, device=self.device)
        dt = mj_model.opt.timestep
        num_steps = int(self.plan_horizon / dt)
        
        # 批量处理以最大化MjData复用
        batch_size = self._pool_size
        
        for batch_start in range(0, self.num_samples, batch_size):
            batch_end = min(batch_start + batch_size, self.num_samples)
            batch_costs = []
            
            # 并行处理批次内的样本
            for i, sample_idx in enumerate(range(batch_start, batch_end)):
                mj_data = self._mj_data_pool[i]  # 复用已创建的MjData
                
                cost = self._evaluate_single_rollout_fast(
                    control_samples[sample_idx], 
                    mj_model, 
                    mj_data,  # 传入预创建的MjData
                    initial_qpos, 
                    initial_qvel, 
                    dt, 
                    num_steps
                )
                batch_costs.append(cost)
            
            costs[batch_start:batch_end] = torch.tensor(
                batch_costs, device=self.device
            )
        
        return costs
    
    def _evaluate_single_rollout_fast(
        self,
        knots: torch.Tensor,
        mj_model,
        mj_data,  # 预创建的MjData，避免重复创建
        initial_qpos: np.ndarray,
        initial_qvel: np.ndarray,
        dt: float,
        num_steps: int
    ) -> float:
        """快速评估单个控制序列（复用MjData版本）"""
        # 重置状态（比创建新对象快得多）
        mj_data.qpos[:] = initial_qpos
        mj_data.qvel[:] = initial_qvel
        mujoco.mj_forward(mj_model, mj_data)
        
        total_cost = 0.0
        
        # 一次性转换到numpy，避免循环中的GPU-CPU传输
        knots_np = knots.cpu().numpy()
        
        for step in range(num_steps):
            t = step * dt
            
            # 零阶插值
            knot_idx = min(
                int(t / self.plan_horizon * (self.num_knots - 1)),
                self.num_knots - 1
            )
            
            # 应用控制
            mj_data.ctrl[:] = knots_np[knot_idx]
            
            # 仿真步进
            mujoco.mj_step(mj_model, mj_data)
            
            # 计算成本（简化版本，针对人形机器人站立任务）
            # 高度成本
            pelvis_height = mj_data.qpos[2] if len(mj_data.qpos) > 2 else 0.0
            height_cost = (pelvis_height - 1.0) ** 2
            
            # 速度惩罚
            vel_cost = 0.001 * np.sum(mj_data.qvel ** 2)
            
            # 控制惩罚
            ctrl_cost = 0.001 * np.sum(mj_data.ctrl ** 2)
            
            total_cost += dt * (height_cost + vel_cost + ctrl_cost)
        
        return total_cost
    
    def plan(
        self,
        state: Union[torch.Tensor, np.ndarray],
        mj_model,
        mj_data,
        warm_start_knots: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """执行完整的CEM规划（快速版本）"""
        start_time = time.time()
        timing_info = {}
        
        # 状态预处理
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device).float()
        
        # 初始化
        initial_qpos = mj_data.qpos.copy()
        initial_qvel = mj_data.qvel.copy()
        
        # 神经网络预测（如果启用）
        if self.use_network_warmstart:
            nn_start = time.time()
            with torch.no_grad():
                predicted_knots = self.network(state.unsqueeze(0)).view(self.num_knots, -1)
            timing_info['nn_time'] = time.time() - nn_start
            
            # 使用预测结果初始化
            mean = predicted_knots
            std = torch.full_like(mean, self.sigma_start)
        else:
            # 随机初始化
            mean = torch.zeros(self.num_knots, mj_model.nu, device=self.device)
            std = torch.full_like(mean, self.sigma_start)
        
        best_cost = float('inf')
        best_knots = mean.clone()
        
        # CEM迭代优化
        cem_start = time.time()
        for iteration in range(self.iterations):
            # 采样阶段
            sampling_start = time.time()
            noise = torch.randn(self.num_samples, self.num_knots, mj_model.nu, device=self.device)
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            sampling_time = time.time() - sampling_start
            
            # 评估阶段（使用快速版本）
            evaluation_start = time.time()
            costs = self.evaluate_samples_fast(
                samples, mj_model, initial_qpos, initial_qvel
            )
            evaluation_time = time.time() - evaluation_start
            
            # 更新最佳解
            update_start = time.time()
            min_cost_idx = torch.argmin(costs)
            if costs[min_cost_idx] < best_cost:
                best_cost = costs[min_cost_idx].item()
                best_knots = samples[min_cost_idx].clone()
            
            # 更新分布（如果还有剩余迭代）
            if iteration < self.iterations - 1:
                mean, std = self.update_distribution(samples, costs)
            update_time = time.time() - update_start
            
            # 记录第一次迭代的详细时间
            if iteration == 0:
                timing_info.update({
                    'sampling_time': sampling_time,
                    'evaluation_time': evaluation_time,
                    'update_time': update_time
                })
        
        timing_info['cem_time'] = time.time() - cem_start
        
        # 插值阶段
        interp_start = time.time()
        plan_steps = int(self.plan_horizon / mj_model.opt.timestep)
        query_times = torch.linspace(
            0, self.plan_horizon, plan_steps, device=self.device
        )
        controls = self.zero_order_interpolation(best_knots, query_times)
        timing_info['interpolation_time'] = time.time() - interp_start
        
        # 总规划信息
        total_time = time.time() - start_time
        info = {
            'planning_time': total_time,
            'best_cost': best_cost,
            'iterations': self.iterations,
            'num_samples': self.num_samples,
            'num_elites': self.num_elites,
            'best_knots': best_knots.cpu().numpy(),
            **timing_info  # 添加详细的耗时信息
        }
        
        return controls, info


class TorchScriptCEMPlanner(nn.Module):
    """TorchScript兼容的CEM规划器，用于高性能推理"""
    
    def __init__(
        self,
        network: nn.Module,
        num_knots: int = 4,
        plan_horizon: float = 0.5,
        num_ctrl_steps: int = 50
    ):
        super().__init__()
        self.network = network
        self.num_knots = num_knots
        self.plan_horizon = plan_horizon
        
        # 注册缓冲区
        self.register_buffer(
            'query_times',
            torch.linspace(0, plan_horizon, num_ctrl_steps)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """纯前向推理（无CEM优化）
        
        Args:
            state: (95,) 机器人状态
            
        Returns:
            controls: (num_ctrl_steps, nu) 控制序列
        """
        # 网络预测
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        knots_flat = self.network(state)  # (1, 164)
        knots = knots_flat.view(self.num_knots, -1)  # (4, 41)
        
        # 零阶插值
        controls = self._zero_order_interpolation(knots, self.query_times)
        
        return controls
    
    def _zero_order_interpolation(
        self,
        knots: torch.Tensor,
        query_times: torch.Tensor
    ) -> torch.Tensor:
        """零阶插值（TorchScript兼容）"""
        normalized_times = query_times / self.plan_horizon
        knot_indices = torch.clamp(
            (normalized_times * (self.num_knots - 1)).long(),
            0, self.num_knots - 1
        )
        return knots[knot_indices]


def create_torchscript_planner(
    network_path: str,
    output_path: str = "torchscript_cem_planner.ts"
) -> str:
    """创建TorchScript版本的规划器
    
    Args:
        network_path: 神经网络模型路径
        output_path: 输出TorchScript文件路径
        
    Returns:
        生成的TorchScript文件路径
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载网络
    if network_path.endswith('.ts'):
        network = torch.jit.load(network_path, map_location=device)
    else:
        checkpoint = torch.load(network_path, map_location=device)
        network = MLPRegressor(95, 512, 512, 512, 164)
        network.load_state_dict(checkpoint['state_dict'])
        network.to(device).eval()
    
    # 创建规划器
    planner = TorchScriptCEMPlanner(network)
    planner.eval()
    
    # 编译为TorchScript
    dummy_state = torch.randn(95, device=device)
    traced_planner = torch.jit.trace(planner, dummy_state)
    
    # 保存
    traced_planner.save(output_path)
    print(f"TorchScript规划器已保存到: {output_path}")
    
    return output_path


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")
    
    # 创建模拟状态
    test_state = torch.randn(95, device=device)
    
    # 测试基本规划器
    planner = CEMPlanner(
        model_path=None,  # 不使用预训练模型
        num_samples=100,
        iterations=1,
        device=device
    )
    
    print("CEMPlanner初始化成功")
    print(f"设备: {planner.device}")
    print(f"参数: samples={planner.num_samples}, elites={planner.num_elites}") 