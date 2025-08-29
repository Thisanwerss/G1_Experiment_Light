#!/usr/bin/env python3
"""
导出CEM Planner到TorchScript/ONNX格式，消除JAX依赖

运行方法:
    python export_cem_planner.py --model_path nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import mujoco
import pytorch_lightning as pl
from typing import Tuple, Optional

# 定义模型结构（避免导入依赖问题）
# from scripts.run_policy import MLPRegressor  # 注释掉有问题的导入
# from hydrax.tasks.humanoid_standonly import HumanoidStand


class MLPRegressor(pl.LightningModule):
    """神经网络模型定义，与原始模型保持一致"""
    
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


class PyTorchCEM:
    """纯PyTorch实现的CEM优化器，用于替代JAX版本"""
    
    def __init__(
        self,
        num_samples: int = 500,
        num_elites: int = 20,
        sigma_start: float = 0.3,
        sigma_min: float = 0.05,
        plan_horizon: float = 0.5,
        num_knots: int = 4,
        iterations: int = 1,
        device: str = 'cuda'
    ):
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.sigma_start = sigma_start
        self.sigma_min = sigma_min
        self.plan_horizon = plan_horizon
        self.num_knots = num_knots
        self.iterations = iterations
        self.device = device
        
        # 预计算时间网格
        self.tk = torch.linspace(0, plan_horizon, num_knots, device=device)
        
    def sample_knots(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """采样控制knots
        
        Args:
            mean: (num_knots, nu) 均值
            std: (num_knots, nu) 标准差
            
        Returns:
            samples: (num_samples, num_knots, nu) 采样的控制序列
        """
        noise = torch.randn(self.num_samples, *mean.shape, device=self.device)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        return samples
    
    def evaluate_samples_mujoco(
        self, 
        samples: torch.Tensor, 
        mj_model, 
        initial_qpos: torch.Tensor,
        initial_qvel: torch.Tensor,
        dt: float = 0.01
    ) -> torch.Tensor:
        """使用MuJoCo评估采样轨迹的成本
        
        Args:
            samples: (num_samples, num_knots, nu) 控制样本
            mj_model: MuJoCo模型
            initial_qpos: 初始位置
            initial_qvel: 初始速度
            dt: 仿真时间步长
            
        Returns:
            costs: (num_samples,) 每条轨迹的总成本
        """
        costs = []
        num_steps = int(self.plan_horizon / dt)
        
        for i in range(self.num_samples):
            # 为每个样本创建独立的MuJoCo data
            mj_data = mujoco.MjData(mj_model)
            mj_data.qpos[:] = initial_qpos.cpu().numpy()
            mj_data.qvel[:] = initial_qvel.cpu().numpy()
            mujoco.mj_forward(mj_model, mj_data)
            
            total_cost = 0.0
            knots = samples[i].cpu().numpy()  # (num_knots, nu)
            
            # 对每个时间步进行仿真
            for step in range(num_steps):
                # 当前时间
                t = step * dt
                
                # 零阶插值（zero-order hold）
                knot_idx = min(int(t / self.plan_horizon * (self.num_knots - 1)), self.num_knots - 1)
                control = knots[knot_idx]
                
                # 应用控制并前进一步
                mj_data.ctrl[:] = control
                mujoco.mj_step(mj_model, mj_data)
                
                # 计算成本（简化的站立成本函数）
                # 主要目标：保持直立，脚部接触地面
                com_height = mj_data.qpos[2]  # 重心高度
                angular_vel = np.linalg.norm(mj_data.qvel[3:6])  # 角速度
                
                step_cost = (
                    10.0 * max(0, 1.0 - com_height)**2 +  # 高度惩罚
                    0.1 * angular_vel**2 +                # 角速度惩罚
                    0.01 * np.sum(control**2)              # 控制成本
                )
                total_cost += step_cost
                
            costs.append(total_cost)
        
        return torch.tensor(costs, device=self.device)
    
    def update_distribution(
        self, 
        samples: torch.Tensor, 
        costs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据成本更新分布参数
        
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
        
        # 计算新的均值和标准差
        new_mean = torch.mean(elites, dim=0)
        new_std = torch.clamp(torch.std(elites, dim=0), min=self.sigma_min)
        
        return new_mean, new_std
    
    def zero_order_interpolation(
        self, 
        knots: torch.Tensor, 
        query_times: torch.Tensor
    ) -> torch.Tensor:
        """零阶插值实现
        
        Args:
            knots: (num_knots, nu) 控制节点
            query_times: (T,) 查询时间点
            
        Returns:
            controls: (T, nu) 插值后的控制序列
        """
        # 归一化查询时间到[0, 1]
        normalized_times = query_times / self.plan_horizon
        
        # 计算对应的knot索引
        knot_indices = torch.clamp(
            (normalized_times * (self.num_knots - 1)).long(),
            0, self.num_knots - 1
        )
        
        return knots[knot_indices]


class CEMPlannerModule(nn.Module):
    """完整的CEM Planning Pipeline，可导出为TorchScript"""
    
    def __init__(
        self,
        network: nn.Module,
        num_samples: int = 500,
        num_elites: int = 20,
        sigma_start: float = 0.3,
        sigma_min: float = 0.05,
        plan_horizon: float = 0.5,
        num_knots: int = 4,
        iterations: int = 1,
        num_ctrl_steps: int = 50
    ):
        super().__init__()
        self.network = network
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.sigma_start = sigma_start
        self.sigma_min = sigma_min
        self.plan_horizon = plan_horizon
        self.num_knots = num_knots
        self.iterations = iterations
        self.num_ctrl_steps = num_ctrl_steps
        
        # 预计算时间网格
        self.register_buffer(
            'query_times', 
            torch.linspace(0, plan_horizon, num_ctrl_steps)
        )
    
    def forward(
        self, 
        state: torch.Tensor,
        initial_mean: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播：从状态预测控制序列
        
        Args:
            state: (95,) 机器人状态 [qpos, qvel]
            initial_mean: (num_knots, nu) 可选的初始均值
            
        Returns:
            controls: (num_ctrl_steps, nu) 控制序列
        """
        # 神经网络预测knots
        if initial_mean is None:
            knots_flat = self.network(state.unsqueeze(0))  # (1, 164)
            knots = knots_flat.view(self.num_knots, -1)    # (4, 41)
        else:
            knots = initial_mean
        
        # 零阶插值生成控制序列
        controls = self._zero_order_interpolation(knots, self.query_times)
        
        return controls
    
    def _zero_order_interpolation(
        self, 
        knots: torch.Tensor, 
        query_times: torch.Tensor
    ) -> torch.Tensor:
        """零阶插值实现（TorchScript兼容）"""
        # 归一化查询时间到[0, 1]
        normalized_times = query_times / self.plan_horizon
        
        # 计算对应的knot索引
        knot_indices = torch.clamp(
            (normalized_times * (self.num_knots - 1)).long(),
            0, self.num_knots - 1
        )
        
        return knots[knot_indices]


def load_original_model(model_path: str, device: torch.device) -> nn.Module:
    """加载原始PyTorch Lightning模型并转换为纯PyTorch模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建PyTorch Lightning模型
    pl_net = MLPRegressor(95, 512, 512, 512, 164)
    pl_net.load_state_dict(checkpoint['state_dict'])
    pl_net.eval()
    
    # 提取纯PyTorch模型（去掉Lightning包装）
    net = pl_net.model  # 只取nn.Sequential部分
    net.to(device).eval()
    
    print(f"已加载模型: {model_path}")
    print(f"模型类型: {type(net)}")
    return net


def export_models(
    network: nn.Module,
    output_dir: str = "exported_models",
    device: torch.device = torch.device('cuda')
):
    """导出模型到TorchScript和ONNX格式"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 导出纯神经网络
    print("导出神经网络...")
    with torch.no_grad():
        # 创建示例输入
        dummy_state = torch.randn(1, 95, device=device)
        
        # TorchScript导出
        traced_net = torch.jit.trace(network, dummy_state)
        net_ts_path = os.path.join(output_dir, "network.ts")
        traced_net.save(net_ts_path)
        print(f"神经网络TorchScript已保存: {net_ts_path}")
        
        # ONNX导出
        net_onnx_path = os.path.join(output_dir, "network.onnx")
        torch.onnx.export(
            network,
            dummy_state,
            net_onnx_path,
            input_names=['state'],
            output_names=['knots'],
            dynamic_axes={
                'state': {0: 'batch_size'},
                'knots': {0: 'batch_size'}
            },
            opset_version=11
        )
        print(f"神经网络ONNX已保存: {net_onnx_path}")
    
    # 2. 导出完整的Pipeline
    print("导出完整Pipeline...")
    pipeline = CEMPlannerModule(network)
    pipeline.eval()
    
    with torch.no_grad():
        # 创建示例输入
        dummy_state = torch.randn(95, device=device)
        
        # TorchScript导出
        traced_pipeline = torch.jit.trace(pipeline, dummy_state)
        pipeline_ts_path = os.path.join(output_dir, "cem_planner.ts")
        traced_pipeline.save(pipeline_ts_path)
        print(f"完整Pipeline TorchScript已保存: {pipeline_ts_path}")
        
        # ONNX导出
        pipeline_onnx_path = os.path.join(output_dir, "cem_planner.onnx")
        torch.onnx.export(
            pipeline,
            dummy_state,
            pipeline_onnx_path,
            input_names=['state'],
            output_names=['controls'],
            dynamic_axes={
                'state': {0: 'batch_size'},
                'controls': {0: 'batch_size'}
            },
            opset_version=11
        )
        print(f"完整Pipeline ONNX已保存: {pipeline_onnx_path}")


def test_exported_models(output_dir: str = "exported_models"):
    """测试导出的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("测试导出的模型...")
    
    # 测试TorchScript版本
    pipeline_path = os.path.join(output_dir, "cem_planner.ts")
    if os.path.exists(pipeline_path):
        loaded_pipeline = torch.jit.load(pipeline_path, map_location=device)
        
        # 创建测试输入
        test_state = torch.randn(95, device=device)
        
        with torch.no_grad():
            controls = loaded_pipeline(test_state)
            print(f"TorchScript推理成功，输出形状: {controls.shape}")
    
    # 测试ONNX版本（需要onnxruntime）
    try:
        import onnxruntime as ort
        
        pipeline_onnx_path = os.path.join(output_dir, "cem_planner.onnx")
        if os.path.exists(pipeline_onnx_path):
            # 创建ONNX推理会话
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(pipeline_onnx_path, providers=providers)
            
            # 测试推理
            test_state = np.random.randn(95).astype(np.float32)
            outputs = session.run(['controls'], {'state': test_state})
            print(f"ONNX推理成功，输出形状: {outputs[0].shape}")
    
    except ImportError:
        print("未安装onnxruntime，跳过ONNX测试")


def main():
    parser = argparse.ArgumentParser(description="导出CEM Planner到TorchScript/ONNX")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt",
        help="原始模型checkpoint路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exported_models",
        help="导出模型保存目录"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="是否测试导出的模型"
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载原始模型
    network = load_original_model(args.model_path, device)
    
    # 导出模型
    export_models(network, args.output_dir, device)
    
    # 测试模型
    if args.test:
        test_exported_models(args.output_dir)
    
    print("导出完成！")
    print(f"\n使用方法 (DOOM环境中):")
    print(f"  # 加载TorchScript版本")
    print(f"  planner = torch.jit.load('{args.output_dir}/cem_planner.ts')")
    print(f"  controls = planner(state_tensor)")
    print(f"\n  # 加载ONNX版本")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{args.output_dir}/cem_planner.onnx')")
    print(f"  controls = session.run(['controls'], {{'state': state_array}})[0]")


if __name__ == "__main__":
    main() 