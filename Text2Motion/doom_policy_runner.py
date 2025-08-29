#!/usr/bin/env python3
"""
DOOM环境中运行PyTorch CEM Policy的示例脚本

这个脚本展示了如何在没有JAX依赖的DOOM环境中：
1. 加载导出的TorchScript/ONNX模型
2. 执行神经网络预测或在线CEM优化
3. 控制机器人运动

使用方法:
    python doom_policy_runner.py --model_type torchscript --model_path exported_models/cem_planner.ts
    python doom_policy_runner.py --model_type onnx --model_path exported_models/cem_planner.onnx
    python doom_policy_runner.py --model_type pytorch_cem --model_path exported_models/network.ts
"""

import argparse
import time
import os
import numpy as np
import torch
import mujoco
import mujoco.viewer
from typing import Optional, Union

# 尝试导入ONNX Runtime（可选）
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("警告: 未安装onnxruntime，无法使用ONNX模型")

# 导入我们的PyTorch CEM实现
from pytorch_cem_planner import CEMPlanner


class DOOMPolicyRunner:
    """DOOM环境中的策略运行器"""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'torchscript',
        mujoco_model_path: str = None,
        device: str = 'auto',
        planning_frequency: float = 50.0
    ):
        """初始化策略运行器
        
        Args:
            model_path: 模型文件路径
            model_type: 模型类型 ('torchscript', 'onnx', 'pytorch_cem')
            mujoco_model_path: MuJoCo模型文件路径
            device: 计算设备
            planning_frequency: 规划频率 (Hz)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.planning_frequency = planning_frequency
        self.plan_dt = 1.0 / planning_frequency
        
        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"初始化DOOM策略运行器")
        print(f"模型类型: {model_type}")
        print(f"模型路径: {model_path}")
        print(f"设备: {self.device}")
        print(f"规划频率: {planning_frequency} Hz")
        
        # 加载模型
        self.model = None
        self.onnx_session = None
        self.cem_planner = None
        
        self._load_model()
        
        # 设置MuJoCo环境
        self._setup_mujoco(mujoco_model_path)
    
    def _load_model(self):
        """加载指定类型的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        if self.model_type == 'torchscript':
            self._load_torchscript_model()
        elif self.model_type == 'onnx':
            self._load_onnx_model()
        elif self.model_type == 'pytorch_cem':
            self._load_pytorch_cem_model()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _load_torchscript_model(self):
        """加载TorchScript模型"""
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            print(f"成功加载TorchScript模型: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"加载TorchScript模型失败: {e}")
    
    def _load_onnx_model(self):
        """加载ONNX模型"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime未安装，无法加载ONNX模型")
        
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(self.model_path, providers=providers)
            print(f"成功加载ONNX模型: {self.model_path}")
            
            # 显示模型信息
            input_info = self.onnx_session.get_inputs()[0]
            output_info = self.onnx_session.get_outputs()[0]
            print(f"输入: {input_info.name}, 形状: {input_info.shape}")
            print(f"输出: {output_info.name}, 形状: {output_info.shape}")
            
        except Exception as e:
            raise RuntimeError(f"加载ONNX模型失败: {e}")
    
    def _load_pytorch_cem_model(self):
        """加载PyTorch CEM模型（支持在线优化）"""
        try:
            self.cem_planner = CEMPlanner(
                model_path=self.model_path,
                num_samples=200,  # 在DOOM中使用较少的样本以提高速度
                num_elites=10,
                iterations=1,
                device=self.device,
                use_network_warmstart=True
            )
            print(f"成功加载PyTorch CEM模型: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"加载PyTorch CEM模型失败: {e}")
    
    def _setup_mujoco(self, model_path: Optional[str] = None):
        """设置MuJoCo环境"""
        # 如果没有指定模型路径，使用默认的人形机器人模型
        if model_path is None:
            # 假设DOOM环境中有可用的人形机器人模型
            # 这里需要根据实际DOOM环境调整
            model_path = "path/to/humanoid/model.xml"  # 需要替换为实际路径
        
        if not os.path.exists(model_path):
            print(f"警告: MuJoCo模型文件不存在: {model_path}")
            print("将使用模拟参数代替实际仿真")
            self.mj_model = None
            self.mj_data = None
            return
        
        try:
            self.mj_model = mujoco.MjModel.from_xml_path(model_path)
            self.mj_data = mujoco.MjData(self.mj_model)
            
            # 设置仿真参数
            self.mj_model.opt.timestep = 0.01
            self.mj_model.opt.iterations = 10
            
            print(f"成功加载MuJoCo模型: {model_path}")
            print(f"自由度: {self.mj_model.nq}, 控制维度: {self.mj_model.nu}")
            
        except Exception as e:
            print(f"加载MuJoCo模型失败: {e}")
            self.mj_model = None
            self.mj_data = None
    
    def predict_with_torchscript(self, state: np.ndarray) -> np.ndarray:
        """使用TorchScript模型进行预测"""
        state_tensor = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            controls = self.model(state_tensor)
            
        return controls.cpu().numpy()
    
    def predict_with_onnx(self, state: np.ndarray) -> np.ndarray:
        """使用ONNX模型进行预测"""
        # 确保输入类型正确
        state_input = state.astype(np.float32)
        
        # 执行推理
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: state_input})
        
        return outputs[0]
    
    def predict_with_pytorch_cem(self, state: np.ndarray) -> tuple:
        """使用PyTorch CEM进行在线优化"""
        if self.mj_model is None or self.mj_data is None:
            # 如果没有MuJoCo模型，回退到纯网络预测
            knots = self.cem_planner.predict_initial_knots(
                torch.from_numpy(state).float().to(self.device)
            )
            
            # 简单的零阶插值
            query_times = torch.linspace(0, 0.5, 50, device=self.device)
            controls = self.cem_planner.zero_order_interpolation(knots, query_times)
            
            info = {
                'planning_time': 0.001,  # 估计值
                'best_cost': 0.0,
                'method': 'network_only'
            }
            
            return controls.cpu().numpy(), info
        
        # 执行完整的CEM优化
        controls, info = self.cem_planner.plan(state, self.mj_model, self.mj_data)
        return controls.cpu().numpy(), info
    
    def get_current_state(self) -> np.ndarray:
        """获取当前机器人状态"""
        if self.mj_model is None or self.mj_data is None:
            # 生成模拟状态用于测试
            return np.random.randn(95).astype(np.float32)
        
        # 从MuJoCo获取真实状态
        qpos = self.mj_data.qpos.copy()
        qvel = self.mj_data.qvel.copy()
        state = np.concatenate([qpos, qvel]).astype(np.float32)
        
        return state
    
    def apply_controls(self, controls: np.ndarray, start_idx: int = 0):
        """应用控制序列"""
        if self.mj_model is None or self.mj_data is None:
            # 模拟模式：只打印控制信息
            print(f"模拟应用控制 [{start_idx}]: {controls[start_idx][:5]}...")
            return
        
        # 应用实际控制
        if start_idx < len(controls):
            self.mj_data.ctrl[:] = controls[start_idx]
            mujoco.mj_step(self.mj_model, self.mj_data)
    
    def run_policy(self, duration: float = 10.0, visualize: bool = True):
        """运行策略控制
        
        Args:
            duration: 运行时长（秒）
            visualize: 是否可视化
        """
        print(f"\n开始运行策略，时长: {duration}秒")
        print(f"规划频率: {self.planning_frequency} Hz")
        
        # 可视化设置
        viewer = None
        if visualize and self.mj_model is not None:
            try:
                viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            except:
                print("无法启动MuJoCo可视化器")
                viewer = None
        
        # 运行主循环
        start_time = time.time()
        last_plan_time = 0
        plan_step = 0
        current_controls = None
        planning_times = []
        
        try:
            while time.time() - start_time < duration:
                current_time = time.time() - start_time
                
                # 检查是否需要重新规划
                if current_time - last_plan_time >= self.plan_dt:
                    plan_start = time.time()
                    
                    # 获取当前状态
                    state = self.get_current_state()
                    
                    # 执行预测/规划
                    if self.model_type == 'torchscript':
                        current_controls = self.predict_with_torchscript(state)
                        info = {'method': 'torchscript', 'planning_time': time.time() - plan_start}
                    
                    elif self.model_type == 'onnx':
                        current_controls = self.predict_with_onnx(state)
                        info = {'method': 'onnx', 'planning_time': time.time() - plan_start}
                    
                    elif self.model_type == 'pytorch_cem':
                        current_controls, info = self.predict_with_pytorch_cem(state)
                    
                    planning_times.append(info['planning_time'])
                    last_plan_time = current_time
                    plan_step = 0
                    
                    # 打印规划信息
                    if len(planning_times) % 10 == 1:  # 每10次规划打印一次
                        avg_time = np.mean(planning_times[-10:])
                        print(f"时间: {current_time:.2f}s, 规划时间: {info['planning_time']:.4f}s "
                              f"(平均: {avg_time:.4f}s), 方法: {info.get('method', 'unknown')}")
                
                # 应用当前控制
                if current_controls is not None:
                    self.apply_controls(current_controls, plan_step)
                    plan_step += 1
                
                # 更新可视化
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                
                # 控制循环频率
                time.sleep(0.001)  # 1ms最小延迟
        
        except KeyboardInterrupt:
            print("\n用户中断")
        
        finally:
            if viewer is not None:
                viewer.close()
        
        # 打印统计信息
        print(f"\n运行完成")
        print(f"总规划次数: {len(planning_times)}")
        if planning_times:
            print(f"平均规划时间: {np.mean(planning_times):.4f}s")
            print(f"最大规划时间: {np.max(planning_times):.4f}s")
            print(f"最小规划时间: {np.min(planning_times):.4f}s")


def main():
    parser = argparse.ArgumentParser(description="DOOM环境中运行PyTorch CEM Policy")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型文件路径"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['torchscript', 'onnx', 'pytorch_cem'],
        default='torchscript',
        help="模型类型"
    )
    
    parser.add_argument(
        "--mujoco_model",
        type=str,
        default=None,
        help="MuJoCo模型文件路径"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="运行时长（秒）"
    )
    
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
        help="规划频率 (Hz)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default='auto',
        help="计算设备 (auto/cuda/cpu)"
    )
    
    parser.add_argument(
        "--no_visualize",
        action='store_true',
        help="禁用可视化"
    )
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 创建策略运行器
    try:
        runner = DOOMPolicyRunner(
            model_path=args.model_path,
            model_type=args.model_type,
            mujoco_model_path=args.mujoco_model,
            device=args.device,
            planning_frequency=args.frequency
        )
        
        # 运行策略
        runner.run_policy(
            duration=args.duration,
            visualize=not args.no_visualize
        )
        
    except Exception as e:
        print(f"运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 