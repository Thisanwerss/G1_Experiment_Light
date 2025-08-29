#!/usr/bin/env python3
"""
在DOOM环境中使用编译后的run_policy pipeline

这个脚本展示如何在DOOM环境中加载和使用编译后的pipeline：
1. 零启动开销的函数调用
2. 与DOOM环境的接口
3. 性能监控和统计

运行方法:
    python doom_compiled_policy_runner.py --pipeline_path exported_models/run_policy_compiled.pkl
"""

import argparse
import time
import numpy as np
from typing import Dict, Any, Tuple

# 假设DOOM环境的接口
class DoomEnvironment:
    """模拟的DOOM环境接口"""
    
    def __init__(self):
        # 模拟人形机器人在DOOM中的状态
        self.nq = 48  # 关节位置维度
        self.nv = 47  # 关节速度维度
        self.nu = 41  # 控制维度
        
        # 当前状态
        self.qpos = np.zeros(self.nq)
        self.qvel = np.zeros(self.nv)
        self.time = 0.0
        
        # 仿真参数
        self.dt = 0.01  # 与MuJoCo保持一致
        
        print(f"DOOM环境初始化: nq={self.nq}, nv={self.nv}, nu={self.nu}")
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """获取当前状态"""
        return self.qpos.copy(), self.qvel.copy(), self.time
    
    def apply_controls(self, controls: np.ndarray) -> bool:
        """应用控制并更新状态
        
        Args:
            controls: (T, nu) 控制序列
            
        Returns:
            是否成功应用
        """
        if controls.shape[1] != self.nu:
            print(f"控制维度不匹配: 期望{self.nu}, 实际{controls.shape[1]}")
            return False
        
        # 模拟状态更新（简化版本）
        num_steps = len(controls)
        for i, control in enumerate(controls):
            # 简单的状态更新（实际DOOM中会有更复杂的物理仿真）
            self.qpos += self.qvel * self.dt
            self.qvel += np.random.randn(self.nv) * 0.01  # 添加一些随机性
            self.time += self.dt
        
        return True
    
    def is_running(self) -> bool:
        """环境是否还在运行"""
        return self.time < 100.0  # 运行100秒后停止


class DoomCompiledPolicyRunner:
    """DOOM环境中的编译policy运行器"""
    
    def __init__(self, pipeline_path: str):
        """初始化运行器
        
        Args:
            pipeline_path: 编译后的pipeline路径
        """
        print(f"初始化DOOM Policy Runner...")
        print(f"Pipeline路径: {pipeline_path}")
        
        # 加载编译后的pipeline
        from export_run_policy import load_compiled_pipeline
        self.pipeline = load_compiled_pipeline(pipeline_path)
        
        # 获取仿真参数
        self.sim_params = self.pipeline.get_simulation_params()
        print("仿真参数:")
        for key, value in self.sim_params.items():
            print(f"  {key}: {value}")
        
        # 性能统计
        self.stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'prediction_times': [],
            'nn_times': [],
            'prep_times': [],
            'cem_times': [],
            'interp_times': []
        }
        
        print("DOOM Policy Runner初始化完成!")
    
    def predict_controls(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        current_time: float = 0.0
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """预测控制序列（带性能统计）
        
        Args:
            qpos: 关节位置
            qvel: 关节速度
            current_time: 当前时间
            
        Returns:
            controls: 控制序列
            timing_info: 时间统计
        """
        start_time = time.time()
        
        # 调用编译后的pipeline
        controls, timing_info = self.pipeline.predict_controls(
            qpos, qvel, 
            current_time=current_time,
            return_timing=True
        )
        
        total_time = time.time() - start_time
        
        # 更新统计信息
        self.stats['total_calls'] += 1
        self.stats['total_time'] += total_time
        self.stats['prediction_times'].append(total_time)
        
        if timing_info:
            self.stats['nn_times'].append(timing_info.get('nn_time', 0))
            self.stats['prep_times'].append(timing_info.get('prep_time', 0))
            self.stats['cem_times'].append(timing_info.get('cem_time', 0))
            self.stats['interp_times'].append(timing_info.get('interp_time', 0))
        
        return controls, timing_info
    
    def run_doom_simulation(
        self,
        doom_env: DoomEnvironment,
        duration: float = 30.0,
        verbose: bool = True
    ):
        """在DOOM环境中运行仿真
        
        Args:
            doom_env: DOOM环境实例
            duration: 运行时长（秒）
            verbose: 是否打印详细信息
        """
        print(f"\n开始DOOM仿真，时长: {duration}秒")
        
        start_time = time.time()
        step_count = 0
        prediction_count = 0
        
        # 获取仿真步长
        steps_per_prediction = self.sim_params['sim_steps_per_replan']
        
        while doom_env.is_running() and (time.time() - start_time) < duration:
            # 获取当前状态
            qpos, qvel, env_time = doom_env.get_state()
            
            # 预测控制
            controls, timing_info = self.predict_controls(qpos, qvel, env_time)
            prediction_count += 1
            
            # 应用控制
            success = doom_env.apply_controls(controls)
            if not success:
                print("❌ 控制应用失败")
                break
            
            step_count += len(controls)
            
            # 打印进度（每10次预测）
            if verbose and prediction_count % 10 == 0:
                elapsed = time.time() - start_time
                avg_pred_time = np.mean(self.stats['prediction_times'][-10:])
                print(f"时间: {elapsed:.1f}s, 预测#{prediction_count}, "
                      f"平均预测时间: {avg_pred_time:.4f}s")
        
        # 仿真结束统计
        total_time = time.time() - start_time
        print(f"\n仿真完成!")
        print(f"总时间: {total_time:.2f}s")
        print(f"总预测次数: {prediction_count}")
        print(f"总仿真步数: {step_count}")
        print(f"平均预测频率: {prediction_count / total_time:.2f} Hz")
        
        self.print_performance_stats()
    
    def print_performance_stats(self):
        """打印详细的性能统计"""
        if self.stats['total_calls'] == 0:
            print("无性能数据")
            return
        
        print(f"\n📊 性能统计 ({self.stats['total_calls']}次调用):")
        
        # 总体统计
        times = np.array(self.stats['prediction_times'])
        print(f"总体预测:")
        print(f"  平均耗时: {np.mean(times):.4f}s")
        print(f"  最小耗时: {np.min(times):.4f}s")
        print(f"  最大耗时: {np.max(times):.4f}s")
        print(f"  标准差: {np.std(times):.4f}s")
        print(f"  理论最大频率: {1.0/np.mean(times):.2f} Hz")
        
        # 详细分解
        if self.stats['nn_times']:
            nn_times = np.array(self.stats['nn_times'])
            prep_times = np.array(self.stats['prep_times'])
            cem_times = np.array(self.stats['cem_times'])
            interp_times = np.array(self.stats['interp_times'])
            
            print(f"\n详细耗时分解:")
            print(f"  NN推理: {np.mean(nn_times):.4f}s ({np.mean(nn_times)/np.mean(times)*100:.1f}%)")
            print(f"  数据准备: {np.mean(prep_times):.4f}s ({np.mean(prep_times)/np.mean(times)*100:.1f}%)")
            print(f"  CEM优化: {np.mean(cem_times):.4f}s ({np.mean(cem_times)/np.mean(times)*100:.1f}%)")
            print(f"  插值: {np.mean(interp_times):.4f}s ({np.mean(interp_times)/np.mean(times)*100:.1f}%)")
        
        # 与目标频率对比
        target_freq = self.sim_params['frequency']
        actual_freq = 1.0 / np.mean(times)
        success_rate = min(1.0, actual_freq / target_freq) * 100
        print(f"\n频率对比:")
        print(f"  目标频率: {target_freq:.2f} Hz")
        print(f"  实际频率: {actual_freq:.2f} Hz")
        print(f"  达成率: {success_rate:.1f}%")
    
    def benchmark_performance(self, num_tests: int = 100):
        """性能基准测试"""
        print(f"\n🚀 开始性能基准测试 ({num_tests}次)")
        
        # 生成测试数据
        nq, nv = 48, 47
        test_data = [
            (np.random.randn(nq) * 0.1, np.random.randn(nv) * 0.1, i * 0.02)
            for i in range(num_tests)
        ]
        
        # 预热（排除第一次JIT编译的影响）
        print("预热...")
        for i in range(3):
            qpos, qvel, t = test_data[i]
            self.predict_controls(qpos, qvel, t)
        
        # 清空统计
        self.stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'prediction_times': [],
            'nn_times': [],
            'prep_times': [],
            'cem_times': [],
            'interp_times': []
        }
        
        # 正式测试
        print(f"开始基准测试...")
        overall_start = time.time()
        
        for i, (qpos, qvel, t) in enumerate(test_data):
            controls, timing_info = self.predict_controls(qpos, qvel, t)
            
            # 每20次打印进度
            if (i + 1) % 20 == 0:
                progress = (i + 1) / num_tests * 100
                print(f"进度: {progress:.1f}% ({i+1}/{num_tests})")
        
        overall_time = time.time() - overall_start
        print(f"基准测试完成，总耗时: {overall_time:.2f}s")
        
        # 打印结果
        self.print_performance_stats()


def main():
    parser = argparse.ArgumentParser(description="DOOM环境中的编译policy运行器")
    
    parser.add_argument(
        "--pipeline_path",
        type=str,
        default="exported_models/run_policy_compiled.pkl",
        help="编译后的pipeline路径"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "benchmark", "both"],
        default="both",
        help="运行模式"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="仿真时长（秒）"
    )
    
    parser.add_argument(
        "--num_benchmark",
        type=int,
        default=100,
        help="基准测试次数"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建运行器
        runner = DoomCompiledPolicyRunner(args.pipeline_path)
        
        if args.mode in ["simulation", "both"]:
            # 创建DOOM环境并运行仿真
            doom_env = DoomEnvironment()
            runner.run_doom_simulation(doom_env, args.duration)
        
        if args.mode in ["benchmark", "both"]:
            # 性能基准测试
            runner.benchmark_performance(args.num_benchmark)
        
        print(f"\n🎉 运行完成!")
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 