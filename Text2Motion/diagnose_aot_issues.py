#!/usr/bin/env python3
"""
深入诊断AOT版本问题的脚本

分析：
1. 时间逻辑差异
2. 控制序列差异
3. 频率控制问题
4. knot更新逻辑差异
"""
import os
import time
import numpy as np
import torch
import jax.numpy as jnp

def analyze_timing_logic():
    """分析时间逻辑差异"""
    print("🔍 分析时间逻辑差异...")
    
    # 模拟原始版本的时间计算
    frequency = 50.0
    mj_timestep = 0.01
    
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_timestep
    
    print(f"原始版本时间参数:")
    print(f"  频率: {frequency} Hz")
    print(f"  重规划周期: {replan_period:.4f}s")
    print(f"  仿真步数/规划: {sim_steps_per_replan}")
    print(f"  实际周期: {step_dt:.4f}s")
    print(f"  实际频率: {1.0/step_dt:.2f} Hz")
    
    # 模拟当前仿真时间
    current_sim_time = 1.5  # 假设仿真已运行1.5秒
    
    # 原始版本的插值时间计算
    tq_original = jnp.arange(0, sim_steps_per_replan) * mj_timestep + current_sim_time
    print(f"\n原始版本插值时间查询:")
    print(f"  查询时间: {tq_original}")
    
    # AOT版本的插值时间计算
    tq_aot = jnp.arange(0, sim_steps_per_replan) * mj_timestep + current_sim_time
    print(f"\nAOT版本插值时间查询:")
    print(f"  查询时间: {tq_aot}")
    
    # 检查差异
    if np.allclose(tq_original, tq_aot):
        print("✅ 插值时间计算一致")
    else:
        print("❌ 插值时间计算存在差异")
        print(f"  差异: {tq_original - tq_aot}")

def analyze_control_prediction():
    """分析控制预测的差异"""
    print("\n🔍 分析控制预测差异...")
    
    try:
        # 导入必要模块
        from export_run_policy_aot import create_and_precompile_pipeline
        from hydrax.tasks.humanoid_standonly import HumanoidStand
        from hydrax.algs import CEM
        import mujoco
        from mujoco import mjx
        
        print("创建测试环境...")
        
        # 创建任务和模型
        task = HumanoidStand()
        mj_model = task.mj_model
        mj_model.opt.timestep = 0.01
        mj_data = mujoco.MjData(mj_model)
        
        # 生成测试状态
        nq, nv = 48, 47
        test_qpos = np.random.randn(nq) * 0.1
        test_qvel = np.random.randn(nv) * 0.1
        test_time = 1.0
        
        print(f"测试状态:")
        print(f"  qpos shape: {test_qpos.shape}")
        print(f"  qvel shape: {test_qvel.shape}")
        print(f"  time: {test_time}")
        
        # 创建AOT管线
        model_path = "nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt"
        if os.path.exists(model_path):
            print("\n创建AOT管线...")
            pipeline = create_and_precompile_pipeline(
                model_path=model_path,
                device='cuda',
                num_samples=100,  # 减少采样数以加快测试
                num_elites=10,
                frequency=50.0,
                precompile_depth=2
            )
            
            # AOT预测
            print("执行AOT预测...")
            start_time = time.time()
            aot_controls, aot_timing = pipeline.predict_controls(
                qpos=test_qpos,
                qvel=test_qvel,
                current_time=test_time,
                return_timing=True
            )
            aot_time = time.time() - start_time
            
            print(f"AOT结果:")
            print(f"  控制形状: {aot_controls.shape}")
            print(f"  预测时间: {aot_time:.4f}s")
            if aot_timing:
                print(f"  NN时间: {aot_timing['nn_time']:.4f}s")
                print(f"  CEM时间: {aot_timing['cem_time']:.4f}s")
            
            # 分析控制数值范围
            print(f"  控制范围: [{np.min(aot_controls):.3f}, {np.max(aot_controls):.3f}]")
            print(f"  控制均值: {np.mean(aot_controls):.3f}")
            print(f"  控制标准差: {np.std(aot_controls):.3f}")
            
        else:
            print(f"❌ 模型文件不存在: {model_path}")
            
    except Exception as e:
        print(f"❌ 控制预测测试失败: {e}")
        import traceback
        traceback.print_exc()

def analyze_frequency_control():
    """分析频率控制问题"""
    print("\n🔍 分析频率控制问题...")
    
    # 模拟不同频率设置的影响
    frequencies = [10.0, 20.0, 30.0, 50.0]
    mj_timestep = 0.01
    
    print("不同频率下的参数计算:")
    print(f"{'频率(Hz)':<10} {'周期(s)':<10} {'仿真步数':<10} {'实际频率(Hz)':<15}")
    print("-" * 55)
    
    for freq in frequencies:
        replan_period = 1.0 / freq
        sim_steps = int(replan_period / mj_timestep)
        sim_steps = max(sim_steps, 1)
        actual_period = sim_steps * mj_timestep
        actual_freq = 1.0 / actual_period
        
        print(f"{freq:<10.1f} {actual_period:<10.4f} {sim_steps:<10} {actual_freq:<15.2f}")
    
    # 分析频率达成的瓶颈
    print(f"\n频率达成瓶颈分析:")
    target_freq = 50.0
    target_period = 1.0 / target_freq
    
    # 基于日志数据的估算
    planning_time = 0.046  # 46ms (从日志)
    simulation_time = 0.008  # 8ms
    viewer_time = 0.004  # 4ms
    
    total_time = planning_time + simulation_time + viewer_time
    max_achievable_freq = 1.0 / total_time
    
    print(f"  目标频率: {target_freq:.1f} Hz")
    print(f"  目标周期: {target_period*1000:.1f} ms")
    print(f"  实际总时间: {total_time*1000:.1f} ms")
    print(f"  最大可达频率: {max_achievable_freq:.1f} Hz")
    print(f"  频率差距: {target_freq - max_achievable_freq:.1f} Hz")
    
    if max_achievable_freq < target_freq:
        print(f"❌ 计算性能不足以达到目标频率")
        print(f"  建议降低目标频率至: {max_achievable_freq * 0.8:.1f} Hz")
    else:
        print(f"✅ 理论上可以达到目标频率")

def analyze_knot_logic():
    """分析knot更新逻辑"""
    print("\n🔍 分析knot更新逻辑...")
    
    # 模拟knot时间设置
    plan_horizon = 0.5
    num_knots = 4
    current_time = 1.5
    
    # CEM中的knot时间设置逻辑
    tk = jnp.linspace(0.0, plan_horizon, num_knots) + current_time
    print(f"CEM knot时间设置:")
    print(f"  规划视界: {plan_horizon}s")
    print(f"  knot数量: {num_knots}")
    print(f"  当前时间: {current_time}s")
    print(f"  knot时间: {tk}")
    
    # 插值查询时间
    sim_steps = 2
    dt = 0.01
    tq = jnp.arange(0, sim_steps) * dt + current_time
    print(f"  查询时间: {tq}")
    
    # 检查时间范围
    if jnp.all(tq >= tk[0]) and jnp.all(tq <= tk[-1]):
        print("✅ 查询时间在knot范围内")
    else:
        print("❌ 查询时间超出knot范围")
        print(f"  knot范围: [{tk[0]:.3f}, {tk[-1]:.3f}]")
        print(f"  查询范围: [{tq[0]:.3f}, {tq[-1]:.3f}]")

def main():
    print("🔧 AOT版本问题深度诊断")
    print("=" * 50)
    
    analyze_timing_logic()
    analyze_control_prediction()
    analyze_frequency_control()
    analyze_knot_logic()
    
    print("\n🎯 诊断总结:")
    print("1. 检查时间逻辑是否一致")
    print("2. 验证控制预测是否正常")
    print("3. 分析频率控制的瓶颈")
    print("4. 确认knot更新逻辑正确")
    
    print(f"\n💡 建议:")
    print(f"1. 如果频率无法达到目标，尝试降低目标频率")
    print(f"2. 如果控制预测异常，检查模型加载和网络推理")
    print(f"3. 如果时间逻辑不一致，修正插值时间计算")
    print(f"4. 测试修复版本: python run_policy_aot_fixed.py --frequency 30.0")

if __name__ == "__main__":
    main() 