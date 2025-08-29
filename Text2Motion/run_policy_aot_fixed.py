#!/usr/bin/env python3
"""
修复版本的AOT预编译管线run_policy脚本

修复的问题：
1. 时间同步逻辑
2. 频率控制逻辑  
3. 减少"其他开销"
4. 确保与原始版本完全一致的控制行为

运行方法:
    python run_policy_aot_fixed.py --show_reference
    python run_policy_aot_fixed.py --frequency 30.0
"""
import argparse
import os
import time

import numpy as np
import mujoco
import mujoco.viewer
import jax.numpy as jnp

# 导入AOT预编译管线
from export_run_policy_aot import load_precompiled_pipeline, create_and_precompile_pipeline
from hydrax.tasks.humanoid_standonly import HumanoidStand


def main():
    parser = argparse.ArgumentParser(
        description="修复版本的AOT预编译管线运行策略控制"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt",
        help="PyTorch模型checkpoint路径"
    )
    parser.add_argument(
        "--precompiled_dir",
        type=str,
        default="exported_models/precompiled",
        help="预编译管线目录路径"
    )
    parser.add_argument(
        "--show_reference", 
        action="store_true",
        help="显示参考轨迹的透明鬼影"
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
        help="规划频率 (Hz)"
    )
    parser.add_argument(
        "--recompile",
        action="store_true",
        help="强制重新编译管线"
    )

    args = parser.parse_args()

    print(f"🔧 修复版本AOT预编译管线运行策略控制")
    print(f"📁 模型路径: {args.model_path}")
    print(f"📂 预编译目录: {args.precompiled_dir}")
    print(f"🎯 规划频率: {args.frequency} Hz")

    # 1. 加载或创建预编译管线
    pipeline = None
    
    if not args.recompile and os.path.exists(args.precompiled_dir):
        try:
            print("📦 加载现有预编译管线...")
            pipeline = load_precompiled_pipeline(args.precompiled_dir)
            print("✅ 预编译管线加载成功")
        except Exception as e:
            print(f"⚠️ 加载预编译管线失败: {e}")
            print("🔄 将创建新的预编译管线...")
    
    if pipeline is None:
        print("🔨 创建新的预编译管线...")
        pipeline = create_and_precompile_pipeline(
            model_path=args.model_path,
            device='cuda',
            frequency=args.frequency,
            precompile_depth=3
        )
        print("✅ 预编译管线创建完成")

    # 2. 设置MuJoCo环境 (与原始run_policy.py完全一致)
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

    # 3. 关键修复：使用与原始版本完全相同的时间参数计算
    replan_period = 1.0 / args.frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    
    print(f"\n📊 修复后的仿真参数:")
    print(f"   - 规划频率: {args.frequency} Hz")
    print(f"   - 重规划周期: {step_dt:.4f}s")
    print(f"   - 每次规划的仿真步数: {sim_steps_per_replan}")
    print(f"   - MuJoCo时间步: {mj_model.opt.timestep:.4f}s")

    # 4. 设置参考轨迹显示 (与原始run_policy.py完全一致)
    ref_data = None
    vopt = None
    pert = None
    catmask = None
    
    if reference is not None:
        ref_data = mujoco.MjData(mj_model)
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

    # 5. 启动MuJoCo可视化器
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    print("🎮 MuJoCo可视化器已启动")

    # 添加参考轨迹几何体
    if reference is not None:
        mujoco.mjv_addGeoms(mj_model, ref_data, vopt, pert, catmask, viewer.user_scn)
        print("👻 参考轨迹鬼影已添加")

    # 6. 初始化时间统计
    start_time = time.time()
    planning_times = []
    nn_times = []
    cem_times = []
    simulation_times = []
    viewer_times = []
    total_cycles = 0
    last_stats_time = start_time

    print(f"\n🎬 开始仿真 (修复版本)...")

    # 7. 主仿真循环 (修复版本)
    while True:
        # 记录循环开始时间 (用于精确的频率控制)
        cycle_start_time = time.time()
        
        # 检查可视化器状态
        if viewer is not None and not viewer.is_running():
            break
        
        # 更新循环计数
        total_cycles += 1
        
        # 规划步骤 - 修复版本，减少额外开销
        plan_start = time.time()
        
        # 关键修复：直接使用仿真时间而不进行额外的时间计算
        controls, timing_info = pipeline.predict_controls(
            qpos=mj_data.qpos,
            qvel=mj_data.qvel,
            mocap_pos=mj_data.mocap_pos,
            mocap_quat=mj_data.mocap_quat,
            current_time=mj_data.time,  # 使用仿真的实际时间
            return_timing=True
        )
        
        plan_time = time.time() - plan_start
        planning_times.append(plan_time)
        
        # 记录时间信息 (简化版本，减少开销)
        if timing_info:
            nn_times.append(timing_info['nn_time'])
            cem_times.append(timing_info['cem_time'])
        else:
            nn_times.append(0.0)
            cem_times.append(0.0)

        # 简化的时间记录打印 (减少频次)
        if total_cycles % 20 == 1:
            print(f"\n📊 Cycle #{total_cycles} - 修复版AOT时间记录:")
            print(f"   ⏱️ 总规划时间: {plan_time:.4f}s")
            if timing_info:
                print(f"      - NN推理: {timing_info['nn_time']:.4f}s")
                print(f"      - CEM优化: {timing_info['cem_time']:.4f}s")
            
            # 运行平均
            recent_planning = planning_times[-20:]
            recent_nn = nn_times[-20:]
            recent_cem = cem_times[-20:]
            print(f"   📈 滑动平均 (最近20个周期):")
            print(f"      - 总规划: {np.mean(recent_planning):.4f}s")
            print(f"      - NN推理: {np.mean(recent_nn):.4f}s")
            print(f"      - CEM优化: {np.mean(recent_cem):.4f}s")

        # 更新参考轨迹 (与原始run_policy.py完全一致)
        if viewer is not None and reference is not None:
            t_ref = mj_data.time * 30  # reference_fps = 30
            i_ref = min(int(t_ref), reference.shape[0] - 1)
            ref_data.qpos[:] = reference[i_ref]
            mujoco.mj_forward(mj_model, ref_data)
            mujoco.mjv_updateScene(
                mj_model, ref_data, vopt, pert, viewer.cam, catmask, viewer.user_scn
            )

        # 关键修复：仿真步骤与原始版本完全一致
        sim_start = time.time()
        
        for i in range(sim_steps_per_replan):
            # 应用控制 (与原始版本逻辑一致)
            if i < len(controls):
                mj_data.ctrl[:] = np.array(controls[i])
            else:
                # 如果控制序列不够长，使用最后一个控制
                mj_data.ctrl[:] = np.array(controls[-1])
            
            # 仿真一步
            mujoco.mj_step(mj_model, mj_data)

            # 可视化器同步
            if viewer is not None:
                viewer_start = time.time()
                viewer.sync()
                viewer_times.append(time.time() - viewer_start)

        sim_time = time.time() - sim_start
        simulation_times.append(sim_time)

        # 综合统计报告 (降低频次以减少开销)
        current_time = time.time()
        if total_cycles % 100 == 0:
            elapsed_total = current_time - start_time
            
            print(f"\n🎯 === 修复版AOT统计报告 (Cycle #{total_cycles}) ===")
            print(f"📊 总体统计:")
            print(f"   - 总运行时间: {elapsed_total:.2f}s")
            print(f"   - 总循环次数: {total_cycles}")
            print(f"   - 平均循环频率: {total_cycles / elapsed_total:.2f} Hz")
            print(f"   - 目标频率: {args.frequency:.2f} Hz")
            print(f"   - 频率达成率: {(total_cycles / elapsed_total) / args.frequency * 100:.1f}%")
            
            # 最近性能分析
            recent_planning = planning_times[-100:]
            recent_nn = nn_times[-100:]
            recent_cem = cem_times[-100:]
            recent_sim = simulation_times[-100:]
            recent_viewer = viewer_times[-len(recent_sim)*sim_steps_per_replan:] if viewer_times else []
            
            print(f"\n⏱️ 修复版AOT耗时分析 (最近100个循环):")
            print(f"   - AOT规划时间: {np.mean(recent_planning)*1000:.2f}ms")
            print(f"      └─ NN推理: {np.mean(recent_nn)*1000:.2f}ms ({np.mean(recent_nn)/np.mean(recent_planning)*100:.1f}%)")
            print(f"      └─ CEM优化: {np.mean(recent_cem)*1000:.2f}ms ({np.mean(recent_cem)/np.mean(recent_planning)*100:.1f}%)")
            total_accounted = np.mean(recent_nn) + np.mean(recent_cem)
            other_overhead = np.mean(recent_planning) - total_accounted
            print(f"      └─ 其他开销: {other_overhead*1000:.2f}ms ({other_overhead/np.mean(recent_planning)*100:.1f}%)")
            print(f"   - 仿真时间: {np.mean(recent_sim)*1000:.2f}ms")
            if recent_viewer:
                print(f"   - 可视化时间: {np.mean(recent_viewer)*1000:.2f}ms")
            
            total_cycle_time = np.mean(recent_planning) + np.mean(recent_sim)
            print(f"   - 单循环总时间: {total_cycle_time*1000:.2f}ms")
            print(f"   - 理论最大频率: {1.0/total_cycle_time:.2f} Hz")
            
            # 频率达成评估
            actual_freq = total_cycles / elapsed_total
            if actual_freq >= args.frequency * 0.8:
                print("✅ 频率控制：优秀，接近目标频率")
            elif actual_freq >= args.frequency * 0.5:
                print("⚡ 频率控制：良好，达到目标频率的50%以上")
            else:
                print("⚠️ 频率控制：需要优化，远低于目标频率")
            
            last_stats_time = current_time

        # 关键修复：精确的频率控制 (与原始版本一致)
        cycle_elapsed = time.time() - cycle_start_time
        target_cycle_time = step_dt  # 使用正确的周期时间
        if cycle_elapsed < target_cycle_time:
            time.sleep(target_cycle_time - cycle_elapsed)

    print(f"\n🎉 仿真结束")
    print(f"📊 最终统计:")
    print(f"   - 总循环数: {total_cycles}")
    print(f"   - 总运行时间: {time.time() - start_time:.2f}s")
    if planning_times:
        print(f"   - 平均规划时间: {np.mean(planning_times)*1000:.2f}ms")
        print(f"   - 平均仿真时间: {np.mean(simulation_times)*1000:.2f}ms")
        actual_freq = total_cycles / (time.time() - start_time)
        print(f"   - 实际平均频率: {actual_freq:.2f} Hz")
        print(f"   - 频率达成率: {actual_freq / args.frequency * 100:.1f}%")


if __name__ == "__main__":
    main() 