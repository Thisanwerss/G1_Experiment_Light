#!/usr/bin/env python3
"""
锁步管线测试脚本 - Dummy Simulation
用于验证与 pruned policy 的数据交互和频率同步

功能：
1. 模拟机器人状态发布（DDS LowState）
2. 接收策略控制命令（DDS LowCmd）
3. 监控数据交互频率和质量
4. 验证 PD target 格式正确性
"""

import argparse
import time
import numpy as np
from typing import Optional, Dict, Any
import json

# DDS 相关导入
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_, unitree_hg_msg_dds__LowCmd_


class DummyG1State:
    """模拟 G1 机器人状态"""
    
    def __init__(self):
        self.time = 0.0
        self.dt = 0.01
        
        # 模拟关节状态 (29 个关节)
        self.num_joints = 29
        self.q = np.zeros(self.num_joints)
        self.dq = np.zeros(self.num_joints)
        self.tau = np.zeros(self.num_joints)
        
        # 设置初始站立姿态
        self._setup_standing_pose()
        
        # 添加噪声参数
        self.noise_scale = 0.01
        
    def _setup_standing_pose(self):
        """设置站立姿态"""
        # G1 站立关节角度（简化版本）
        standing_angles = [
            # 左腿 (6 DOF)
            0.0,    # 左髋俯仰
            0.0,    # 左髋滚转  
            -0.3,   # 左髋偏航
            0.6,    # 左膝
            -0.3,   # 左踝俯仰
            0.0,    # 左踝滚转
            # 右腿 (6 DOF)
            0.0,    # 右髋俯仰
            0.0,    # 右髋滚转
            -0.3,   # 右髋偏航
            0.6,    # 右膝
            -0.3,   # 右踝俯仰
            0.0,    # 右踝滚转
            # 腰部 (3 DOF)
            0.0, 0.0, 0.0,
            # 左臂 (7 DOF)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # 右臂 (7 DOF)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        
        for i, angle in enumerate(standing_angles):
            if i < self.num_joints:
                self.q[i] = angle
    
    def step(self, control_targets: Optional[np.ndarray] = None):
        """更新状态"""
        self.time += self.dt
        
        # 模拟关节动力学 (简单的PD控制)
        if control_targets is not None and len(control_targets) >= self.num_joints:
            # 简单的PD控制器
            kp = 100.0
            kd = 10.0
            
            for i in range(self.num_joints):
                target = control_targets[i]
                error = target - self.q[i]
                derror = -self.dq[i]
                
                # 计算力矩
                tau = kp * error + kd * derror
                self.tau[i] = tau
                
                # 简单积分（假设单位质量）
                ddq = tau * 0.1  # 简化的动力学
                self.dq[i] += ddq * self.dt
                self.q[i] += self.dq[i] * self.dt
        
        # 添加噪声
        noise = np.random.normal(0, self.noise_scale, self.num_joints)
        self.q += noise * 0.1
        self.dq += noise


class LockstepPipelineTester:
    """锁步管线测试器"""
    
    def __init__(self, test_frequency: float = 100.0):
        """初始化测试器"""
        print(f"🧪 初始化锁步管线测试器")
        print(f"   测试频率: {test_frequency} Hz")
        
        self.test_frequency = test_frequency
        
        # 初始化模拟状态
        self.dummy_robot = DummyG1State()
        
        # 设置 DDS 通信
        print("📡 设置 DDS 通信...")
        self._setup_dds()
        
        # 统计变量
        self.cycle_count = 0
        self.successful_exchanges = 0
        self.timeout_count = 0
        self.invalid_cmd_count = 0
        
        # 时间记录
        self.exchange_times = []
        self.last_exchange_time = 0.0
        
        print("✅ 锁步管线测试器初始化完成")
    
    def _setup_dds(self):
        """设置 DDS 通信"""
        # 初始化 DDS，增加队列大小
        ChannelFactoryInitialize(1, "lo")
        
        # 状态发布者，配置更大的队列
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.low_state_pub = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_pub.Init()
        
        # 控制命令订阅者，增加队列大小
        self.received_cmd = None
        self.cmd_receive_time = 0.0
        self.low_cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_sub.Init(self._cmd_handler, 100)  # 增加队列大小到 100
        
        print("✅ DDS 通信设置完成")
    
    def _cmd_handler(self, msg: LowCmd_):
        """处理接收到的控制命令"""
        self.received_cmd = msg
        self.cmd_receive_time = time.time()
    
    def publish_dummy_state(self):
        """发布模拟机器人状态"""
        try:
            # 更新状态时间戳
            self.low_state.tick = int(self.dummy_robot.time * 1000)  # 毫秒
            
            # 填充关节状态
            # 处理 motor_state 可能是固定大小数组的情况
            motor_state_size = 30  # G1 通常有 29-30 个关节
            for i in range(min(self.dummy_robot.num_joints, motor_state_size)):
                try:
                    self.low_state.motor_state[i].q = float(self.dummy_robot.q[i])
                    self.low_state.motor_state[i].dq = float(self.dummy_robot.dq[i])
                    self.low_state.motor_state[i].tau_est = float(self.dummy_robot.tau[i])
                    
                    # 设置模式和温度（模拟值）
                    self.low_state.motor_state[i].mode = 1  # 位置模式
                    self.low_state.motor_state[i].temperature = 35.0 + np.random.normal(0, 2.0)
                except (IndexError, AttributeError) as e:
                    print(f"⚠️ 关节 {i} 状态设置失败: {e}")
                    break
            
            # 模拟 IMU 数据
            try:
                # 设置四元数 (w, x, y, z)
                for i in range(4):  # 四元数
                    self.low_state.imu_state.quaternion[i] = 0.0
                self.low_state.imu_state.quaternion[0] = 1.0  # w=1, x=y=z=0
                
                # 设置角速度和加速度
                for i in range(3):  # 角速度和加速度
                    self.low_state.imu_state.gyroscope[i] = np.random.normal(0, 0.01)
                    self.low_state.imu_state.accelerometer[i] = np.random.normal(0, 0.1)
                self.low_state.imu_state.accelerometer[2] += 9.81  # 重力
            except (IndexError, AttributeError) as e:
                print(f"⚠️ IMU 数据设置失败: {e}")
                # 如果 IMU 设置失败，至少确保基本的 tick 设置成功
            
            # 发布状态
            self.low_state_pub.Write(self.low_state)
            return True
            
        except Exception as e:
            print(f"❌ 状态发布错误: {e}")
            return False
    
    def wait_for_control_cmd(self, timeout_ms: float = 50.0) -> Optional[np.ndarray]:
        """等待控制命令"""
        start_time = time.time()
        timeout_sec = timeout_ms / 1000.0
        
        # 清除之前的命令
        self.received_cmd = None
        
        while (time.time() - start_time) < timeout_sec:
            if self.received_cmd is not None:
                # 提取控制目标
                control_targets = np.zeros(self.dummy_robot.num_joints)
                
                for i in range(min(self.dummy_robot.num_joints, len(self.received_cmd.motor_cmd))):
                    # 假设接收到的是位置目标
                    control_targets[i] = self.received_cmd.motor_cmd[i].q
                
                return control_targets
            
            time.sleep(0.001)  # 1ms 轮询间隔
        
        return None
    
    def validate_control_cmd(self, targets: np.ndarray) -> Dict[str, Any]:
        """验证控制命令格式和内容"""
        validation = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        # 检查数组长度
        if len(targets) != self.dummy_robot.num_joints:
            validation['valid'] = False
            validation['issues'].append(f"关节数量不匹配: 期望{self.dummy_robot.num_joints}, 实际{len(targets)}")
        
        # 检查数值范围（简单检查）
        for i, target in enumerate(targets):
            if np.isnan(target) or np.isinf(target):
                validation['valid'] = False
                validation['issues'].append(f"关节{i}值无效: {target}")
            elif abs(target) > 10.0:  # 简单的范围检查
                validation['issues'].append(f"关节{i}值可能过大: {target}")
        
        # 统计信息
        validation['stats'] = {
            'mean': np.mean(targets),
            'std': np.std(targets),
            'min': np.min(targets),
            'max': np.max(targets),
            'range': np.max(targets) - np.min(targets)
        }
        
        return validation
    
    def run_lockstep_test(self, duration: float = 30.0):
        """运行锁步测试"""
        print(f"🚀 开始锁步管线测试 (时长: {duration}s)")
        print("💡 测试流程:")
        print("   1. 发布模拟机器人状态")
        print("   2. 等待策略控制命令")
        print("   3. 验证命令格式和内容")
        print("   4. 更新机器人状态")
        print("   5. 监控频率和性能")
        
        start_time = time.time()
        test_timeout = 1.0 / self.test_frequency * 2 * 1000  # 2倍周期作为超时 (ms)
        
        try:
            while time.time() - start_time < duration:
                cycle_start = time.time()
                self.cycle_count += 1
                
                print(f"\n🔄 === 测试周期 #{self.cycle_count} ===")
                
                # 步骤 1: 发布状态
                print("📤 发布模拟状态...")
                if not self.publish_dummy_state():
                    print("❌ 状态发布失败")
                    continue
                
                # 步骤 2: 等待控制命令
                print(f"⏳ 等待控制命令 (超时: {test_timeout:.0f}ms)...")
                control_targets = self.wait_for_control_cmd(test_timeout)
                
                if control_targets is None:
                    print(f"⚠️ 控制命令接收超时")
                    self.timeout_count += 1
                    continue
                
                print(f"✅ 接收到控制命令")
                exchange_time = time.time() - cycle_start
                self.exchange_times.append(exchange_time)
                
                # 步骤 3: 验证命令
                validation = self.validate_control_cmd(control_targets)
                if not validation['valid']:
                    print(f"❌ 控制命令验证失败:")
                    for issue in validation['issues']:
                        print(f"   - {issue}")
                    self.invalid_cmd_count += 1
                else:
                    print(f"✅ 控制命令验证通过")
                    stats = validation['stats']
                    print(f"   目标统计: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
                
                # 步骤 4: 更新机器人状态
                self.dummy_robot.step(control_targets)
                self.successful_exchanges += 1
                
                # 步骤 5: 频率控制
                cycle_elapsed = time.time() - cycle_start
                target_cycle_time = 1.0 / self.test_frequency
                
                print(f"⏱️ 周期耗时: {cycle_elapsed*1000:.1f}ms (目标: {target_cycle_time*1000:.1f}ms)")
                
                sleep_time = target_cycle_time - cycle_elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
                elif sleep_time < -0.005:
                    print(f"⚠️ 周期延迟: {-sleep_time*1000:.1f}ms")
                
                # 定期打印统计
                if self.cycle_count % 10 == 0:
                    self.print_test_stats()
                        
        except KeyboardInterrupt:
            print("\n🛑 收到中断信号，停止测试...")
        
        finally:
            self.print_final_stats()
    
    def print_test_stats(self):
        """打印测试统计"""
        if len(self.exchange_times) > 0:
            mean_time = np.mean(self.exchange_times) * 1000
            std_time = np.std(self.exchange_times) * 1000
            success_rate = self.successful_exchanges / self.cycle_count * 100
            
            print(f"\n📊 === 测试统计 (周期 #{self.cycle_count}) ===")
            print(f"💯 成功率: {success_rate:.1f}% ({self.successful_exchanges}/{self.cycle_count})")
            print(f"⏱️ 交换时间: {mean_time:.1f}±{std_time:.1f}ms")
            print(f"⚠️ 超时次数: {self.timeout_count}")
            print(f"❌ 无效命令: {self.invalid_cmd_count}")
            
            if len(self.exchange_times) >= 2:
                frequencies = []
                for i in range(1, len(self.exchange_times)):
                    dt = self.exchange_times[i] - self.exchange_times[i-1]
                    if dt > 0:
                        frequencies.append(1.0 / dt)
                
                if frequencies:
                    mean_freq = np.mean(frequencies)
                    print(f"📈 实际频率: {mean_freq:.1f} Hz (目标: {self.test_frequency:.1f} Hz)")
    
    def print_final_stats(self):
        """打印最终统计"""
        print(f"\n🏁 === 锁步管线测试最终报告 ===")
        print(f"📊 总体统计:")
        print(f"   总周期数: {self.cycle_count}")
        print(f"   成功交换: {self.successful_exchanges}")
        print(f"   超时次数: {self.timeout_count}")
        print(f"   无效命令: {self.invalid_cmd_count}")
        
        if self.cycle_count > 0:
            success_rate = self.successful_exchanges / self.cycle_count * 100
            print(f"   成功率: {success_rate:.1f}%")
        
        if len(self.exchange_times) > 0:
            mean_time = np.mean(self.exchange_times) * 1000
            std_time = np.std(self.exchange_times) * 1000
            print(f"   平均交换时间: {mean_time:.1f}±{std_time:.1f}ms")
            
            min_time = np.min(self.exchange_times) * 1000
            max_time = np.max(self.exchange_times) * 1000
            print(f"   交换时间范围: [{min_time:.1f}, {max_time:.1f}]ms")
        
        print("✅ 锁步管线测试完成")


def main():
    parser = argparse.ArgumentParser(
        description="锁步管线测试脚本 - 验证与 pruned policy 的数据交互"
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=100.0,
        help="测试频率 (Hz)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="测试时长 (秒)"
    )

    args = parser.parse_args()

    print(f"🧪 启动锁步管线测试")
    print(f"请确保 pruned policy 正在运行并监听 DDS 通信")
    
    # 创建并运行测试
    tester = LockstepPipelineTester(test_frequency=args.frequency)
    tester.run_lockstep_test(duration=args.duration)


if __name__ == "__main__":
    main() 