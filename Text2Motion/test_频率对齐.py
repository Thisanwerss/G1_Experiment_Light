#!/usr/bin/env python3
"""
测试修复后的频率对齐机制
"""

import time
import numpy as np
import zmq

def test_frequency_alignment():
    """测试频率对齐"""
    print("🧪 测试频率对齐机制...")
    
    # 连接到控制端
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2秒超时
    
    target_frequency = 10.0  # 测试用低频率
    target_cycle_time = 1.0 / target_frequency
    
    print(f"📊 目标频率: {target_frequency} Hz")
    print(f"📊 目标周期时间: {target_cycle_time:.3f}s")
    
    cycle_times = []
    request_times = []
    
    try:
        for cycle in range(10):  # 测试10个循环
            cycle_start = time.time()
            
            print(f"📡 循环 #{cycle+1}: 发送请求...")
            
            # 创建测试请求
            request = {
                'qpos': np.random.normal(0, 0.1, 54),
                'qvel': np.random.normal(0, 0.1, 53),
                'time': cycle * target_cycle_time
            }
            
            # 发送请求并计时
            request_start = time.time()
            socket.send_pyobj(request)
            response = socket.recv_pyobj()
            request_end = time.time()
            
            request_time = request_end - request_start
            request_times.append(request_time)
            
            # 验证响应
            if 'controls' in response:
                controls = response['controls']
                print(f"✅ 收到控制命令: 形状={controls.shape}, 请求时间={request_time*1000:.1f}ms")
            else:
                print(f"❌ 无效响应")
                break
            
            # 严格的频率控制
            cycle_elapsed = time.time() - cycle_start
            if cycle_elapsed < target_cycle_time:
                sleep_time = target_cycle_time - cycle_elapsed
                time.sleep(sleep_time)
            
            actual_cycle_time = time.time() - cycle_start
            cycle_times.append(actual_cycle_time)
            
            print(f"⏱️  循环时间: {actual_cycle_time:.3f}s (目标: {target_cycle_time:.3f}s)")
        
        # 分析结果
        if len(cycle_times) > 1:
            avg_cycle_time = np.mean(cycle_times)
            avg_frequency = 1.0 / avg_cycle_time
            frequency_error = abs(avg_frequency - target_frequency) / target_frequency * 100
            
            avg_request_time = np.mean(request_times)
            max_request_time = np.max(request_times)
            
            print(f"\n📊 === 频率对齐分析 ===")
            print(f"平均周期时间: {avg_cycle_time:.3f}s")
            print(f"实际频率: {avg_frequency:.2f} Hz")
            print(f"频率误差: {frequency_error:.1f}%")
            print(f"平均请求时间: {avg_request_time*1000:.1f}ms")
            print(f"最大请求时间: {max_request_time*1000:.1f}ms")
            
            if frequency_error < 5.0:
                print("✅ 频率对齐测试通过")
                return True
            else:
                print("❌ 频率误差过大")
                return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    finally:
        socket.close()
        context.term()

def main():
    """主测试函数"""
    print("🚀 开始频率对齐测试...")
    print("请确保 run_policy_pruned.py 已经在运行")
    
    success = test_frequency_alignment()
    
    if success:
        print("\n🎉 频率对齐测试成功！")
        print("现在可以运行完整的 isolated_simulation.py")
    else:
        print("\n❌ 频率对齐测试失败")
    
    return success

if __name__ == "__main__":
    main() 