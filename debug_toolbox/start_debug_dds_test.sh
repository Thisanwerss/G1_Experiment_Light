#!/bin/bash
# 启动 DDS 通信调试测试 - 完整中间件支持
# 确保端到端 DDS 通信测试的真实性

echo "🧪 启动 DDS 通信调试测试系统"
echo "================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查参数
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --no-policy     不检查策略服务（仅测试 DDS 发布）"
    echo "  --duration N    测试持续时间（秒，默认 30）"
    echo "  --frequency N   测试频率（Hz，默认 5）"
    echo "  --help          显示此帮助信息"
    echo ""
    echo "说明："
    echo "  此脚本启动完整的 DDS 测试环境，包括："
    echo "  1. 策略服务检查/启动"
    echo "  2. ZeroMQ 中继器"
    echo "  3. SDK 桥接器"
    echo "  4. DDS 通信调试器"
    exit 0
fi

# 解析参数
CHECK_POLICY=true
TEST_DURATION=30
TEST_FREQUENCY=5

for arg in "$@"; do
    case $arg in
        --no-policy)
            CHECK_POLICY=false
            shift
            ;;
        --duration)
            TEST_DURATION="$2"
            shift 2
            ;;
        --frequency)
            TEST_FREQUENCY="$2"
            shift 2
            ;;
    esac
done

# 返回主目录
cd "$(dirname "$0")/.."

# 终止函数
cleanup() {
    echo -e "\n${RED}🛑 停止所有测试进程...${NC}"
    
    # 终止所有子进程
    jobs -p | xargs -r kill 2>/dev/null
    
    # 等待进程结束
    sleep 2
    
    # 强制终止残留进程
    jobs -p | xargs -r kill -9 2>/dev/null
    
    echo -e "${GREEN}✅ 清理完成${NC}"
    exit 0
}

# 捕获中断信号
trap cleanup SIGINT SIGTERM

# 创建日志目录
LOG_DIR="logs/debug_dds_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo -e "${BLUE}📁 日志目录: $LOG_DIR${NC}"
echo ""

# 1. 检查策略服务（如果需要）
if [ "$CHECK_POLICY" == true ]; then
    echo -e "${YELLOW}🔍 检查策略服务 5555 端口...${NC}"
    if timeout 2 bash -c "</dev/tcp/localhost/5555" 2>/dev/null; then
        echo -e "${GREEN}✅ 策略服务在 5555 端口可连通${NC}"
    else
        echo -e "${RED}❌ 无法连通 5555 端口${NC}"
        echo -e "${YELLOW}   启动内置 DDS 模拟策略服务...${NC}"
        
        # 启动简化的策略模拟器
        cat > /tmp/dds_policy_sim.py << 'EOF'
import time
import zmq
import numpy as np
import pickle
import struct

context = zmq.Context()

# 策略接收 socket (PULL) - 接收状态
policy_recv_socket = context.socket(zmq.PULL)
policy_recv_socket.bind("tcp://*:5556")

# 策略发送 socket (PUSH) - 发送控制命令  
policy_send_socket = context.socket(zmq.PUSH)
policy_send_socket.bind("tcp://*:5555")

print("🤖 DDS 策略模拟器启动")
print("   策略接收端口: 5556 (接收状态)")
print("   策略发送端口: 5555 (发送控制)")

# 设置非阻塞接收
policy_recv_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms 超时

cycle_count = 0
while True:
    try:
        # 尝试接收状态
        try:
            parts = policy_recv_socket.recv_multipart(zmq.NOBLOCK)
            if len(parts) == 2:
                cycle_id_bytes, state_bytes = parts
                cycle_id = struct.unpack('I', cycle_id_bytes)[0]
                state = pickle.loads(state_bytes)
                
                # 生成随机控制命令
                control_targets = np.random.normal(0, 0.05, 41).tolist()
                
                # 构造控制响应
                response = {
                    'controls': np.random.normal(0, 0.05, (1, 41)),  # 1x41 array
                    'timing': {'total_time': 0.02}
                }
                
                # 发送响应
                response_bytes = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
                policy_send_socket.send_multipart([cycle_id_bytes, response_bytes], zmq.NOBLOCK)
                
                cycle_count += 1
                if cycle_count % 10 == 0:
                    print(f"🔄 策略模拟器已处理 {cycle_count} 个周期")
                    
        except zmq.Again:
            # 没有消息，继续
            pass
            
        time.sleep(0.01)  # 10ms 休眠
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ 策略模拟器错误: {e}")
        time.sleep(0.1)

policy_recv_socket.close()
policy_send_socket.close()
context.term()
print("🛑 策略模拟器停止")
EOF
        
        python /tmp/dds_policy_sim.py > $LOG_DIR/policy_sim.log 2>&1 &
        POLICY_SIM_PID=$!
        echo -e "${GREEN}   策略模拟器 PID: $POLICY_SIM_PID${NC}"
        sleep 2
    fi
fi

# 2. 启动策略 ZeroMQ 中继器
echo -e "${YELLOW}🔄 启动策略 ZeroMQ 中继器...${NC}"
python3 policy_zmq_relay.py > $LOG_DIR/zmq_relay.log 2>&1 &
RELAY_PID=$!
echo -e "${GREEN}   中继器 PID: $RELAY_PID${NC}"
sleep 1

# 3. 启动 SDK 桥接控制器
echo -e "${YELLOW}🌉 启动 SDK 桥接控制器 (仿真模式)...${NC}"
python3 run_policy_sdk_bridge.py > $LOG_DIR/sdk_bridge.log 2>&1 &
BRIDGE_PID=$!
echo -e "${GREEN}   桥接器 PID: $BRIDGE_PID${NC}"
sleep 2

echo ""
echo -e "${GREEN}✅ 中间件启动完成！${NC}"
echo ""
echo "🔗 测试通信链路："
echo "   debug_dds_communication.py"
echo "           ↓ (DDS 发布)"
echo "   run_policy_sdk_bridge.py"
echo "           ↓ (ZeroMQ)"
echo "   policy_zmq_relay.py"
echo "           ↓ (ZeroMQ)"
echo "   策略服务/模拟器"
echo ""

# 4. 启动 DDS 通信调试器
echo -e "${YELLOW}🔧 启动 DDS 通信调试器...${NC}"
echo -e "${BLUE}   测试参数: 频率=${TEST_FREQUENCY}Hz, 持续=${TEST_DURATION}s${NC}"
echo ""

# 修改调试脚本的测试参数
python3 debug_toolbox/debug_dds_communication.py \
    --duration $TEST_DURATION \
    --frequency $TEST_FREQUENCY \
    2>&1 | tee $LOG_DIR/dds_test.log

echo ""
echo -e "${BLUE}📋 测试完成！检查日志文件:${NC}"
echo "   DDS 测试: $LOG_DIR/dds_test.log"
echo "   ZeroMQ 中继: $LOG_DIR/zmq_relay.log"
echo "   SDK 桥接: $LOG_DIR/sdk_bridge.log"
if [ "$CHECK_POLICY" == true ] && [ -n "$POLICY_SIM_PID" ]; then
    echo "   策略模拟: $LOG_DIR/policy_sim.log"
fi

# 清理
cleanup 