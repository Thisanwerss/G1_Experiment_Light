#!/bin/bash
# 启动锁步管线测试 - 完整中间件支持
# 模拟完整的锁步控制管线测试

echo "🔒 启动锁步管线测试系统"
echo "========================="

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
    echo "  --with-policy   使用真实策略服务（需要外部启动）"
    echo "  --duration N    测试持续时间（秒，默认 60）"
    echo "  --frequency N   测试频率（Hz，默认 100）"
    echo "  --help          显示此帮助信息"
    echo ""
    echo "说明："
    echo "  此脚本启动完整的锁步测试环境，包括："
    echo "  1. 策略服务（真实/模拟）"
    echo "  2. ZeroMQ 中继器"
    echo "  3. SDK 桥接器"
    echo "  4. 锁步管线测试器"
    echo ""
    echo "测试模式："
    echo "  默认: 使用内置策略模拟器（快速测试）"
    echo "  --with-policy: 连接外部真实策略服务"
    exit 0
fi

# 解析参数
USE_REAL_POLICY=false
TEST_DURATION=60
TEST_FREQUENCY=100

for arg in "$@"; do
    case $arg in
        --with-policy)
            USE_REAL_POLICY=true
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
LOG_DIR="logs/lockstep_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo -e "${BLUE}📁 日志目录: $LOG_DIR${NC}"
echo ""

# 1. 策略服务设置
if [ "$USE_REAL_POLICY" == true ]; then
    echo -e "${YELLOW}🔍 检查真实策略服务 5555 端口...${NC}"
    if timeout 3 bash -c "</dev/tcp/localhost/5555" 2>/dev/null; then
        echo -e "${GREEN}✅ 真实策略服务在 5555 端口可连通${NC}"
    else
        echo -e "${RED}❌ 无法连通真实策略服务${NC}"
        echo -e "${YELLOW}   请先启动 run_policy_pruned.py${NC}"
        echo -e "${YELLOW}   或使用默认模式（不带 --with-policy）${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}🤖 启动锁步策略模拟器...${NC}"
    
    # 创建锁步策略模拟器
    cat > /tmp/lockstep_policy_sim.py << 'EOF'
import time
import zmq
import numpy as np
import json

print("🔒 锁步策略模拟器启动")

context = zmq.Context()
control_socket = context.socket(zmq.REP)
control_socket.bind("tcp://*:5555")

print("   端口: 5555")
print("   模式: 锁步同步")

cycle_count = 0

while True:
    try:
        # 接收状态请求
        message = control_socket.recv_json(zmq.NOBLOCK)
        cycle_count += 1
        
        # 模拟策略计算延迟
        time.sleep(0.001)  # 1ms 计算时间
        
        # 生成控制目标（模拟站立控制）
        standing_pose = [
            # 左腿 (6 DOF)
            0.0, 0.0, -0.3, 0.6, -0.3, 0.0,
            # 右腿 (6 DOF)  
            0.0, 0.0, -0.3, 0.6, -0.3, 0.0,
            # 腰部 (3 DOF)
            0.0, 0.0, 0.0,
            # 左臂 (7 DOF)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # 右臂 (7 DOF)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        
        # 添加小幅随机扰动
        control_targets = [pos + np.random.normal(0, 0.01) for pos in standing_pose]
        
        response = {
            'control_targets': control_targets,
            'cycle_id': message.get('cycle_id', cycle_count),
            'timestamp': time.time()
        }
        
        control_socket.send_json(response)
        
        if cycle_count % 100 == 0:
            print(f"🔄 处理周期 #{cycle_count}")
        
    except zmq.Again:
        time.sleep(0.001)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ 策略模拟器错误: {e}")
        time.sleep(0.01)

control_socket.close()
context.term()
print("🛑 锁步策略模拟器停止")
EOF
    
    python /tmp/lockstep_policy_sim.py > $LOG_DIR/policy_sim.log 2>&1 &
    POLICY_SIM_PID=$!
    echo -e "${GREEN}   策略模拟器 PID: $POLICY_SIM_PID${NC}"
    sleep 2
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
sleep 3  # 等待桥接器完全初始化

echo ""
echo -e "${GREEN}✅ 中间件启动完成！${NC}"
echo ""
echo "🔗 锁步测试通信链路："
echo "   test_lockstep_pipeline.py (Dummy G1)"
echo "           ↓ (DDS LowState 发布)"
echo "   run_policy_sdk_bridge.py"
echo "           ↓ (ZeroMQ 状态/控制)"
echo "   policy_zmq_relay.py"
echo "           ↓ (ZeroMQ REQ/REP)"
if [ "$USE_REAL_POLICY" == true ]; then
    echo "   run_policy_pruned.py (真实策略)"
else
    echo "   锁步策略模拟器"
fi
echo "           ↓ (ZeroMQ 控制命令)"
echo "   ... (反向路径) ..."
echo "           ↓ (DDS LowCmd 接收)"
echo "   test_lockstep_pipeline.py (验证)"
echo ""

# 4. 启动锁步管线测试器
echo -e "${YELLOW}🔒 启动锁步管线测试器...${NC}"
echo -e "${BLUE}   测试参数: 频率=${TEST_FREQUENCY}Hz, 持续=${TEST_DURATION}s${NC}"
if [ "$USE_REAL_POLICY" == true ]; then
    echo -e "${GREEN}   策略模式: 真实策略服务${NC}"
else
    echo -e "${YELLOW}   策略模式: 内置模拟器${NC}"
fi
echo ""

# 启动锁步管线测试
python3 debug_toolbox/test_lockstep_pipeline.py \
    --duration $TEST_DURATION \
    --frequency $TEST_FREQUENCY \
    2>&1 | tee $LOG_DIR/lockstep_test.log

echo ""
echo -e "${BLUE}📋 测试完成！检查日志文件:${NC}"
echo "   锁步测试: $LOG_DIR/lockstep_test.log"
echo "   ZeroMQ 中继: $LOG_DIR/zmq_relay.log"
echo "   SDK 桥接: $LOG_DIR/sdk_bridge.log"
if [ "$USE_REAL_POLICY" == false ] && [ -n "$POLICY_SIM_PID" ]; then
    echo "   策略模拟: $LOG_DIR/policy_sim.log"
fi

echo ""
echo -e "${YELLOW}📊 查看详细日志:${NC}"
echo "   tail -f $LOG_DIR/*.log"

# 清理
cleanup 