#!/bin/bash
# 简化的 G1 DDS 通信测试启动脚本 - 使用完整MuJoCo仿真
# 架构: 完整MuJoCo仿真 ←→ 简化桥接器 ←→ 策略服务

echo "🚀 启动简化 G1 DDS 通信测试 (完整MuJoCo仿真)"
echo "======================================================"

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
    echo "  --duration N    测试持续时间（秒，默认 30）"
    echo "  --frequency N   控制频率（Hz，默认 50）"
    echo "  --freerun       自由运行模式（不等待控制命令）"
    echo "  --no_viewer     禁用MuJoCo查看器"
    echo "  --help          显示此帮助信息"
    echo ""
    echo "简化架构（完整仿真版）："
    echo "  完整MuJoCo仿真 ←→ 简化桥接器 ←→ 策略服务"
    echo "  (3个进程，直接DDS通信，带可视化)"
    exit 0
fi

# 解析参数
TEST_DURATION=30
TEST_FREQUENCY=50
FREERUN_MODE=""
NO_VIEWER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            TEST_DURATION="$2"
            shift 2
            ;;
        --frequency)
            TEST_FREQUENCY="$2"
            shift 2
            ;;
        --freerun)
            FREERUN_MODE="--freerun"
            shift
            ;;
        --no_viewer)
            NO_VIEWER="--no_viewer"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# 返回主目录
cd "$(dirname "$0")"

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
LOG_DIR="logs/simple_g1_mujoco_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo -e "${BLUE}📁 日志目录: $LOG_DIR${NC}"
echo ""

echo -e "${BLUE}🏗️ 完整仿真架构说明:${NC}"
echo "  1. 完整 MuJoCo G1 仿真 - 带可视化界面"
echo "  2. 简化桥接器 - DDS ↔ ZMQ 转换"
echo "  3. 策略服务 - NN+CEM 控制计算"
echo ""
echo -e "${YELLOW}🔧 测试参数:${NC}"
echo "  持续时间: ${TEST_DURATION}s"
echo "  控制频率: ${TEST_FREQUENCY}Hz"
echo "  仿真模式: $([ -n "$FREERUN_MODE" ] && echo '自由运行' || echo '锁步')"
echo "  可视化: $([ -n "$NO_VIEWER" ] && echo '禁用' || echo '启用')"
echo ""

# 1. 启动策略服务
echo -e "${YELLOW}🧠 启动策略计算服务...${NC}"
python3 run_policy_pruned.py \
    --frequency $TEST_FREQUENCY \
    --zmq_state_port 5555 \
    --zmq_ctrl_port 5556 \
    > $LOG_DIR/policy.log 2>&1 &
POLICY_PID=$!
echo -e "${GREEN}   策略服务 PID: $POLICY_PID${NC}"

# 等待策略服务初始化
echo -e "${BLUE}   等待策略服务初始化...${NC}"
sleep 5

# 检查策略服务是否正常运行
if ! kill -0 $POLICY_PID 2>/dev/null; then
    echo -e "${RED}❌ 策略服务启动失败${NC}"
    echo -e "${YELLOW}检查日志: $LOG_DIR/policy.log${NC}"
    cleanup
fi

echo -e "${GREEN}✅ 策略服务运行正常${NC}"

# 2. 启动简化桥接器
echo -e "${YELLOW}🌉 启动简化桥接器...${NC}"
python3 run_simple_g1_policy_bridge.py \
    --simulate \
    --frequency $TEST_FREQUENCY \
    > $LOG_DIR/bridge.log 2>&1 &
BRIDGE_PID=$!
echo -e "${GREEN}   桥接器 PID: $BRIDGE_PID${NC}"

# 等待桥接器初始化
echo -e "${BLUE}   等待桥接器初始化...${NC}"
sleep 3

# 检查桥接器是否正常运行
if ! kill -0 $BRIDGE_PID 2>/dev/null; then
    echo -e "${RED}❌ 桥接器启动失败${NC}"
    echo -e "${YELLOW}检查日志: $LOG_DIR/bridge.log${NC}"
    cleanup
fi

echo -e "${GREEN}✅ 桥接器运行正常${NC}"

# 3. 启动完整MuJoCo仿真
echo -e "${YELLOW}🤖 启动完整 G1 MuJoCo 仿真...${NC}"
python3 run_g1_mujoco_dds_sim.py \
    --frequency $TEST_FREQUENCY \
    --duration $TEST_DURATION \
    $FREERUN_MODE \
    $NO_VIEWER \
    2>&1 | tee $LOG_DIR/simulation.log &
SIM_PID=$!
echo -e "${GREEN}   MuJoCo 仿真 PID: $SIM_PID${NC}"

echo ""
echo -e "${GREEN}✅ 完整测试系统启动完成！${NC}"
echo ""
echo -e "${BLUE}🔗 通信链路:${NC}"
echo "   完整MuJoCo仿真 ←→ DDS ←→ 简化桥接器 ←→ ZMQ ←→ 策略服务"
echo ""
echo -e "${YELLOW}📊 实时监控:${NC}"
echo "   策略日志: tail -f $LOG_DIR/policy.log"
echo "   桥接器日志: tail -f $LOG_DIR/bridge.log"  
echo "   仿真日志: tail -f $LOG_DIR/simulation.log"
echo ""

if [ -z "$NO_VIEWER" ]; then
    echo -e "${BLUE}👁️ MuJoCo 可视化界面应该已启动${NC}"
    echo -e "${YELLOW}💡 提示: 关闭MuJoCo窗口将停止仿真${NC}"
fi

# 等待仿真完成
wait $SIM_PID

echo ""
echo -e "${BLUE}📋 测试完成！检查结果:${NC}"
echo "   策略日志: $LOG_DIR/policy.log"
echo "   桥接器日志: $LOG_DIR/bridge.log"
echo "   仿真日志: $LOG_DIR/simulation.log"

# 自动清理
cleanup 