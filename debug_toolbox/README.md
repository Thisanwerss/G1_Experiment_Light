# 调试工具箱 (Debug Toolbox)

本目录包含用于调试和测试 G1 机器人控制系统的工具。

## 🧪 测试工具

### 1. DDS 通信调试器
- **脚本**: `debug_dds_communication.py`
- **启动器**: `start_debug_dds_test.sh`
- **功能**: 测试基本的 DDS 通信发布/订阅

### 2. 锁步管线测试器
- **脚本**: `test_lockstep_pipeline.py`
- **启动器**: `start_lockstep_pipeline_test.sh`
- **功能**: 测试完整的锁步控制管线

## 🚀 快速开始

### DDS 通信测试
```bash
# 基本 DDS 通信测试（包含所有中间件）
./debug_toolbox/start_debug_dds_test.sh

# 自定义参数
./debug_toolbox/start_debug_dds_test.sh --duration 60 --frequency 10

# 仅测试 DDS 发布（不启动策略服务）
./debug_toolbox/start_debug_dds_test.sh --no-policy
```

### 锁步管线测试
```bash
# 使用内置策略模拟器（推荐用于快速测试）
./debug_toolbox/start_lockstep_pipeline_test.sh

# 使用真实策略服务（需要先启动 run_policy_pruned.py）
./debug_toolbox/start_lockstep_pipeline_test.sh --with-policy

# 自定义参数
./debug_toolbox/start_lockstep_pipeline_test.sh --duration 30 --frequency 50
```

## 📊 测试模式

### DDS 通信测试模式
1. **完整模式** (默认): 启动所有中间件，测试端到端通信
2. **仅发布模式** (`--no-policy`): 只测试 DDS 发布功能

### 锁步管线测试模式
1. **模拟器模式** (默认): 使用内置策略模拟器
2. **真实策略模式** (`--with-policy`): 连接外部策略服务

## 🔧 测试架构

### DDS 通信测试架构
```
debug_dds_communication.py
         ↓ (DDS LowState)
run_policy_sdk_bridge.py
         ↓ (ZeroMQ)
policy_zmq_relay.py
         ↓ (ZeroMQ)
策略服务/模拟器
```

### 锁步管线测试架构
```
test_lockstep_pipeline.py (Dummy G1)
         ↓ (DDS LowState 发布)
run_policy_sdk_bridge.py
         ↓ (ZeroMQ 状态/控制)
policy_zmq_relay.py
         ↓ (ZeroMQ REQ/REP)
策略服务/锁步模拟器
         ↓ (控制命令返回)
... (反向路径) ...
         ↓ (DDS LowCmd 接收)
test_lockstep_pipeline.py (验证)
```

## 📝 日志文件

所有测试的日志文件保存在 `logs/` 目录下：
- `logs/debug_dds_YYYYMMDD_HHMMSS/` - DDS 通信测试日志
- `logs/lockstep_test_YYYYMMDD_HHMMSS/` - 锁步管线测试日志

每个测试会生成以下日志：
- `*_test.log` - 主测试日志
- `zmq_relay.log` - ZeroMQ 中继器日志
- `sdk_bridge.log` - SDK 桥接器日志
- `policy_sim.log` - 策略模拟器日志（如果使用）

## 🔍 故障排除

### 常见问题

1. **DDS 初始化失败**
   ```
   selected interface "lo" is not multicast-capable
   ```
   - 这是正常警告，使用本地环回接口

2. **端口占用**
   ```
   Address already in use
   ```
   - 检查是否有其他进程占用端口
   - 使用 `pkill -f "python.*policy"` 清理残留进程

3. **策略服务连接失败**
   ```
   无法连通 5555 端口
   ```
   - 确保 `run_policy_pruned.py` 已启动并完成预热
   - 或使用内置模拟器模式

### 诊断步骤

1. **检查进程状态**
   ```bash
   ps aux | grep python
   netstat -tlnp | grep :555
   ```

2. **查看实时日志**
   ```bash
   tail -f logs/*/\*.log
   ```

3. **手动测试单个组件**
   ```bash
   # 仅测试 DDS 发布
   python debug_toolbox/debug_dds_communication.py --duration 10

   # 仅测试策略服务
   curl -X GET http://localhost:5555 || echo "策略服务未响应"
   ```

## ⚙️ 高级选项

### 自定义测试参数
- `--duration N`: 测试持续时间（秒）
- `--frequency N`: 测试频率（Hz）

### 环境变量
- `DDS_DOMAIN_ID`: DDS 域 ID（默认 1）
- `PYTHONUNBUFFERED`: 设置为 1 启用实时日志输出

### 性能调优
对于高频率测试，建议：
- 降低日志详细程度
- 增加 DDS 队列大小
- 调整系统调度优先级

## 📞 支持

如遇问题，请检查：
1. 系统依赖是否完整安装
2. Python 环境是否正确激活
3. 网络端口是否可用
4. 日志文件中的详细错误信息 