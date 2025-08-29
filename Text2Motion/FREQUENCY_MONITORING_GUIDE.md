# 🔍 run_policy.py 频率监控功能使用指南

## 📖 功能概述

修改后的 `run_policy.py` 添加了专门的频率监控系统，用于深入分析：
- **系统循环频率** - 完整控制循环的运行频率
- **Knots更新频率** - 样条结点计算的更新频率  
- **PD目标更新频率** - 关节控制目标的设置频率
- **仿真步骤频率** - MuJoCo物理仿真的实际频率

## 🚀 快速开始

### 1. 运行基础监控
```bash
# 非Hybrid模式 (纯NN推理)
python scripts/run_policy.py --frequency 30.0

# Hybrid模式 (NN + CEM优化)
python scripts/run_policy.py --hybrid --frequency 20.0 --num_samples 200
```

### 2. 观察输出
程序会每20个循环输出详细的频率报告：

```
🔍 === 频率监控报告 ===
📊 瞬时频率 (最近20个样本):
   🔄 系统循环频率: 18.45 ± 0.23 Hz
   🧠 Knots更新频率: 18.45 ± 0.23 Hz
   🎯 PD目标更新频率: 36.90 ± 0.46 Hz
   ⚙️  仿真步骤频率: 36.90 ± 0.46 Hz

🔗 解耦分析:
   ✅ 仿真与控制耦合良好: 1.00
   ✅ 系统循环与Knots更新同步
```

## 📊 关键指标解读

### 🎯 正常耦合的指标
- **系统循环频率 ≈ Knots更新频率** - 每次循环都更新控制策略
- **PD更新频率 ≈ 系统循环频率 × 2** - 每个控制周期包含2个仿真步
- **仿真步骤频率 ≈ PD更新频率** - 仿真与控制同步
- **频率达成率 > 80%** - 接近目标性能

### ⚠️ 异常情况
- **系统循环频率 << 目标频率** - 计算性能瓶颈
- **频率比例异常** - 存在逻辑错误或解耦问题
- **高频率标准差** - 性能不稳定

## 🔧 测试不同场景

### 场景1: 性能基准测试
```bash
# 测试纯NN性能
python scripts/run_policy.py --frequency 50.0

# 期望结果: 系统频率 ~40-50Hz
```

### 场景2: CEM性能测试  
```bash
# 测试CEM影响
python scripts/run_policy.py --hybrid --frequency 20.0 --num_samples 500

# 期望结果: 系统频率 ~5-15Hz (受CEM拖累)
```

### 场景3: 优化CEM测试
```bash
# 优化CEM参数
python scripts/run_policy.py --hybrid --frequency 30.0 --num_samples 200 --plan_horizon 0.3

# 期望结果: 频率有所提升
```

## 🛠️ 自动化测试

### 运行测试套件
```bash
python test_frequency_monitoring.py
```

### 查看分析指南  
```bash
python analyze_frequency_results.py
```

## 📈 性能优化建议

### 根据监控结果优化

1. **如果系统频率 < 目标频率的50%**:
   - 减少CEM samples: `--num_samples 200`
   - 缩短planning horizon: `--plan_horizon 0.3`
   - 降低目标频率: `--frequency 20.0`

2. **如果频率比例异常**:
   - 检查是否有逻辑错误
   - 验证sim_steps_per_replan计算
   - 确认时间控制逻辑

3. **如果频率不稳定**:
   - 检查GPU内存是否充足
   - 验证JAX JIT编译是否完成
   - 减少系统负载

## 🔍 解耦验证检查表

运行监控后，检查以下指标：

- [ ] 系统循环频率合理 (> 目标频率的50%)
- [ ] Knots更新 = 系统循环 (±5%)  
- [ ] PD更新 = 系统循环 × 2 (±10%)
- [ ] 仿真步骤 = PD更新 (±5%)
- [ ] 频率标准差小 (< 平均值的20%)

## 🎯 预期结果

### 非Hybrid模式
```
目标频率: 30Hz
系统循环: ~25-30Hz (83-100%达成率)
PD更新: ~50-60Hz  
仿真步骤: ~50-60Hz
解读: NN推理快，接近目标性能
```

### Hybrid模式  
```
目标频率: 20Hz  
系统循环: ~10-15Hz (50-75%达成率)
PD更新: ~20-30Hz
仿真步骤: ~20-30Hz  
解读: CEM拖累性能，但比例关系正常
```

## 🚨 常见问题

### Q: 为什么PD更新频率是系统频率的2倍？
A: 每个系统循环包含 `sim_steps_per_replan = 2` 个仿真步，每步都设置新的PD目标。

### Q: 仿真频率异常低怎么办？
A: 检查是否有额外的计算开销，确认viewer.sync()没有过度拖累性能。

### Q: 频率达成率很低怎么优化？
A: 按优先级：降低CEM复杂度 > 降低目标频率 > 优化代码实现。

## 📚 相关文档

- [AOT预编译优化指南](AOT_USAGE_GUIDE.md)
- [CEM参数调优指南](hydrax/algs/cem.py)
- [异步仿真架构](hydrax/simulation/asynchronous.py) 