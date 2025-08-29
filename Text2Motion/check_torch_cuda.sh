#!/usr/bin/env bash
# check_torch_cuda.sh
# Usage: chmod +x check_torch_cuda.sh && ./check_torch_cuda.sh

set -e

echo "🖥️  === GPU / CUDA / PyTorch 环境检测 ==="

echo ""
echo "🔹 系统 GPU 驱动和显卡信息 (nvidia-smi)："
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "⚠️  nvidia-smi 未找到，可能不支持 NVIDIA GPU 或命令未安装"
fi

echo ""
echo "🔹 CUDA Toolkit 版本 (nvcc)："
if command -v nvcc &>/dev/null; then
  nvcc --version | grep "release"
else
  echo "⚠️  nvcc 未找到，CUDA Toolkit 可能未安装或未加入 PATH"
fi

echo ""
echo "🔹 Python & PyTorch 信息："
python3 - << 'PYCODE'
import torch, sys
print(f"Python: {sys.executable} ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")
print(f"torch.__version__: {torch.__version__}")
cuda_avail = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {cuda_avail}")
if cuda_avail:
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"CUDA devices count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
else:
    print("⚠️  CUDA 不可用，所有运算将跑在 CPU")

# 测试 torch.compile 和 Inductor 后端
try:
    import torch._inductor as inductor
    print("torch._inductor: available")
except ImportError:
    print("torch._inductor: NOT available")

# 简单测试 torch.compile
try:
    @torch.compile(fullgraph=True)
    def foo(x):
        return x * x + 2
    x = torch.randn(10, device='cuda' if cuda_avail else 'cpu')
    out = foo(x)
    print("torch.compile test: SUCCESS")
except Exception as e:
    print(f"torch.compile test: FAILED ({e})")
PYCODE

echo ""
echo "✅ 检测完成，请根据输出确认："
echo "   - PyTorch ≥2.0"
echo "   - torch.cuda.is_available() 为 True"
echo "   - torch.version.cuda 与 nvcc 列出的 CUDA 版本一致"
echo "   - torch._inductor 可用"
echo "   - torch.compile 可以正常运行"

