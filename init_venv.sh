#!/usr/bin/env bash
# init_venv.sh — activate your venv when you're in /home/atari/workspace

# 1) 确保你是在 workspace 下（可选校验）
if [ "$(basename "$PWD")" != "workspace" ]; then
  echo "⚠ 请先 cd 到 ~/workspace 再运行这个脚本" >&2
  exit 1
fi

# 2) 上移一级到家目录
cd ..

# 3) 激活虚拟环境
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  echo "❌ 找不到 .venv/bin/activate" >&2
  exit 1
fi

# 4) 回到 workspace
cd workspace

echo "✅ venv 已激活，当前路径：$PWD"
