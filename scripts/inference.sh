#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 项目根目录
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"


# 运行推理脚本
python scripts/inference_example.py \
    --config configs/causal_inference.yaml