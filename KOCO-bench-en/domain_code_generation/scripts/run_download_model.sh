#!/bin/bash
# 快速下载模型脚本
# 用法: ./run_download_model.sh [模型名] [保存目录]
# 示例: ./run_download_model.sh Qwen/Qwen2.5-Coder-7B-Instruct ~/models

# 获取参数，如果没有提供则使用默认值
MODEL_NAME="${1:-Qwen/Qwen2.5-Coder-7B}"
SAVE_DIR="${2:-~/models}"

# 执行下载
python download_model.py "$MODEL_NAME" "$SAVE_DIR"