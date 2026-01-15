#!/bin/bash

# CANOE测试运行脚本
# 运行所有核心函数的测试

echo "========================================="
echo "Running CANOE Core Function Tests"
echo "========================================="
echo ""

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../train/src"

# 运行各个测试文件
echo "1. Testing accuracy_reward function..."
python test_accuracy_reward.py
if [ $? -eq 0 ]; then
    echo "✓ accuracy_reward tests passed"
else
    echo "✗ accuracy_reward tests failed"
fi
echo ""

echo "2. Testing influence_reward function..."
python test_influence_reward.py
if [ $? -eq 0 ]; then
    echo "✓ influence_reward tests passed"
else
    echo "✗ influence_reward tests failed"
fi
echo ""

echo "3. Testing format_reward function..."
python test_format_reward.py
if [ $? -eq 0 ]; then
    echo "✓ format_reward tests passed"
else
    echo "✗ format_reward tests failed"
fi
echo ""

echo "4. Testing len_reward function..."
python test_len_reward.py
if [ $? -eq 0 ]; then
    echo "✓ len_reward tests passed"
else
    echo "✗ len_reward tests failed"
fi
echo ""

echo "========================================="
echo "All tests completed"
echo "========================================="
