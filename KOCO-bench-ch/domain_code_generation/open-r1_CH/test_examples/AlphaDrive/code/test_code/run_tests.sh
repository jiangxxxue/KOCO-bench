#!/bin/bash

# AlphaDrive测试运行脚本
# 运行所有核心函数的测试

echo "========================================="
echo "Running AlphaDrive Core Function Tests"
echo "========================================="
echo ""

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# 测试1: plan_speed_reward
echo "Test 1: Testing plan_speed_reward function..."
python test_plan_speed_reward.py
if [ $? -eq 0 ]; then
    echo "✓ plan_speed_reward tests passed"
else
    echo "✗ plan_speed_reward tests failed"
fi
echo ""

# 测试2: plan_path_reward
echo "Test 2: Testing plan_path_reward function..."
python test_plan_path_reward.py
if [ $? -eq 0 ]; then
    echo "✓ plan_path_reward tests passed"
else
    echo "✗ plan_path_reward tests failed"
fi
echo ""

# 测试3: plan_format_reward
echo "Test 3: Testing plan_format_reward function..."
python test_plan_format_reward.py
if [ $? -eq 0 ]; then
    echo "✓ plan_format_reward tests passed"
else
    echo "✗ plan_format_reward tests failed"
fi
echo ""

# 测试4: _get_per_token_logps
echo "Test 4: Testing _get_per_token_logps function..."
python test_get_per_token_logps.py
if [ $? -eq 0 ]; then
    echo "✓ _get_per_token_logps tests passed"
else
    echo "✗ _get_per_token_logps tests failed"
fi
echo ""

# 测试5: compute_loss
echo "Test 5: Testing compute_loss function..."
python test_compute_loss.py
if [ $? -eq 0 ]; then
    echo "✓ compute_loss tests passed"
else
    echo "✗ compute_loss tests failed"
fi
echo ""

echo "========================================="
echo "All tests completed!"
echo "========================================="
