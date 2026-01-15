#!/bin/bash
# ByteMLPerf 测试运行脚本

echo "ByteMLPerf 测试套件"
echo "==================="

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python 未安装"
    exit 1
fi

# 检查测试文件是否存在
if [ ! -f "test_loss_balancer.py" ]; then
    echo "❌ 测试文件 test_loss_balancer.py 不存在"
    exit 1
fi

if [ ! -f "test_mcore_distillation_adjust.py" ]; then
    echo "❌ 测试文件 test_mcore_distillation_adjust.py 不存在"
    exit 1
fi

# 运行测试
echo "运行损失平衡器测试..."
python test_loss_balancer.py

echo "运行MCore适配测试..."
python test_mcore_distillation_adjust.py

echo "所有测试完成！"