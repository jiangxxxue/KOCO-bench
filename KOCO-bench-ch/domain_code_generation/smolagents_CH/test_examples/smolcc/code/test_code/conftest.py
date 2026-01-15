import sys
import os
from pathlib import Path

# 将副本实现的包根目录（.../KOCO-bench/KOCO-bench-ch/domain_code_generation/smolagents/test_examples/smolcc/code）
# 动态插入到 sys.path 的最前面，以便 import smolcc 指向该副本包
REPLICA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLICA_ROOT))

# 确保当前工作目录也是项目根目录
# 这对于相对导入和文件路径解析很重要
os.chdir(str(REPLICA_ROOT))

# 打印调试信息（可选，有助于排查问题）
print(f"Added to sys.path: {REPLICA_ROOT}")
print(f"Current working directory: {os.getcwd()}")
print(f"sys.path[0]: {sys.path[0]}")