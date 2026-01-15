import sys
import os
from pathlib import Path

# Dynamically insert the replica implementation's package root directory
# (.../KOCO-bench/KOCO-bench-en/domain_code_generation/smolagents/test_examples/smolcc/code)
# to the front of sys.path so that import smolcc points to this replica package
REPLICA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLICA_ROOT))

# Ensure current working directory is also the project root directory
# This is important for relative imports and file path resolution
os.chdir(str(REPLICA_ROOT))

# Print debug information (optional, helpful for troubleshooting)
print(f"Added to sys.path: {REPLICA_ROOT}")
print(f"Current working directory: {os.getcwd()}")
print(f"sys.path[0]: {sys.path[0]}")