import sys
import os
from pathlib import Path

# 1) 将本实例的 code 根目录加入 sys.path，并切换为当前工作目录
REPLICA_ROOT = Path(__file__).resolve().parents[1]  # .../ToolBrain/code
sys.path.insert(0, str(REPLICA_ROOT))
os.chdir(str(REPLICA_ROOT))

print(f"[ToolBrain tests] Added to sys.path (replica root): {REPLICA_ROOT}")
print(f"[ToolBrain tests] CWD: {os.getcwd()}")

# 2) 优先确保本地 peft 包可被发现（避免 transformers 可选依赖探测时报 peft.__spec__ is None）
#    我们在 code/peft/__init__.py 下提供了最小可用的 peft 包。
def _ensure_local_peft():
    try:
        import peft  # type: ignore
        # 若已加载但无 __spec__，显式重新导入以确保 spec 正确
        if getattr(peft, "__spec__", None) is None:
            print("[ToolBrain tests] peft present without __spec__, reloading...")
            del sys.modules["peft"]
            import importlib
            peft = importlib.import_module("peft")  # noqa: F401
        print("[ToolBrain tests] peft import OK (local stub)")
    except Exception as e:
        print(f"[ToolBrain tests] WARNING: failed to import local peft: {e}")

_ensure_local_peft()

# 3) 确保可以导入 smolagents（优先使用 knowledge_corpus 下的版本）
def _ensure_smolagents_on_path():
    test_file = Path(__file__).resolve()

    # 优先：KOCO-bench/KOCO-bench-ch/domain_code_generation/smolagents/knowledge_corpus/smolagents
    added_kc = False
    for up in range(2, 12):
        # kc_parent = test_file.parents[up] / "KOCO-bench/KOCO-bench-ch/domain_code_generation" / "smolagents" / "knowledge_corpus"
        kc_parent = test_file.parents[up] / "smolagents" / "knowledge_corpus"
        if kc_parent.exists() and (kc_parent / "smolagents").exists():
            sys.path.insert(0, str(kc_parent))
            print(f"[ToolBrain tests] smolagents (knowledge_corpus) added to sys.path: {kc_parent}")
            added_kc = True
            break

    # 尝试导入（若上一步已添加 knowledge_corpus，则将优先从该处导入）
    try:
        import smolagents  # noqa: F401
        src = getattr(__import__("smolagents"), "__file__", None)
        print(f"[ToolBrain tests] smolagents import OK ({'knowledge_corpus' if added_kc else 'environment'}) -> {src}")
        return
    except Exception:
        pass

    # 回退：workspace 根下的 smolagents/src
    for up in range(2, 12):
        candidate = test_file.parents[up] / "smolagents" / "src"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            print(f"[ToolBrain tests] smolagents/src added to sys.path: {candidate}")
            break

    try:
        import smolagents  # noqa: F401
        src = getattr(__import__("smolagents"), "__file__", None)
        print(f"[ToolBrain tests] smolagents import OK (from local src) -> {src}")
    except Exception as e:
        print(f"[ToolBrain tests] WARNING: failed to import smolagents: {e}")

_ensure_smolagents_on_path()