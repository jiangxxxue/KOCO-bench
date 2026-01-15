import sys
import os
from pathlib import Path

# 1) Add this instance's code root directory to sys.path and switch to it as current working directory
REPLICA_ROOT = Path(__file__).resolve().parents[1]  # .../ToolBrain/code
sys.path.insert(0, str(REPLICA_ROOT))
os.chdir(str(REPLICA_ROOT))

print(f"[ToolBrain tests] Added to sys.path (replica root): {REPLICA_ROOT}")
print(f"[ToolBrain tests] CWD: {os.getcwd()}")

# 2) Ensure local peft package can be discovered (avoid peft.__spec__ is None error during transformers optional dependency detection)
#    We provide a minimal usable peft package under code/peft/__init__.py.
def _ensure_local_peft():
    try:
        import peft  # type: ignore
        # If already loaded but without __spec__, explicitly reimport to ensure correct spec
        if getattr(peft, "__spec__", None) is None:
            print("[ToolBrain tests] peft present without __spec__, reloading...")
            del sys.modules["peft"]
            import importlib
            peft = importlib.import_module("peft")  # noqa: F401
        print("[ToolBrain tests] peft import OK (local stub)")
    except Exception as e:
        print(f"[ToolBrain tests] WARNING: failed to import local peft: {e}")

_ensure_local_peft()

# 3) Ensure smolagents can be imported (prioritize version under knowledge_corpus)
def _ensure_smolagents_on_path():
    test_file = Path(__file__).resolve()

    # Priority: KOCO-bench/KOCO-bench-en/domain_code_generation/smolagents/knowledge_corpus/smolagents
    added_kc = False
    for up in range(2, 12):
        # kc_parent = test_file.parents[up] / "KOCO-bench/KOCO-bench-en/domain_code_generation" / "smolagents" / "knowledge_corpus"
        kc_parent = test_file.parents[up] / "smolagents" / "knowledge_corpus"
        if kc_parent.exists() and (kc_parent / "smolagents").exists():
            sys.path.insert(0, str(kc_parent))
            print(f"[ToolBrain tests] smolagents (knowledge_corpus) added to sys.path: {kc_parent}")
            added_kc = True
            break

    # Try importing (if knowledge_corpus was added in previous step, it will be imported from there first)
    try:
        import smolagents  # noqa: F401
        src = getattr(__import__("smolagents"), "__file__", None)
        print(f"[ToolBrain tests] smolagents import OK ({'knowledge_corpus' if added_kc else 'environment'}) -> {src}")
        return
    except Exception:
        pass

    # Fallback: smolagents/src under workspace root
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