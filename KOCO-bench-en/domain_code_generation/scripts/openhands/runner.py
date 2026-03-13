"""OpenHands agent logic for KOCO-bench code generation.

Invokes the OpenHands headless CLI (subprocess) to run an agent that explores
a repository and implements a function.  Each invocation gets an isolated
workspace copy so agents cannot interfere with each other or pollute the
source tree.

Provides: prompt construction, single-instance agent execution, JSONL I/O,
and resume helpers.
"""

import json
import os
import shutil
import subprocess
import tempfile


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    """Load records from a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, path: str) -> None:
    """Save records to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_ids(path: str) -> set:
    """Load set of completed function names from a progress file."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("completed_ids", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def save_completed_ids(ids: set, path: str) -> None:
    """Save set of completed function names to a progress file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"completed_ids": sorted(ids)}, f, indent=2)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _to_workspace_path(relative_path: str, work_dir: str) -> str:
    """Convert a code-relative path to an absolute path inside the workspace.

    Data files use paths like ``code/examples/foo.py`` relative to
    ``test_examples/{example}/``.  Since OpenHands CLI runs locally with
    ``cwd`` set to the workspace directory, convert to absolute paths.
    """
    if relative_path.startswith("code/"):
        relative_path = relative_path[len("code/"):]
    return os.path.join(work_dir, relative_path)


def build_prompt(record: dict, framework: str, work_dir: str) -> str:
    """Build the task prompt for the OpenHands headless agent.

    The agent runs locally with cwd set to ``work_dir`` (the repo code).
    """
    function_name = record["function_name"]
    impl_location = record.get("implementation_location", "")
    test_code_path = record.get("test_code_path", "")

    # Convert paths to absolute paths in the workspace
    impl_path_part = impl_location.split(":")[0] if ":" in impl_location else impl_location
    impl_abs = _to_workspace_path(impl_path_part, work_dir)
    impl_line_info = impl_location[len(impl_path_part):] if impl_path_part else ""
    test_abs = _to_workspace_path(test_code_path, work_dir) if test_code_path else ""

    # Extract system/user context from the pre-built prompt
    system_context = ""
    user_task = ""
    if record.get("prompt") and isinstance(record["prompt"], list):
        for msg in record["prompt"]:
            if msg.get("role") == "system":
                system_context = msg["content"]
            elif msg.get("role") == "user":
                user_task = msg["content"]

    result_file = os.path.join(work_dir, "implementation_result.json")

    return f"""You are working in a repository for the {framework} framework.
{system_context}

Your current working directory is: {work_dir}

TASK: Implement the function `{function_name}`.

{user_task}

INSTRUCTIONS (follow exactly in order):
1. Read the implementation file: {impl_abs}
   Focus on lines around {impl_line_info} to understand imports, class structure, and the function signature.
2. Read the test file: {test_abs}
3. Write your implementation of `{function_name}`.
4. MANDATORY FINAL STEP — you MUST do this before finishing:
   Write the file {result_file} using the file_editor tool with this EXACT JSON:
   {{"function_name": "{function_name}", "implementation": "<complete function code here>"}}
   The "implementation" value must be a valid JSON string with escaped newlines (\\n) and quotes (\\").
   It MUST contain the COMPLETE function, starting with the def/async def line and including the full body.
   Example: "def foo(x):\\n    return x + 1"

RULES:
- Do NOT run tests. Do NOT create helper scripts. Do NOT debug.
- Your ONLY deliverable is {result_file}. If you do not create this file, your work is lost.
- Finish as soon as you have written {result_file}.
"""


# ---------------------------------------------------------------------------
# Single-instance agent execution
# ---------------------------------------------------------------------------

def _extract_from_events(persist_dir: str, function_name: str) -> str:
    """Fallback: scan OpenHands conversation events for created .py files
    and extract the function body for ``function_name``."""
    import glob
    import re as _re

    conv_dirs = glob.glob(os.path.join(persist_dir, "conversations", "*", "events"))
    if not conv_dirs:
        return ""

    events_dir = conv_dirs[0]
    # Collect file_editor create actions (newest last)
    created_files = {}
    for evt_path in sorted(glob.glob(os.path.join(events_dir, "*.json"))):
        try:
            with open(evt_path, "r", encoding="utf-8") as f:
                evt = json.load(f)
            if evt.get("tool_name") == "file_editor":
                action = evt.get("action", {})
                if action.get("command") == "create" and action.get("file_text"):
                    created_files[action["path"]] = action["file_text"]
        except Exception:
            continue

    if not created_files:
        return ""

    # Look for a file containing the target function
    for path, content in reversed(list(created_files.items())):
        if function_name in content:
            # Try to extract the function body
            pattern = _re.compile(
                rf'^(def\s+{_re.escape(function_name)}\s*\(.*?\)\s*.*?:\s*\n)'
                r'((?:(?:[ \t]+.+|[ \t]*#.+|[ \t]*)\n)*)',
                _re.MULTILINE,
            )
            m = pattern.search(content)
            if m:
                body = m.group(2)
                # Dedent the body
                lines = body.rstrip('\n').split('\n')
                if lines:
                    indent = len(lines[0]) - len(lines[0].lstrip())
                    body = '\n'.join(l[indent:] for l in lines)
                print(f"    [{function_name}] Fallback: extracted from {os.path.basename(path)}")
                return body

    return ""


def _resolve_llm_model(model: str, base_url: str) -> str:
    """Add ``openrouter/`` prefix when the base URL is OpenRouter.

    litellm uses the model-name prefix for provider routing.  Without the
    ``openrouter/`` prefix, ``deepseek/…`` gets routed directly to the
    DeepSeek API, ignoring the custom base URL.
    """
    if "openrouter.ai" in base_url and not model.startswith("openrouter/"):
        return f"openrouter/{model}"
    return model


def run_single_instance(
    record: dict,
    framework: str,
    workspace_root: str,
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    max_iterations: int = 50,
    isolate: bool = True,
) -> dict:
    """Run the OpenHands headless agent for one function.

    Steps:
      1. (optionally) copy workspace to a temp directory for isolation
      2. write the task prompt to a temp file
      3. invoke ``openhands --headless`` with cwd set to the workspace
      4. read ``implementation_result.json`` from the (temp) workspace
      5. clean up

    Returns the *record* dict augmented with ``completions`` and ``status``.
    The ``completions`` field is a list of code strings (length 0 or 1),
    which is what ``execution_evaluation_pure.py`` expects downstream.
    """
    function_name = record["function_name"]
    print(f"    [{function_name}] Starting agent...")

    tmp_dir = tempfile.mkdtemp(prefix=f"oh_{function_name}_")

    try:
        # --- Workspace isolation ---
        if isolate:
            work_dir = os.path.join(tmp_dir, "code")
            shutil.copytree(workspace_root, work_dir, symlinks=True)
        else:
            work_dir = workspace_root

        prompt = build_prompt(record, framework, work_dir)

        # --- Write prompt to file (avoids shell length limits) ---
        prompt_file = os.path.join(tmp_dir, "task_prompt.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)

        # --- Environment for OpenHands ---
        env = os.environ.copy()
        llm_model = _resolve_llm_model(model, base_url)
        env["LLM_API_KEY"] = api_key
        env["LLM_MODEL"] = llm_model
        if base_url:
            env["LLM_BASE_URL"] = base_url

        # Use an isolated persistence dir so we can write
        # agent_settings.json without polluting ~/.openhands.
        oh_persist_dir = os.path.join(tmp_dir, "oh_persist")
        os.makedirs(oh_persist_dir, exist_ok=True)
        env["OPENHANDS_PERSISTENCE_DIR"] = oh_persist_dir

        # Write agent_settings.json — sets max_output_tokens to a sane
        # value.  litellm's model_info reports max_output_tokens=163840
        # for deepseek-v3.2, which equals the full context window and
        # causes every request to fail with "context length exceeded".
        agent_settings = {
            "llm": {
                "model": llm_model,
                "api_key": api_key,
                "base_url": base_url,
                "max_output_tokens": 65536,
                "temperature": 0.0,
                "usage_id": "agent",
            },
            "tools": [
                {"name": "terminal", "params": {}},
                {"name": "file_editor", "params": {}},
                {"name": "task_tracker", "params": {}},
                {"name": "delegate", "params": {}},
            ],
            "include_default_tools": ["FinishTool", "ThinkTool"],
        }
        with open(os.path.join(oh_persist_dir, "agent_settings.json"), "w") as f:
            json.dump(agent_settings, f)

        # --- Invoke OpenHands headless ---
        cmd = [
            "openhands", "--headless",
            "--override-with-envs",
            "-f", prompt_file,
        ]

        print(f"    [{function_name}] Running: openhands --headless (model={llm_model}) ...")
        log_file = os.path.join(tmp_dir, "openhands.log")
        with open(log_file, "w", encoding="utf-8") as log_fh:
            proc = subprocess.run(
                cmd,
                env=env,
                cwd=work_dir,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=max_iterations * 120,
            )

        if proc.returncode != 0:
            print(f"    [{function_name}] openhands exited with code {proc.returncode}")

        # Show last lines of log for diagnostics
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                log_lines = f.readlines()
            tail = log_lines[-20:] if len(log_lines) > 20 else log_lines
            for line in tail:
                print(f"      {line.rstrip()}")

        # --- Extract result ---
        result_file = os.path.join(work_dir, "implementation_result.json")
        implementation = ""

        # Primary: read implementation_result.json
        if os.path.exists(result_file):
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    result_data = json.load(f)
                implementation = result_data.get("implementation", "")
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: scan conversation events for file_editor create actions
        if not implementation:
            implementation = _extract_from_events(oh_persist_dir, function_name)

        if implementation:
            record["completions"] = [implementation]
            record["status"] = "success"
            print(f"    [{function_name}] Success ({len(implementation)} chars)")
        else:
            record["completions"] = []
            record["status"] = "no_result"
            print(f"    [{function_name}] No implementation found")

    except subprocess.TimeoutExpired:
        print(f"    [{function_name}] Timeout after {max_iterations * 120}s")
        record["completions"] = []
        record["status"] = "timeout"
    except FileNotFoundError:
        print(f"    [{function_name}] Error: 'openhands' command not found.")
        print("      Install with: uv tool install openhands --python 3.12")
        record["completions"] = []
        record["status"] = "error"
        record["error"] = "openhands not installed"
    except Exception as e:
        print(f"    [{function_name}] Error: {e}")
        record["completions"] = []
        record["status"] = "error"
        record["error"] = str(e)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return record
