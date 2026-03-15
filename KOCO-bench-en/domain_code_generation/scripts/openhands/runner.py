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

def _collect_gt_locations(records):
    """Collect GT function line ranges from all records for the example.

    Returns: dict mapping file paths (relative to code/) to lists of
             (start_line, end_line) tuples (1-indexed, inclusive).
    """
    locations = {}
    for r in records:
        impl_loc = r.get("implementation_location", "")
        if not impl_loc:
            continue
        # Format: "code/path/to/file.py:line 86-87"
        # or with backslashes: "code\\path\\to\\file.py:line 86-87"
        parts = impl_loc.split(":line ")
        if len(parts) != 2:
            continue
        file_part = parts[0].replace("\\", "/")
        if file_part.startswith("code/"):
            file_part = file_part[len("code/"):]
        try:
            start_s, end_s = parts[1].split("-")
            locations.setdefault(file_part, []).append((int(start_s), int(end_s)))
        except ValueError:
            continue
    return locations


def _strip_gt_lines(code_dst, gt_locations):
    """Remove annotated GT function lines from copied files."""
    for rel_path, ranges in gt_locations.items():
        file_path = os.path.join(code_dst, rel_path)
        if not os.path.exists(file_path):
            continue
        lines_to_remove = set()
        for start, end in ranges:
            lines_to_remove.update(range(start, end + 1))
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        new_lines = [l for i, l in enumerate(lines, 1) if i not in lines_to_remove]
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)


def _prepare_workspace(workspace_root, knowledge_corpus_root, gt_locations, tmp_dir):
    """Copy workspace + knowledge_corpus, strip GT function bodies and test_code.

    Both directories are placed under ``tmp_dir/workspace/`` so the agent's
    cwd can be set there and ``ls .`` shows exactly ``code/`` and
    ``knowledge_corpus/`` — nothing else (no oh_persist, logs, etc.).

    Returns: {"workspace": abs_path, "knowledge_corpus": abs_path, "code": abs_path}
    """
    ws_dir = os.path.join(tmp_dir, "workspace")
    os.makedirs(ws_dir, exist_ok=True)

    # Copy knowledge_corpus
    kc_dst = os.path.join(ws_dir, "knowledge_corpus")
    shutil.copytree(knowledge_corpus_root, kc_dst, symlinks=True)

    # Copy code/ excluding test_code and caches
    code_dst = os.path.join(ws_dir, "code")
    def _ignore(_dir, contents):
        return {c for c in contents if c in ("test_code", "__pycache__", ".pytest_cache")}
    shutil.copytree(workspace_root, code_dst, symlinks=True, ignore=_ignore)

    # Strip GT function lines from copied files
    _strip_gt_lines(code_dst, gt_locations)

    return {"workspace": ws_dir, "knowledge_corpus": kc_dst, "code": code_dst}


def build_prompt(record: dict, framework: str, repo_paths: dict) -> str:
    """Build the task prompt for the OpenHands headless agent.

    The agent runs with cwd set to the workspace directory which contains
    ``code/`` and ``knowledge_corpus/`` as its only children.
    ``repo_paths`` has keys ``workspace``, ``knowledge_corpus``, and ``code``.
    """
    function_name = record["function_name"]

    # Extract system/user context from the pre-built prompt
    system_context = ""
    user_task = ""
    if record.get("prompt") and isinstance(record["prompt"], list):
        for msg in record["prompt"]:
            if msg.get("role") == "system":
                system_context = msg["content"]
            elif msg.get("role") == "user":
                user_task = msg["content"]

    result_file = os.path.join(repo_paths["code"], "implementation_result.py")

    return f"""You are working in a repository for the {framework} framework.
{system_context}

TASK: Implement the function `{function_name}`.

{user_task}

You can freely explore the following known repositories to obtain the required information:
- Framework Knowledge Base: {repo_paths["knowledge_corpus"]}
- Development Repository: {repo_paths["code"]}

Please use the code in these repositories to implement the required functionality.

INSTRUCTIONS:
1. Explore the repositories to understand the codebase, domain knowledge, and callable functions.
2. Write your implementation of `{function_name}`.
3. MANDATORY FINAL STEP — you MUST do this before finishing:
   Write the file {result_file} using the file_editor tool.
   The file must contain ONLY the function implementation as plain Python code.

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


def _sanitize_completion(code: str, function_name: str) -> str:
    """Clean up agent output to plain Python, enforced in code rather than prompt.

    Handles common agent output issues:
    - Double-escaped newlines (literal \\n instead of real newlines)
    - JSON wrapping ({"implementation": "..."})
    - Markdown fences (```python ... ```)
    """
    if not code or not code.strip():
        return code

    # 1. Unwrap JSON (agent wrote {"implementation": "..."} or similar)
    #    Must run before escape-fixing so json.loads handles escapes correctly.
    stripped = code.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                for key in ("implementation", "code", "function", function_name):
                    if key in obj and isinstance(obj[key], str):
                        code = obj[key]
                        print(f"    [{function_name}] Sanitize: unwrapped JSON key '{key}'")
                        break
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Fix double-escaped newlines (literal \n instead of real newlines)
    if '\n' not in code and '\\n' in code:
        code = (code
                .replace('\\n', '\n')
                .replace('\\t', '\t')
                .replace('\\"', '"'))
        print(f"    [{function_name}] Sanitize: fixed escaped newlines")

    # 3. Strip markdown fences
    stripped = code.strip()
    for fence in ('```python', '```py', '```'):
        if stripped.startswith(fence):
            code = stripped[len(fence):]
            end = code.rfind('```')
            if end != -1:
                code = code[:end]
            code = code.strip()
            print(f"    [{function_name}] Sanitize: stripped markdown fences")
            break

    return code


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
    example: str,
    workspace_root: str,
    knowledge_corpus_root: str,
    gt_locations: dict,
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    max_iterations: int = 50,
) -> dict:
    """Run the OpenHands headless agent for one function.

    Steps:
      1. copy workspace to a temp directory for isolation
      2. write the task prompt to a temp file
      3. invoke ``openhands --headless`` with cwd set to the workspace
      4. read ``implementation_result.py`` from the (temp) workspace
      5. clean up

    Returns the *record* dict augmented with ``completions`` and ``status``.
    The ``completions`` field is always a list with exactly one string:
    the implementation code on success, or an empty string on failure.
    This ensures failed attempts are counted in the evaluation denominator.
    """
    function_name = record["function_name"]
    print(f"    [{function_name}] Starting agent...")

    tmp_dir = tempfile.mkdtemp(prefix=f"oh_{function_name}_")

    try:
        # --- Workspace isolation (strip GT, exclude test_code) ---
        repo_paths = _prepare_workspace(workspace_root, knowledge_corpus_root, gt_locations, tmp_dir)
        work_dir = repo_paths["workspace"]

        prompt = build_prompt(record, framework, repo_paths)

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
                {"name": "task", "params": {}},
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
        result_py = os.path.join(repo_paths["code"], "implementation_result.py")
        result_json = os.path.join(repo_paths["code"], "implementation_result.json")
        implementation = ""

        # Primary: read implementation_result.py (plain Python)
        if os.path.exists(result_py):
            with open(result_py, "r", encoding="utf-8") as f:
                implementation = f.read().strip()

        # Fallback 1: legacy implementation_result.json
        if not implementation and os.path.exists(result_json):
            try:
                with open(result_json, "r", encoding="utf-8") as f:
                    result_data = json.load(f)
                implementation = result_data.get("implementation", "")
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback 2: scan conversation events for file_editor create actions
        if not implementation:
            implementation = _extract_from_events(oh_persist_dir, function_name)

        # Sanitize: fix escaping, unwrap JSON/markdown
        implementation = _sanitize_completion(implementation, function_name)

        if implementation:
            record["completions"] = [implementation]
            record["status"] = "success"
            print(f"    [{function_name}] Success ({len(implementation)} chars)")
        else:
            record["completions"] = [""]
            record["status"] = "no_result"
            record["results"] = [False]
            record["pass_ratios"] = [0.0]
            print(f"    [{function_name}] No implementation found")

    except subprocess.TimeoutExpired:
        print(f"    [{function_name}] Timeout after {max_iterations * 120}s")
        record["completions"] = [""]
        record["status"] = "timeout"
        record["results"] = [False]
        record["pass_ratios"] = [0.0]
    except FileNotFoundError:
        print(f"    [{function_name}] Error: 'openhands' command not found.")
        print("      Install with: uv tool install openhands --python 3.12")
        record["completions"] = [""]
        record["status"] = "error"
        record["error"] = "openhands not installed"
        record["results"] = [False]
        record["pass_ratios"] = [0.0]
    except Exception as e:
        print(f"    [{function_name}] Error: {e}")
        record["completions"] = [""]
        record["status"] = "error"
        record["error"] = str(e)
        record["results"] = [False]
        record["pass_ratios"] = [0.0]
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return record
