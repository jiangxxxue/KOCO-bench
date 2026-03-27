"""Tests for runner.py — core agent logic for the OpenHands pipeline."""

import ast
import json
import os
import shutil
import tempfile
import textwrap

import pytest

# Ensure the openhands package is importable
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from runner import (
    _collect_gt_locations,
    _extract_from_events,
    _extract_function_from_file,
    _parse_impl_location,
    _sanitize_completion,
    _stub_gt_functions,
    _stub_one_function,
    _stub_one_function_regex,
    _prepare_workspace,
    build_prompt,
    load_completed_ids,
    load_jsonl,
    save_completed_ids,
    save_jsonl,
)
from agent.sdk import _resolve_llm_model


# ═══════════════════════════════════════════════════════════════════════════
# JSONL I/O
# ═══════════════════════════════════════════════════════════════════════════


class TestJsonlIO:
    """Tests for load_jsonl / save_jsonl round-trip."""

    def test_round_trip(self, tmp_path):
        records = [
            {"function_name": "foo", "value": 1},
            {"function_name": "bar", "value": 2},
        ]
        path = str(tmp_path / "test.jsonl")
        save_jsonl(records, path)
        loaded = load_jsonl(path)
        assert loaded == records

    def test_round_trip_unicode(self, tmp_path):
        records = [{"name": "日本語テスト", "emoji": "🚀"}]
        path = str(tmp_path / "unicode.jsonl")
        save_jsonl(records, path)
        loaded = load_jsonl(path)
        assert loaded == records

    def test_load_skips_blank_lines(self, tmp_path):
        path = str(tmp_path / "blanks.jsonl")
        with open(path, "w") as f:
            f.write('{"a":1}\n\n{"b":2}\n  \n')
        loaded = load_jsonl(path)
        assert loaded == [{"a": 1}, {"b": 2}]

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "out.jsonl")
        save_jsonl([{"x": 1}], path)
        assert load_jsonl(path) == [{"x": 1}]

    def test_load_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        with open(path, "w") as f:
            f.write("")
        assert load_jsonl(path) == []


# ═══════════════════════════════════════════════════════════════════════════
# Resume support
# ═══════════════════════════════════════════════════════════════════════════


class TestResumeSupport:

    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "progress.json")
        ids = {"func_a", "func_b"}
        save_completed_ids(ids, path)
        loaded = load_completed_ids(path)
        assert loaded == ids

    def test_load_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        assert load_completed_ids(path) == set()

    def test_load_corrupted_file(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not json")
        assert load_completed_ids(path) == set()

    def test_load_missing_key(self, tmp_path):
        path = str(tmp_path / "nokey.json")
        with open(path, "w") as f:
            json.dump({"other_key": []}, f)
        # Should return empty set because completed_ids key is missing
        # but .get("completed_ids", []) returns [] → set()
        assert load_completed_ids(path) == set()


# ═══════════════════════════════════════════════════════════════════════════
# _collect_gt_locations
# ═══════════════════════════════════════════════════════════════════════════


class TestCollectGtLocations:

    def test_normal_case(self):
        records = [
            {"implementation_location": "code/path/to/file.py:line 10-20"},
            {"implementation_location": "code/path/to/file.py:line 30-40"},
        ]
        result = _collect_gt_locations(records)
        assert result == {"path/to/file.py": [(10, 20), (30, 40)]}

    def test_backslash_paths(self):
        records = [
            {"implementation_location": "code\\path\\to\\file.py:line 10-20"},
        ]
        result = _collect_gt_locations(records)
        assert result == {"path/to/file.py": [(10, 20)]}

    def test_no_code_prefix(self):
        """Paths without code/ prefix should be kept as-is."""
        records = [
            {"implementation_location": "path/to/file.py:line 5-15"},
        ]
        result = _collect_gt_locations(records)
        assert result == {"path/to/file.py": [(5, 15)]}

    def test_missing_location(self):
        records = [
            {"function_name": "foo"},  # no implementation_location
            {"implementation_location": ""},
        ]
        result = _collect_gt_locations(records)
        assert result == {}

    def test_invalid_format(self):
        records = [
            {"implementation_location": "no_colon_line"},
            {"implementation_location": "file.py:line abc"},
        ]
        result = _collect_gt_locations(records)
        assert result == {}

    def test_multiple_files(self):
        records = [
            {"implementation_location": "code/a.py:line 1-10"},
            {"implementation_location": "code/b.py:line 5-15"},
        ]
        result = _collect_gt_locations(records)
        assert "a.py" in result
        assert "b.py" in result


# ═══════════════════════════════════════════════════════════════════════════
# _parse_impl_location
# ═══════════════════════════════════════════════════════════════════════════


class TestParseImplLocation:

    def test_normal(self):
        path, start, end = _parse_impl_location("code/src/module.py:line 58-133")
        assert path == "src/module.py"
        assert start == 58
        assert end == 133

    def test_backslash(self):
        path, start, end = _parse_impl_location("code\\src\\module.py:line 10-20")
        assert path == "src/module.py"

    def test_no_code_prefix(self):
        path, start, end = _parse_impl_location("src/module.py:line 10-20")
        assert path == "src/module.py"

    def test_invalid_no_line(self):
        path, start, end = _parse_impl_location("src/module.py:10-20")
        assert path is None

    def test_invalid_no_dash(self):
        path, start, end = _parse_impl_location("code/file.py:line 10")
        assert path is None

    def test_empty_string(self):
        path, start, end = _parse_impl_location("")
        assert path is None


# ═══════════════════════════════════════════════════════════════════════════
# _stub_one_function (AST-based)
# ═══════════════════════════════════════════════════════════════════════════


class TestStubOneFunction:

    def _make_lines(self, code: str):
        """Helper: dedent code and split into lines with newlines."""
        return textwrap.dedent(code).splitlines(keepends=True)

    def test_simple_function_no_docstring(self):
        code = """\
        def foo(x):
            return x + 1
        """
        lines = self._make_lines(code)
        result = _stub_one_function(lines, 1, 2)
        text = "".join(result)
        assert "raise NotImplementedError" in text
        assert "return x + 1" not in text
        assert "def foo(x):" in text

    def test_function_with_docstring(self):
        code = """\
        def foo(x):
            \"\"\"Add one to x.\"\"\"
            return x + 1
        """
        lines = self._make_lines(code)
        result = _stub_one_function(lines, 1, 3)
        text = "".join(result)
        assert "raise NotImplementedError" in text
        assert '"""Add one to x."""' in text
        assert "return x + 1" not in text

    def test_function_with_multiline_docstring(self):
        code = """\
        def foo(x):
            \"\"\"
            Add one to x.

            Args:
                x: input
            \"\"\"
            return x + 1
        """
        lines = self._make_lines(code)
        result = _stub_one_function(lines, 1, 9)
        text = "".join(result)
        assert "raise NotImplementedError" in text
        assert "Args:" in text  # docstring kept
        assert "return x + 1" not in text

    def test_async_function(self):
        code = """\
        async def fetch(url):
            response = await get(url)
            return response
        """
        lines = self._make_lines(code)
        result = _stub_one_function(lines, 1, 3)
        text = "".join(result)
        assert "raise NotImplementedError" in text
        assert "async def fetch(url):" in text
        assert "await" not in text

    def test_preserves_surrounding_code(self):
        code = """\
        x = 1

        def target():
            return 42

        y = 2
        """
        lines = self._make_lines(code)
        result = _stub_one_function(lines, 3, 4)
        text = "".join(result)
        assert "x = 1" in text
        assert "y = 2" in text
        assert "return 42" not in text
        assert "raise NotImplementedError" in text

    def test_class_method(self):
        code = """\
        class MyClass:
            def method(self):
                return self.value
        """
        lines = self._make_lines(code)
        result = _stub_one_function(lines, 2, 3)
        text = "".join(result)
        assert "raise NotImplementedError" in text
        assert "return self.value" not in text
        assert "def method(self):" in text

    def test_function_with_decorator(self):
        code = """\
        @staticmethod
        def compute(x, y):
            return x * y
        """
        lines = self._make_lines(code)
        # start=1 covers the decorator, end=3 covers the body
        result = _stub_one_function(lines, 1, 3)
        text = "".join(result)
        assert "@staticmethod" in text
        assert "raise NotImplementedError" in text
        assert "return x * y" not in text

    def test_function_with_multiline_body(self):
        code = """\
        def process(data):
            result = []
            for item in data:
                result.append(item * 2)
            return result
        """
        lines = self._make_lines(code)
        result = _stub_one_function(lines, 1, 5)
        text = "".join(result)
        assert "raise NotImplementedError" in text
        assert "result = []" not in text
        assert "for item" not in text


# ═══════════════════════════════════════════════════════════════════════════
# _stub_one_function_regex (fallback)
# ═══════════════════════════════════════════════════════════════════════════


class TestStubOneFunctionRegex:

    def _make_lines(self, code: str):
        return textwrap.dedent(code).splitlines(keepends=True)

    def test_simple_function(self):
        code = """\
        def foo(x):
            return x + 1
        """
        lines = self._make_lines(code)
        result = _stub_one_function_regex(lines, 1, 2)
        text = "".join(result)
        assert "raise NotImplementedError" in text
        assert "def foo(x):" in text

    def test_multiline_signature_bug(self):
        """BUG: regex fallback breaks multi-line signatures.

        The continuation-line detection stops at parameter lines because
        they are non-empty and don't start with '#' or '@'. This causes
        the stub to replace the parameter list.
        """
        code = """\
        def foo(
            a,
            b,
        ):
            return a + b
        """
        lines = self._make_lines(code)
        result = _stub_one_function_regex(lines, 1, 5)
        text = "".join(result)
        # This SHOULD preserve "a," and "b," but currently doesn't
        # The regex fallback replaces from the first non-empty line after def
        # Documenting the current (buggy) behavior:
        assert "raise NotImplementedError" in text
        # The parameter list is incorrectly removed:
        assert "a," not in text  # BUG: parameters are stripped


# ═══════════════════════════════════════════════════════════════════════════
# _stub_gt_functions (multiple functions in file)
# ═══════════════════════════════════════════════════════════════════════════


class TestStubGtFunctions:

    def test_stub_multiple_functions_in_one_file(self, tmp_path):
        code = textwrap.dedent("""\
        def first():
            return 1

        def second():
            return 2

        def third():
            return 3
        """)
        file_path = tmp_path / "module.py"
        file_path.write_text(code)

        gt_locations = {
            "module.py": [(1, 2), (7, 8)],  # first and third
        }
        _stub_gt_functions(str(tmp_path), gt_locations)

        result = file_path.read_text()
        # first and third should be stubbed
        assert "return 1" not in result
        assert "return 3" not in result
        # second should remain
        assert "return 2" in result
        assert result.count("raise NotImplementedError") == 2

    def test_stub_across_multiple_files(self, tmp_path):
        (tmp_path / "a.py").write_text("def fa():\n    return 'a'\n")
        (tmp_path / "b.py").write_text("def fb():\n    return 'b'\n")

        gt_locations = {
            "a.py": [(1, 2)],
            "b.py": [(1, 2)],
        }
        _stub_gt_functions(str(tmp_path), gt_locations)

        assert "return 'a'" not in (tmp_path / "a.py").read_text()
        assert "return 'b'" not in (tmp_path / "b.py").read_text()

    def test_missing_file_ignored(self, tmp_path):
        gt_locations = {"nonexistent.py": [(1, 5)]}
        # Should not raise
        _stub_gt_functions(str(tmp_path), gt_locations)

    def test_preserves_docstrings_across_stubs(self, tmp_path):
        code = textwrap.dedent("""\
        def first():
            \"\"\"First function.\"\"\"
            return 1

        def second():
            \"\"\"Second function.\"\"\"
            return 2
        """)
        file_path = tmp_path / "module.py"
        file_path.write_text(code)

        gt_locations = {"module.py": [(1, 3), (5, 7)]}
        _stub_gt_functions(str(tmp_path), gt_locations)

        result = file_path.read_text()
        assert '"""First function."""' in result
        assert '"""Second function."""' in result
        assert "return 1" not in result
        assert "return 2" not in result


# ═══════════════════════════════════════════════════════════════════════════
# _prepare_workspace
# ═══════════════════════════════════════════════════════════════════════════


class TestPrepareWorkspace:

    def _setup_workspace(self, tmp_path):
        """Create a minimal workspace structure for testing."""
        workspace = tmp_path / "workspace_src"
        workspace.mkdir()
        (workspace / "main.py").write_text("def foo():\n    return 1\n")
        (workspace / "test_code").mkdir()
        (workspace / "test_code" / "test_main.py").write_text("# tests\n")
        (workspace / "__pycache__").mkdir()
        (workspace / "__pycache__" / "main.cpython-312.pyc").write_text("bytecode")
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "util.py").write_text("x = 1\n")

        kc = tmp_path / "knowledge_corpus"
        kc.mkdir()
        (kc / "docs.md").write_text("# Documentation\n")

        return str(workspace), str(kc)

    def test_creates_correct_structure(self, tmp_path):
        ws_root, kc_root = self._setup_workspace(tmp_path)
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir)

        result = _prepare_workspace(ws_root, kc_root, {}, out_dir)

        assert os.path.isdir(result["workspace"])
        assert os.path.isdir(result["code"])
        assert os.path.isdir(result["knowledge_corpus"])

    def test_excludes_test_code(self, tmp_path):
        ws_root, kc_root = self._setup_workspace(tmp_path)
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir)

        result = _prepare_workspace(ws_root, kc_root, {}, out_dir)

        assert not os.path.exists(os.path.join(result["code"], "test_code"))

    def test_excludes_pycache(self, tmp_path):
        ws_root, kc_root = self._setup_workspace(tmp_path)
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir)

        result = _prepare_workspace(ws_root, kc_root, {}, out_dir)

        assert not os.path.exists(os.path.join(result["code"], "__pycache__"))

    def test_copies_knowledge_corpus(self, tmp_path):
        ws_root, kc_root = self._setup_workspace(tmp_path)
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir)

        result = _prepare_workspace(ws_root, kc_root, {}, out_dir)

        docs_path = os.path.join(result["knowledge_corpus"], "docs.md")
        assert os.path.exists(docs_path)

    def test_copies_subdirectories(self, tmp_path):
        ws_root, kc_root = self._setup_workspace(tmp_path)
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir)

        result = _prepare_workspace(ws_root, kc_root, {}, out_dir)

        util_path = os.path.join(result["code"], "subdir", "util.py")
        assert os.path.exists(util_path)

    def test_stubs_gt_functions(self, tmp_path):
        ws_root, kc_root = self._setup_workspace(tmp_path)
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir)

        gt_locations = {"main.py": [(1, 2)]}
        result = _prepare_workspace(ws_root, kc_root, gt_locations, out_dir)

        code = open(os.path.join(result["code"], "main.py")).read()
        assert "raise NotImplementedError" in code
        assert "return 1" not in code

    def test_does_not_modify_original(self, tmp_path):
        ws_root, kc_root = self._setup_workspace(tmp_path)
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir)

        gt_locations = {"main.py": [(1, 2)]}
        _prepare_workspace(ws_root, kc_root, gt_locations, out_dir)

        # Original should be untouched
        original = open(os.path.join(ws_root, "main.py")).read()
        assert "return 1" in original


# ═══════════════════════════════════════════════════════════════════════════
# build_prompt
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildPrompt:

    def test_basic_prompt(self):
        record = {
            "function_name": "compute_score",
            "prompt": [
                {"role": "system", "content": "You are a Python programmer."},
                {"role": "user", "content": "Implement compute_score."},
            ],
        }
        repo_paths = {
            "workspace": "/tmp/ws",
            "knowledge_corpus": "/tmp/ws/knowledge_corpus",
            "code": "/tmp/ws/code",
        }
        result = build_prompt(record, "verl", repo_paths,
                              stub_file="/tmp/ws/code/module.py", stub_line=42)

        assert "compute_score" in result
        assert "verl" in result
        assert "You are a Python programmer." in result
        assert "Implement compute_score." in result
        assert "module.py" in result
        assert "line 42" in result

    def test_prompt_without_stub(self):
        record = {
            "function_name": "foo",
            "prompt": [{"role": "user", "content": "Do it."}],
        }
        repo_paths = {
            "workspace": "/tmp/ws",
            "knowledge_corpus": "/tmp/ws/kc",
            "code": "/tmp/ws/code",
        }
        result = build_prompt(record, "verl", repo_paths)
        assert "foo" in result
        assert "IMPORTANT CONTEXT" not in result  # no stub context

    def test_prompt_without_prompt_field(self):
        record = {"function_name": "bar"}
        repo_paths = {
            "workspace": "/tmp/ws",
            "knowledge_corpus": "/tmp/ws/kc",
            "code": "/tmp/ws/code",
        }
        result = build_prompt(record, "verl", repo_paths)
        assert "bar" in result

    def test_prompt_contains_rules(self):
        record = {"function_name": "test_fn", "prompt": []}
        repo_paths = {
            "workspace": "/tmp/ws",
            "knowledge_corpus": "/tmp/ws/kc",
            "code": "/tmp/ws/code",
        }
        result = build_prompt(record, "verl", repo_paths)
        assert "Do NOT modify any other functions" in result
        assert "Do NOT run tests" in result


# ═══════════════════════════════════════════════════════════════════════════
# _extract_function_from_file
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractFunctionFromFile:

    def test_simple_function(self, tmp_path):
        code = textwrap.dedent("""\
        def target():
            return 42
        """)
        f = tmp_path / "mod.py"
        f.write_text(code)
        result = _extract_function_from_file(str(f), "target")
        assert "def target():" in result
        assert "return 42" in result

    def test_returns_empty_for_stub(self, tmp_path):
        code = textwrap.dedent("""\
        def target():
            raise NotImplementedError
        """)
        f = tmp_path / "mod.py"
        f.write_text(code)
        result = _extract_function_from_file(str(f), "target")
        assert result == ""

    def test_stub_with_docstring(self, tmp_path):
        code = textwrap.dedent("""\
        def target():
            \"\"\"Docstring.\"\"\"
            raise NotImplementedError
        """)
        f = tmp_path / "mod.py"
        f.write_text(code)
        result = _extract_function_from_file(str(f), "target")
        assert result == ""

    def test_dotted_name_class_method(self, tmp_path):
        code = textwrap.dedent("""\
        class MyClass:
            def method(self):
                return self.value * 2
        """)
        f = tmp_path / "mod.py"
        f.write_text(code)
        result = _extract_function_from_file(str(f), "MyClass.method")
        assert "def method(self):" in result
        assert "return self.value * 2" in result

    def test_dedents_class_method(self, tmp_path):
        code = textwrap.dedent("""\
        class MyClass:
            def method(self):
                return 1
        """)
        f = tmp_path / "mod.py"
        f.write_text(code)
        result = _extract_function_from_file(str(f), "MyClass.method")
        # Should be dedented — no leading spaces on def line
        assert result.startswith("def method(self):")

    def test_async_function(self, tmp_path):
        code = textwrap.dedent("""\
        async def fetch(url):
            data = await get(url)
            return data
        """)
        f = tmp_path / "mod.py"
        f.write_text(code)
        result = _extract_function_from_file(str(f), "fetch")
        assert "async def fetch" in result
        assert "await get(url)" in result

    def test_nonexistent_file(self, tmp_path):
        result = _extract_function_from_file(str(tmp_path / "nope.py"), "foo")
        assert result == ""

    def test_nonexistent_function(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def other():\n    pass\n")
        result = _extract_function_from_file(str(f), "nonexistent")
        assert result == ""

    def test_syntax_error_file(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def foo(\n")  # invalid syntax
        result = _extract_function_from_file(str(f), "foo")
        assert result == ""

    def test_with_decorators(self, tmp_path):
        code = textwrap.dedent("""\
        @staticmethod
        def compute(x, y):
            return x * y
        """)
        f = tmp_path / "mod.py"
        f.write_text(code)
        result = _extract_function_from_file(str(f), "compute")
        assert "@staticmethod" in result
        assert "return x * y" in result

    def test_implemented_function_with_notimplementederror_call(self, tmp_path):
        """NotImplementedError("msg") should NOT be treated as stub."""
        code = textwrap.dedent("""\
        def target():
            if not ready:
                raise NotImplementedError("not ready yet")
            return 42
        """)
        f = tmp_path / "mod.py"
        f.write_text(code)
        result = _extract_function_from_file(str(f), "target")
        # ast.Name check won't match ast.Call, so this should be extracted
        assert "return 42" in result


# ═══════════════════════════════════════════════════════════════════════════
# _sanitize_completion
# ═══════════════════════════════════════════════════════════════════════════


class TestSanitizeCompletion:

    def test_plain_code_unchanged(self):
        code = "return x + 1\n"
        assert _sanitize_completion(code, "fn") == code

    def test_empty_code(self):
        assert _sanitize_completion("", "fn") == ""
        assert _sanitize_completion("  ", "fn") == "  "

    def test_unwraps_json_implementation_key(self):
        wrapped = json.dumps({"implementation": "return 42"})
        result = _sanitize_completion(wrapped, "fn")
        assert result == "return 42"

    def test_unwraps_json_code_key(self):
        wrapped = json.dumps({"code": "x = 1"})
        result = _sanitize_completion(wrapped, "fn")
        assert result == "x = 1"

    def test_unwraps_json_function_name_key(self):
        wrapped = json.dumps({"my_func": "return True"})
        result = _sanitize_completion(wrapped, "my_func")
        assert result == "return True"

    def test_fixes_escaped_newlines(self):
        code = "line1\\nline2\\n    indented"
        result = _sanitize_completion(code, "fn")
        assert "line1\nline2\n    indented" == result

    def test_no_false_positive_escape_fix(self):
        """Real newlines present → don't "fix" literal \\n."""
        code = "line1\nhas_escaped\\n_in_string"
        result = _sanitize_completion(code, "fn")
        assert result == code  # no change

    def test_strips_markdown_python_fence(self):
        code = "```python\ndef foo():\n    return 1\n```"
        result = _sanitize_completion(code, "fn")
        assert result == "def foo():\n    return 1"

    def test_strips_markdown_py_fence(self):
        code = "```py\nreturn 1\n```"
        result = _sanitize_completion(code, "fn")
        assert result == "return 1"

    def test_strips_bare_markdown_fence(self):
        code = "```\nreturn 1\n```"
        result = _sanitize_completion(code, "fn")
        assert result == "return 1"

    def test_combined_json_then_fence(self):
        """JSON wrapping with markdown fences inside."""
        inner = "```python\ndef foo():\n    return 1\n```"
        wrapped = json.dumps({"implementation": inner})
        result = _sanitize_completion(wrapped, "fn")
        assert "def foo():" in result
        assert "```" not in result


# ═══════════════════════════════════════════════════════════════════════════
# _extract_from_events
# ═══════════════════════════════════════════════════════════════════════════


class _FakeEvent:
    """Minimal mock for SDK event objects used by _extract_from_events."""
    def __init__(self, tool_name=None, action=None):
        self.tool_name = tool_name
        self.action = action


class TestExtractFromEvents:
    """Tests for _extract_from_events (SDK event list version)."""

    def test_empty_events(self):
        result = _extract_from_events([], "foo")
        assert result == ""

    def test_no_file_editor_events(self):
        events = [_FakeEvent(tool_name="terminal")]
        result = _extract_from_events(events, "foo")
        assert result == ""

    def test_extracts_function_from_created_file(self):
        code = "def target(x):\n    return x * 2\n"
        events = [_FakeEvent(
            tool_name="file_editor",
            action={
                "command": "create",
                "path": "/workspace/result.py",
                "file_text": code,
            },
        )]
        result = _extract_from_events(events, "target")
        assert "return x * 2" in result

    def test_async_def_not_matched(self):
        """BUG: regex doesn't handle async def."""
        code = "async def fetch(url):\n    return await get(url)\n"
        events = [_FakeEvent(
            tool_name="file_editor",
            action={
                "command": "create",
                "path": "/workspace/result.py",
                "file_text": code,
            },
        )]
        result = _extract_from_events(events, "fetch")
        # Current implementation misses async def — documenting this bug
        assert result == ""  # BUG: should find async def fetch


# ═══════════════════════════════════════════════════════════════════════════
# _resolve_llm_model
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveLlmModel:

    def test_adds_openrouter_prefix(self):
        result = _resolve_llm_model("deepseek/deepseek-v3.2", "https://openrouter.ai/api/v1")
        assert result == "openrouter/deepseek/deepseek-v3.2"

    def test_no_double_prefix(self):
        result = _resolve_llm_model("openrouter/deepseek/deepseek-v3.2", "https://openrouter.ai/api/v1")
        assert result == "openrouter/deepseek/deepseek-v3.2"

    def test_non_openrouter_url(self):
        result = _resolve_llm_model("gpt-4", "https://api.openai.com/v1")
        assert result == "gpt-4"

    def test_empty_base_url(self):
        result = _resolve_llm_model("model", "")
        assert result == "model"


# ═══════════════════════════════════════════════════════════════════════════
# Integration: stubbing then extraction round-trip
# ═══════════════════════════════════════════════════════════════════════════


class TestStubAndExtractRoundTrip:
    """Verify that stubbing a function, modifying it, then extracting works."""

    def test_stub_then_implement_then_extract(self, tmp_path):
        original = textwrap.dedent("""\
        import math

        def helper():
            return 1

        def target(x, y):
            \"\"\"Compute something.\"\"\"
            return math.sqrt(x**2 + y**2)

        def other():
            return 2
        """)
        f = tmp_path / "module.py"
        f.write_text(original)

        # Step 1: Stub the target function
        gt_locations = {"module.py": [(6, 9)]}
        _stub_gt_functions(str(tmp_path), gt_locations)

        stubbed = f.read_text()
        assert "raise NotImplementedError" in stubbed
        assert "math.sqrt" not in stubbed
        assert '"""Compute something."""' in stubbed
        # helper and other are untouched
        assert "return 1" in stubbed
        assert "return 2" in stubbed

        # Step 2: Verify extraction returns empty (it's a stub)
        result = _extract_function_from_file(str(f), "target")
        assert result == ""

        # Step 3: Simulate agent implementing the function
        stubbed_lines = stubbed.splitlines(keepends=True)
        new_lines = []
        skip_stub = False
        for line in stubbed_lines:
            if "# TODO: implement this function" in line:
                skip_stub = True
                continue
            if "raise NotImplementedError" in line and skip_stub:
                # Replace stub with implementation
                indent = len(line) - len(line.lstrip())
                ws = " " * indent
                new_lines.append(f"{ws}return math.sqrt(x**2 + y**2)\n")
                skip_stub = False
                continue
            new_lines.append(line)
        f.write_text("".join(new_lines))

        # Step 4: Extract should now return the implementation
        result = _extract_function_from_file(str(f), "target")
        assert "def target(x, y):" in result
        assert "math.sqrt" in result

    def test_stub_class_method_then_extract(self, tmp_path):
        original = textwrap.dedent("""\
        class Engine:
            def __init__(self):
                self.state = {}

            def process(self, data):
                \"\"\"Process incoming data.\"\"\"
                result = {}
                for k, v in data.items():
                    result[k] = v * 2
                return result

            def reset(self):
                self.state = {}
        """)
        f = tmp_path / "engine.py"
        f.write_text(original)

        # Stub process method (lines 5-10 in the original)
        gt_locations = {"engine.py": [(5, 10)]}
        _stub_gt_functions(str(tmp_path), gt_locations)

        stubbed = f.read_text()
        assert "raise NotImplementedError" in stubbed
        assert "v * 2" not in stubbed
        assert '"""Process incoming data."""' in stubbed

        # __init__ and reset should be untouched
        assert "self.state = {}" in stubbed


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_stub_last_function_in_file(self, tmp_path):
        """Function at the very end of file (no trailing content)."""
        code = "def last():\n    return 99\n"
        f = tmp_path / "mod.py"
        f.write_text(code)

        lines = code.splitlines(keepends=True)
        result = _stub_one_function(lines, 1, 2)
        text = "".join(result)
        assert "raise NotImplementedError" in text
        assert "return 99" not in text
        # Should be valid Python
        ast.parse(text)

    def test_stub_single_line_function(self, tmp_path):
        """Function with body on same line as def (unusual but valid)."""
        code = "x = 0\ndef f(): return 1\ny = 2\n"
        f = tmp_path / "mod.py"
        f.write_text(code)

        lines = code.splitlines(keepends=True)
        result = _stub_one_function(lines, 2, 2)
        text = "".join(result)
        # Should contain stub
        assert "raise NotImplementedError" in text
        assert "x = 0" in text
        assert "y = 2" in text

    def test_sanitize_none_code(self):
        result = _sanitize_completion(None, "fn")
        assert result is None

    def test_collect_gt_locations_with_single_line(self):
        """Location like 'code/file.py:line 5-5' (single-line function)."""
        records = [{"implementation_location": "code/f.py:line 5-5"}]
        result = _collect_gt_locations(records)
        assert result == {"f.py": [(5, 5)]}
