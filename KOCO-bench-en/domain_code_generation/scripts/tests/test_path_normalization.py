"""Tests for path normalization in the parser and evaluator.

Covers the _clean_location() helper and the evaluator's runtime
normalization of implementation_location / test_code_path fields.
Real examples taken from raganything markdown sources.
"""

import os
import sys
import json
import tempfile
import textwrap

import pytest

# Make the scripts package importable
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from parse_algorithm_methods import _clean_location, parse_markdown_file, process_functions


# ---------------------------------------------------------------------------
# _clean_location: pure function tests
# ---------------------------------------------------------------------------

class TestCleanLocation:
    """Core normalization function — all real patterns from raganything markdown."""

    # Backslash paths (BookWorm, Chat-ANYTHING)
    def test_backslash_with_code_prefix_and_line(self):
        assert _clean_location(r"code\raganything_example.py:line 90-245") == "raganything_example.py:line 90-245"

    def test_backslash_test_path(self):
        assert _clean_location(r"code\test_code\my_test_process_with_rag.py") == "test_code/my_test_process_with_rag.py"

    # Backtick-wrapped backslash paths (law-unleashed-rag, obsidianGraphRAG, rag4chat)
    def test_backtick_wrapped_backslash_with_line(self):
        assert _clean_location(r"`code\src\services\rag_anything_service.py:line 46-154`") == "src/services/rag_anything_service.py:line 46-154"

    def test_backtick_wrapped_backslash_test_path(self):
        assert _clean_location(r"`code\test_code\my_test_initialize_rag_storage.py`") == "test_code/my_test_initialize_rag_storage.py"

    def test_backtick_wrapped_nested_backslash(self):
        assert _clean_location(r"`code\services\build_database.py:line 263-346`") == "services/build_database.py:line 263-346"

    # Already-clean forward-slash paths (e.g. verl, open-r1)
    def test_forward_slash_with_code_prefix(self):
        assert _clean_location("code/recipe/prime/prime_core_algos.py:line 21-79") == "recipe/prime/prime_core_algos.py:line 21-79"

    def test_forward_slash_no_code_prefix(self):
        assert _clean_location("src/model.py:line 1-50") == "src/model.py:line 1-50"

    # Edge cases
    def test_dot_slash_prefix(self):
        assert _clean_location("./code/foo.py:line 1-10") == "foo.py:line 1-10"

    def test_double_quoted(self):
        assert _clean_location('"code/bar.py:line 5-15"') == "bar.py:line 5-15"

    def test_single_quoted(self):
        assert _clean_location("'code/baz.py'") == "baz.py"

    def test_whitespace_stripping(self):
        assert _clean_location("  code/foo.py  ") == "foo.py"

    def test_plain_filename(self):
        assert _clean_location("main.py:line 1-100") == "main.py:line 1-100"

    def test_empty_string(self):
        assert _clean_location("") == ""

    def test_only_backticks(self):
        assert _clean_location("``") == ""

    def test_no_code_prefix_backslash(self):
        """Backslash path that doesn't start with code\\."""
        assert _clean_location(r"src\utils\helper.py") == "src/utils/helper.py"


# ---------------------------------------------------------------------------
# parse_markdown_file + process_functions: integration with real markdown
# ---------------------------------------------------------------------------

class TestParserIntegration:
    """Verify that parsing markdown with dirty paths produces clean JSONL fields."""

    SAMPLE_MD = textwrap.dedent(r"""
        # RAGAnything Core Function Description

        # FUNCTION: process_with_rag

        ## Function Overview
        An end-to-end example function.

        ## Function Signature
        async def process_with_rag(file_path: str) -> None

        ## Input Parameters
        - file_path: str

        ## Detailed Description
        Processes a document.

        ## Output
        No return value.

        ## Function Implementation
        code\raganything_example.py:line 90-245

        ## Test Code
        code\test_code\my_test_process_with_rag.py

        ---
    """).lstrip()

    SAMPLE_MD_BACKTICK = textwrap.dedent("""
        # Core Functions

        # FUNCTION: initialize_rag_storage

        ## Function Overview
        Initializes storage.

        ## Function Signature
        async def initialize_rag_storage() -> None

        ## Input Parameters
        None

        ## Detailed Description
        Sets up storage backend.

        ## Output
        None

        ## Function Implementation
        `code\\src\\services\\rag_anything_service.py:line 46-154`

        ## Test Code
        `code\\test_code\\my_test_initialize_rag_storage.py`

        ---
    """).lstrip()

    def _parse_and_process(self, md_content: str, code_base: str = "/tmp/fake") -> list:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(md_content)
            f.flush()
            md_path = f.name

        try:
            raw = parse_markdown_file(md_path)
            return process_functions(raw, code_base, "/tmp/fake_tests")
        finally:
            os.unlink(md_path)

    def test_backslash_paths_normalized(self):
        result = self._parse_and_process(self.SAMPLE_MD)
        assert len(result) == 1
        rec = result[0]
        assert rec["implementation_location"] == "raganything_example.py:line 90-245"
        assert rec["test_code_path"] == "test_code/my_test_process_with_rag.py"

    def test_backtick_paths_normalized(self):
        result = self._parse_and_process(self.SAMPLE_MD_BACKTICK)
        assert len(result) == 1
        rec = result[0]
        assert rec["implementation_location"] == "src/services/rag_anything_service.py:line 46-154"
        assert rec["test_code_path"] == "test_code/my_test_initialize_rag_storage.py"

    def test_no_code_prefix_in_paths(self):
        """Neither implementation_location nor test_code_path should start with 'code/'."""
        for md in [self.SAMPLE_MD, self.SAMPLE_MD_BACKTICK]:
            result = self._parse_and_process(md)
            for rec in result:
                assert not rec["implementation_location"].startswith("code/"), \
                    f"implementation_location still has code/ prefix: {rec['implementation_location']}"
                assert not rec["test_code_path"].startswith("code/"), \
                    f"test_code_path still has code/ prefix: {rec['test_code_path']}"

    def test_no_backslashes_in_paths(self):
        """No backslashes should survive normalization."""
        for md in [self.SAMPLE_MD, self.SAMPLE_MD_BACKTICK]:
            result = self._parse_and_process(md)
            for rec in result:
                assert "\\" not in rec["implementation_location"]
                assert "\\" not in rec["test_code_path"]


# ---------------------------------------------------------------------------
# Evaluator path normalization (inline, not via _clean_location)
# ---------------------------------------------------------------------------

class TestEvaluatorPathNormalization:
    """Verify the evaluator's runtime normalization of JSONL record fields.

    The evaluator normalizes at read-time: strip backticks, backslash -> slash.
    Then it strips the 'code/' prefix before joining with source_dir.
    We simulate the exact logic from execution_evaluation_pure.py.
    """

    @staticmethod
    def _normalize_like_evaluator(raw: str) -> str:
        """Replicate the evaluator's normalization logic."""
        return raw.strip().strip('`').replace('\\', '/')

    @staticmethod
    def _strip_code_prefix(path: str) -> str:
        if path.startswith('code/'):
            return path[5:]
        return path

    @pytest.mark.parametrize("raw,expected_source,expected_test", [
        # Backslash (BookWorm)
        (
            r"code\raganything_example.py:line 90-245",
            "raganything_example.py",
            None,
        ),
        # Backtick + backslash (law-unleashed-rag)
        (
            r"`code\src\services\rag_anything_service.py:line 46-154`",
            "src/services/rag_anything_service.py",
            None,
        ),
        # Already clean (verl)
        (
            "code/recipe/prime/prime_core_algos.py:line 21-79",
            "recipe/prime/prime_core_algos.py",
            None,
        ),
    ])
    def test_source_file_resolution(self, raw, expected_source, expected_test):
        normalized = self._normalize_like_evaluator(raw)
        source_file = normalized.split(':')[0]
        source_file = self._strip_code_prefix(source_file)
        assert source_file == expected_source

    @pytest.mark.parametrize("raw,expected", [
        (r"code\test_code\my_test_process_with_rag.py", "test_code/my_test_process_with_rag.py"),
        (r"`code\test_code\my_test_initialize_rag_storage.py`", "test_code/my_test_initialize_rag_storage.py"),
        ("code/tests/test_prime.py", "tests/test_prime.py"),
    ])
    def test_test_path_resolution(self, raw, expected):
        normalized = self._normalize_like_evaluator(raw)
        test_path = self._strip_code_prefix(normalized)
        assert test_path == expected

    def test_no_double_code_prefix(self):
        """The classic bug: source_dir already ends in /code, path starts with code\\."""
        source_dir = "/workspace/project/raganything/test_examples/BookWorm/code"
        raw_location = r"code\raganything_example.py:line 90-245"

        normalized = self._normalize_like_evaluator(raw_location)
        source_file = normalized.split(':')[0]
        source_file = self._strip_code_prefix(source_file)

        full_path = os.path.join(source_dir, source_file)
        # Should NOT contain code/code
        assert "code/code" not in full_path
        assert full_path == "/workspace/project/raganything/test_examples/BookWorm/code/raganything_example.py"
