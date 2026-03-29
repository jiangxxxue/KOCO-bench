"""Tests for the knowledge_search tool components."""

import math
import os
import tempfile

import pytest

# Ensure the openhands/ directory is on sys.path so imports resolve
import sys
from pathlib import Path
_OPENHANDS_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _OPENHANDS_DIR not in sys.path:
    sys.path.insert(0, _OPENHANDS_DIR)

from tools.knowledge_search.hybrid import (
    build_fts_query,
    bm25_rank_to_score,
    extract_keywords,
    merge_hybrid_results,
)
from tools.knowledge_search.chunker import Chunk, chunk_file
from tools.knowledge_search.search_index import SearchIndex, cosine_similarity


# ═══════════════════════════════════════════════════════════════════════════
# build_fts_query
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildFtsQuery:
    def test_basic_tokens(self):
        result = build_fts_query("hello world")
        assert result == '"hello" AND "world"'

    def test_single_token(self):
        assert build_fts_query("torch") == '"torch"'

    def test_strips_double_quotes(self):
        result = build_fts_query('say "hello" world')
        assert '"' not in result.replace('" AND "', "").replace('"', "X")
        # Each token should be quoted, but internal quotes stripped
        assert result == '"say" AND "hello" AND "world"'

    def test_unicode_tokens(self):
        result = build_fts_query("函数 名前")
        assert result == '"函数" AND "名前"'

    def test_empty_returns_none(self):
        assert build_fts_query("") is None
        assert build_fts_query("   ") is None
        assert build_fts_query("!@#$%") is None

    def test_mixed_alphanum_and_symbols(self):
        result = build_fts_query("def compute_score(self):")
        assert '"def"' in result
        assert '"compute_score"' in result
        assert '"self"' in result


# ═══════════════════════════════════════════════════════════════════════════
# bm25_rank_to_score
# ═══════════════════════════════════════════════════════════════════════════

class TestBm25RankToScore:
    def test_negative_rank(self):
        # FTS5 returns negative ranks; more negative = more relevant
        score = bm25_rank_to_score(-5.0)
        assert 0 < score < 1
        assert abs(score - 5.0 / 6.0) < 1e-9

    def test_zero_rank(self):
        score = bm25_rank_to_score(0.0)
        assert score == 1.0 / (1.0 + 0.0)

    def test_positive_rank(self):
        score = bm25_rank_to_score(3.0)
        assert score == 1.0 / 4.0

    def test_infinite_rank(self):
        score = bm25_rank_to_score(float("inf"))
        assert score == 1 / 1000

    def test_negative_infinite(self):
        score = bm25_rank_to_score(float("-inf"))
        assert score == 1 / 1000

    def test_nan(self):
        score = bm25_rank_to_score(float("nan"))
        assert score == 1 / 1000

    def test_monotonicity(self):
        """More negative rank (stronger match) should produce higher score."""
        s1 = bm25_rank_to_score(-10.0)
        s2 = bm25_rank_to_score(-5.0)
        s3 = bm25_rank_to_score(-1.0)
        assert s1 > s2 > s3


# ═══════════════════════════════════════════════════════════════════════════
# merge_hybrid_results
# ═══════════════════════════════════════════════════════════════════════════

class TestMergeHybridResults:
    def test_weighted_fusion(self):
        vec = [("a", 0.9, {"path": "a.py", "snippet": "vec"})]
        kw = [("a", 0.8, {"path": "a.py", "snippet": "kw"})]
        merged = merge_hybrid_results(vec, kw, vector_weight=0.7, text_weight=0.3)
        assert len(merged) == 1
        chunk_id, score, info = merged[0]
        assert chunk_id == "a"
        expected = 0.7 * 0.9 + 0.3 * 0.8
        assert abs(score - expected) < 1e-9
        # Keyword snippet preferred when same chunk in both
        assert info["snippet"] == "kw"

    def test_union_merge(self):
        vec = [("a", 0.9, {"path": "a.py", "snippet": "a"})]
        kw = [("b", 0.8, {"path": "b.py", "snippet": "b"})]
        merged = merge_hybrid_results(vec, kw)
        ids = {m[0] for m in merged}
        assert ids == {"a", "b"}

    def test_empty_inputs(self):
        assert merge_hybrid_results([], []) == []
        assert len(merge_hybrid_results([("a", 0.5, {})], [])) == 1
        assert len(merge_hybrid_results([], [("b", 0.5, {})])) == 1

    def test_sorted_by_score_desc(self):
        vec = [("a", 0.9, {}), ("b", 0.1, {})]
        kw = [("c", 0.5, {})]
        merged = merge_hybrid_results(vec, kw)
        scores = [m[1] for m in merged]
        assert scores == sorted(scores, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# extract_keywords
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractKeywords:
    def test_stop_word_removal(self):
        result = extract_keywords("what is the compute score function")
        assert "what" not in result
        assert "is" not in result
        assert "the" not in result
        assert "compute" in result
        assert "score" in result
        assert "function" in result

    def test_empty_input(self):
        assert extract_keywords("") == []
        assert extract_keywords("the a an is") == []

    def test_single_char_filtered(self):
        result = extract_keywords("a b c hello")
        assert result == ["hello"]

    def test_lowercase(self):
        result = extract_keywords("ComputeScore")
        assert "computescore" in result


# ═══════════════════════════════════════════════════════════════════════════
# chunk_file
# ═══════════════════════════════════════════════════════════════════════════

class TestChunkFile:
    def test_basic_chunking(self):
        content = "\n".join(f"line {i}" for i in range(100))
        chunks = chunk_file("test.py", content=content, tokens=10, overlap=2)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].start_line == 1
        assert chunks[0].file_path == "test.py"

    def test_chunk_id_format(self):
        # "hello\nworld" has 2 lines; trailing \n adds a 3rd empty line
        content = "hello\nworld"
        chunks = chunk_file("test.py", content=content, tokens=100, overlap=0)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "test.py:1-2"

    def test_min_chars_lower_bound(self):
        """max_chars should be at least 32 (matching OpenClaw)."""
        content = "a" * 100
        # tokens=1 → chars would be 4, but clamped to 32
        chunks = chunk_file("test.py", content=content, tokens=1, overlap=0)
        assert len(chunks) > 0
        assert len(chunks[0].text) <= 32

    def test_overlap(self):
        """Chunks should have overlapping content."""
        lines = [f"line_{i:03d}" for i in range(50)]
        content = "\n".join(lines)
        chunks = chunk_file("test.py", content=content, tokens=10, overlap=3)
        if len(chunks) >= 2:
            # Last lines of chunk 0 should appear in chunk 1
            c0_lines = set(chunks[0].text.split("\n")[-3:])
            c1_lines = set(chunks[1].text.split("\n")[:5])
            assert c0_lines & c1_lines, "Expected overlap between consecutive chunks"

    def test_empty_file(self):
        assert chunk_file("test.py", content="") == []
        assert chunk_file("test.py", content="   \n  \n") == []

    def test_long_line_splitting(self):
        """Lines exceeding max_chars are broken into segments."""
        long_line = "x" * 200
        chunks = chunk_file("test.py", content=long_line, tokens=10, overlap=0)
        # max_chars = max(32, 10*4) = 40, so 200 chars should produce multiple chunks
        assert len(chunks) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# SearchIndex
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchIndex:
    @pytest.fixture
    def corpus_dir(self, tmp_path):
        """Create a small test corpus."""
        (tmp_path / "module.py").write_text(
            "def compute_score(predictions, labels):\n"
            "    '''Compute accuracy score.'''\n"
            "    correct = sum(p == l for p, l in zip(predictions, labels))\n"
            "    return correct / len(labels)\n"
        )
        (tmp_path / "README.md").write_text(
            "# Framework Docs\n\n"
            "## compute_score\n\n"
            "This function computes the accuracy score by comparing\n"
            "predictions against ground truth labels.\n"
        )
        (tmp_path / "config.yaml").write_text("key: value\n")
        return str(tmp_path)

    def test_build_and_keyword_search(self, corpus_dir):
        idx = SearchIndex()
        idx.build(corpus_dir, embed=False)
        results = idx.search_keyword("compute_score", limit=10)
        assert len(results) > 0
        paths = {r[2]["path"] for r in results}
        assert "module.py" in paths

    def test_yaml_now_indexed(self, corpus_dir):
        """YAML is in TEXT_EXTENSIONS and should be indexed."""
        idx = SearchIndex()
        idx.build(corpus_dir, embed=False)
        results = idx.search_keyword("key value", limit=5)
        paths = {info["path"] for _, _, info in results}
        assert "config.yaml" in paths

    def test_fts_returns_score(self, corpus_dir):
        idx = SearchIndex()
        idx.build(corpus_dir, embed=False)
        results = idx.search_keyword("accuracy score", limit=5)
        for chunk_id, score, info in results:
            assert 0 < score <= 1

    def test_no_results_for_absent_term(self, corpus_dir):
        idx = SearchIndex()
        idx.build(corpus_dir, embed=False)
        results = idx.search_keyword("xyznonexistent", limit=5)
        assert results == []

    def test_binary_excluded(self, tmp_path):
        """Binary files (e.g. .png) should never be indexed."""
        (tmp_path / "code.py").write_text("def hello(): pass\n")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        idx = SearchIndex()
        idx.build(str(tmp_path), embed=False)
        results = idx.search_keyword("hello", limit=10)
        paths = {info["path"] for _, _, info in results}
        assert "code.py" in paths
        assert "image.png" not in paths

    def test_grey_zone_text_file_indexed(self, tmp_path):
        """Files with unknown extension but valid UTF-8 should be indexed."""
        (tmp_path / "config.myext").write_text("special_setting = true\n")
        idx = SearchIndex()
        idx.build(str(tmp_path), embed=False)
        results = idx.search_keyword("special_setting", limit=5)
        assert len(results) > 0

    def test_grey_zone_binary_file_skipped(self, tmp_path):
        """Files with unknown extension and invalid UTF-8 should be skipped."""
        (tmp_path / "data.mybin").write_bytes(b"\x80\x81\x82\xff\xfe" * 100)
        (tmp_path / "code.py").write_text("def marker(): pass\n")
        idx = SearchIndex()
        idx.build(str(tmp_path), embed=False)
        paths = {info["path"] for _, _, info in idx.search_keyword("marker", limit=10)}
        assert "code.py" in paths

    def test_extensionless_dockerfile_indexed(self, tmp_path):
        """Dockerfile (no extension) should be indexed via TEXT_BASENAMES."""
        (tmp_path / "Dockerfile").write_text("FROM python:3.12\nRUN pip install torch\n")
        idx = SearchIndex()
        idx.build(str(tmp_path), embed=False)
        results = idx.search_keyword("torch", limit=5)
        paths = {info["path"] for _, _, info in results}
        assert "Dockerfile" in paths

    def test_large_file_skipped(self, tmp_path):
        """Files exceeding MAX_FILE_SIZE should be skipped."""
        (tmp_path / "huge.py").write_text("x = 1\n" * 200_000)  # > 512 KB
        (tmp_path / "small.py").write_text("def tiny(): pass\n")
        idx = SearchIndex()
        idx.build(str(tmp_path), embed=False)
        paths = {info["path"] for _, _, info in idx.search_keyword("tiny", limit=10)}
        assert "small.py" in paths

    def test_multi_directory_build(self, tmp_path):
        """build() should accept a list of directories."""
        dir_a = tmp_path / "corpus"
        dir_a.mkdir()
        (dir_a / "api.py").write_text("def api_call(): pass\n")

        dir_b = tmp_path / "code"
        dir_b.mkdir()
        (dir_b / "app.py").write_text("def app_main(): pass\n")

        idx = SearchIndex()
        idx.build([str(dir_a), str(dir_b)], embed=False)
        paths_api = {info["path"] for _, _, info in idx.search_keyword("api_call", limit=10)}
        paths_app = {info["path"] for _, _, info in idx.search_keyword("app_main", limit=10)}
        assert "api.py" in paths_api
        assert "app.py" in paths_app

    def test_has_embeddings_false_without_api_key(self, corpus_dir, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        idx = SearchIndex()
        idx.build(corpus_dir, embed=True)
        assert not idx.has_embeddings


# ═══════════════════════════════════════════════════════════════════════════
# KnowledgeSearchExecutor (end-to-end, BM25-only)
# ═══════════════════════════════════════════════════════════════════════════

class TestKnowledgeSearchExecutor:
    @pytest.fixture
    def corpus_dir(self, tmp_path):
        # Need multiple diverse files so BM25 IDF is meaningful (a term
        # appearing in 1/N docs gets a meaningful IDF score, vs 1/1 = 0).
        (tmp_path / "loss.py").write_text(
            "def compute_loss(pred, target):\n"
            "    '''Compute MSE loss between predictions and targets.'''\n"
            "    return sum((p - t) ** 2 for p, t in zip(pred, target))\n"
        )
        (tmp_path / "helper.py").write_text(
            "def helper_function(x):\n"
            "    '''Double the input value.'''\n"
            "    return x * 2\n"
        )
        (tmp_path / "trainer.py").write_text(
            "class Trainer:\n"
            "    def __init__(self, model, optimizer):\n"
            "        self.model = model\n"
            "        self.optimizer = optimizer\n"
            "\n"
            "    def train_step(self, batch):\n"
            "        output = self.model(batch)\n"
            "        return output\n"
        )
        (tmp_path / "config.py").write_text(
            "DEFAULT_LR = 0.001\n"
            "DEFAULT_EPOCHS = 10\n"
            "DEFAULT_BATCH_SIZE = 32\n"
        )
        (tmp_path / "docs.md").write_text(
            "# API Reference\n\n"
            "## compute_loss\n"
            "Computes MSE loss between predictions and targets.\n"
        )
        return str(tmp_path)

    def test_end_to_end_search(self, corpus_dir, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Import here to avoid SDK import issues at module level
        from tools.knowledge_search.definition import KnowledgeSearchExecutor, KnowledgeSearchAction
        executor = KnowledgeSearchExecutor(corpus_dirs=corpus_dir)
        action = KnowledgeSearchAction(query="compute_loss", max_results=6, min_score=0.1)
        obs = executor(action)
        assert obs.num_results > 0
        assert obs.query == "compute_loss"
        assert any("compute_loss" in r["snippet"] for r in obs.results)

    def test_min_score_filtering(self, corpus_dir, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from tools.knowledge_search.definition import KnowledgeSearchExecutor, KnowledgeSearchAction
        executor = KnowledgeSearchExecutor(corpus_dirs=corpus_dir)
        # Very high min_score should filter out everything
        action = KnowledgeSearchAction(query="compute_loss", max_results=6, min_score=0.99)
        obs = executor(action)
        assert obs.num_results == 0

    def test_file_type_filter(self, corpus_dir, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from tools.knowledge_search.definition import KnowledgeSearchExecutor, KnowledgeSearchAction
        executor = KnowledgeSearchExecutor(corpus_dirs=corpus_dir)
        action = KnowledgeSearchAction(query="compute_loss", max_results=6, min_score=0.1, file_type="py")
        obs = executor(action)
        for r in obs.results:
            assert r["path"].endswith(".py")

    def test_bm25_only_query_expansion(self, corpus_dir, monkeypatch):
        """BM25-only mode uses keyword extraction for conversational queries."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from tools.knowledge_search.definition import KnowledgeSearchExecutor, KnowledgeSearchAction
        executor = KnowledgeSearchExecutor(corpus_dirs=corpus_dir)
        # Conversational query — stop words should be stripped
        action = KnowledgeSearchAction(
            query="what is the helper function",
            max_results=6,
            min_score=0.1,
        )
        obs = executor(action)
        assert obs.num_results > 0


# ═══════════════════════════════════════════════════════════════════════════
# cosine_similarity
# ═══════════════════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-9

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
