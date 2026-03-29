"""OpenHands SDK tool definition for hybrid knowledge search.

Registers ``knowledge_search`` as a custom OpenHands tool that can be used
via ``Tool(name="knowledge_search", params={"corpus_dirs": ["/path1", "/path2"]})``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.sdk.tool.schema import TextContent

from tools.knowledge_search.hybrid import (
    extract_keywords,
    merge_hybrid_results,
)
from tools.knowledge_search.search_index import SearchIndex

if TYPE_CHECKING:
    from openhands.sdk.conversation import ConversationState

SNIPPET_MAX_CHARS = 700

TOOL_DESCRIPTION = """\
Search the framework knowledge base using hybrid BM25 + semantic search.
Returns ranked code snippets and documentation with file paths and line numbers.
Use this to find: function implementations, class definitions, API usage, config examples.
More efficient than manually browsing the repository with terminal commands."""


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------

class KnowledgeSearchAction(Action):
    query: str = Field(description="Search query: function name, concept, or code pattern")
    max_results: int = Field(default=6, ge=1, le=20, description="Max results (default: 6)")
    min_score: float = Field(default=0.35, ge=0.0, le=1.0, description="Min relevance score (default: 0.35)")
    file_type: str | None = Field(default=None, description="Filter: 'py' for code, 'md' for docs, None for all")


class KnowledgeSearchObservation(Observation):
    results: list[dict] = Field(default_factory=list, description="Search results")
    query: str = Field(default="", description="Original query")
    num_results: int = Field(default=0, description="Number of results returned")


def _format_results(results: list[dict], query: str) -> str:
    """Format search results for the LLM."""
    if not results:
        return f"No results found for query: {query}"
    lines = [f"Found {len(results)} result(s) for: {query}\n"]
    for i, r in enumerate(results, 1):
        snippet = r.get("snippet", "")
        if len(snippet) > SNIPPET_MAX_CHARS:
            snippet = snippet[:SNIPPET_MAX_CHARS] + "..."
        lines.append(
            f"--- Result {i}: {r['path']} (lines {r['start_line']}-{r['end_line']}) "
            f"[score: {r['score']:.3f}] ---"
        )
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

VECTOR_WEIGHT = 0.7
TEXT_WEIGHT = 0.3


class KnowledgeSearchExecutor(ToolExecutor[KnowledgeSearchAction, KnowledgeSearchObservation]):
    """Orchestrates hybrid search (port of OpenClaw's manager.search())."""

    def __init__(self, corpus_dirs: str | list[str] = ""):
        self.index = SearchIndex()
        if corpus_dirs:
            self.index.build(corpus_dirs)

    def __call__(
        self,
        action: KnowledgeSearchAction,
        conversation=None,
    ) -> KnowledgeSearchObservation:
        query = action.query
        max_results = action.max_results
        min_score = action.min_score
        file_type = action.file_type

        candidates = min(200, max(1, max_results * 4))

        if self.index.has_embeddings:
            results = self._search_hybrid(query, candidates, min_score, max_results)
        else:
            results = self._search_bm25_only(query, candidates, min_score, max_results)

        # Apply file_type filter
        if file_type and results:
            suffix = f".{file_type}" if not file_type.startswith(".") else file_type
            results = [r for r in results if r["path"].endswith(suffix)]

        # Truncate snippets
        for r in results:
            if len(r.get("snippet", "")) > SNIPPET_MAX_CHARS:
                r["snippet"] = r["snippet"][:SNIPPET_MAX_CHARS] + "..."

        formatted = _format_results(results, query)

        return KnowledgeSearchObservation(
            content=[TextContent(text=formatted)],
            results=results,
            query=query,
            num_results=len(results),
        )

    def _search_hybrid(
        self, query: str, candidates: int, min_score: float, max_results: int,
    ) -> list[dict]:
        """Path B: hybrid vector + keyword search."""
        # Per-keyword search (same strategy as _search_bm25_only) so that
        # multi-term queries don't require ALL terms in a single chunk.
        keywords = extract_keywords(query)
        if not keywords:
            keywords = [query]
        keyword_results_map: dict[str, tuple[float, dict]] = {}
        for term in keywords:
            try:
                hits = self.index.search_keyword(term, candidates)
            except Exception:
                continue
            for chunk_id, score, info in hits:
                if chunk_id not in keyword_results_map or score > keyword_results_map[chunk_id][0]:
                    keyword_results_map[chunk_id] = (score, info)
        keyword_results = [
            (cid, sc, info) for cid, (sc, info) in keyword_results_map.items()
        ]

        try:
            vector_results = self.index.search_vector(query, candidates)
        except Exception:
            vector_results = []

        merged = merge_hybrid_results(
            vector_results, keyword_results,
            vector_weight=VECTOR_WEIGHT, text_weight=TEXT_WEIGHT,
        )

        # Strict filter
        strict = [(cid, sc, info) for cid, sc, info in merged if sc >= min_score]

        # Relaxed fallback (matching OpenClaw): if strict filter removes all
        # results but keyword matches exist, relax threshold for keyword-only
        # matches to prevent BM25-only results (max score = textWeight * 1.0)
        # from being discarded when minScore > textWeight.
        if not strict and keyword_results:
            relaxed_threshold = min(min_score, TEXT_WEIGHT)
            strict = [(cid, sc, info) for cid, sc, info in merged if sc >= relaxed_threshold]

        results = []
        for chunk_id, score, info in strict[:max_results]:
            results.append({
                "path": info["path"],
                "start_line": info["start_line"],
                "end_line": info["end_line"],
                "score": round(score, 4),
                "snippet": info["snippet"],
            })
        return results

    def _search_bm25_only(
        self, query: str, candidates: int, min_score: float, max_results: int,
    ) -> list[dict]:
        """Path A: BM25-only search with keyword expansion."""
        keywords = extract_keywords(query)
        if not keywords:
            # Fall back to raw query
            keywords = [query]

        all_results: dict[str, tuple[float, dict]] = {}
        for term in keywords:
            try:
                hits = self.index.search_keyword(term, candidates)
            except Exception:
                continue
            for chunk_id, score, info in hits:
                if chunk_id not in all_results or score > all_results[chunk_id][0]:
                    all_results[chunk_id] = (score, info)

        # Sort by score, filter, take top
        sorted_results = sorted(all_results.items(), key=lambda x: x[1][0], reverse=True)
        results = []
        for chunk_id, (score, info) in sorted_results:
            if score < min_score:
                continue
            results.append({
                "path": info["path"],
                "start_line": info["start_line"],
                "end_line": info["end_line"],
                "score": round(score, 4),
                "snippet": info["snippet"],
            })
            if len(results) >= max_results:
                break
        return results

    def close(self) -> None:
        if self.index.db is not None:
            self.index.db.close()


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

class KnowledgeSearchTool(ToolDefinition[KnowledgeSearchAction, KnowledgeSearchObservation]):
    @classmethod
    def create(
        cls,
        conv_state: "ConversationState | None" = None,
        corpus_dirs: str | list[str] = "",
        **kwargs,
    ) -> Sequence["KnowledgeSearchTool"]:
        executor = KnowledgeSearchExecutor(corpus_dirs=corpus_dirs)
        return [cls(
            description=TOOL_DESCRIPTION,
            action_type=KnowledgeSearchAction,
            observation_type=KnowledgeSearchObservation,
            annotations=ToolAnnotations(
                title="knowledge_search",
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
            executor=executor,
        )]


register_tool(KnowledgeSearchTool.name, KnowledgeSearchTool)
