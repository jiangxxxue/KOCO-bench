"""OpenClaw-style hybrid search algorithms.

Ports of OpenClaw's ``extensions/memory-core/src/memory/hybrid.ts``:
- FTS5 query building with Unicode tokenization
- BM25 rank-to-score normalization
- Weighted vector + keyword score fusion
- Keyword extraction for BM25-only mode
"""

import math

import regex


# ---------------------------------------------------------------------------
# FTS5 query building
# ---------------------------------------------------------------------------

def build_fts_query(raw: str) -> str | None:
    """Tokenize *raw* with Unicode regex and build an FTS5 MATCH expression.

    Each token is double-quoted to prevent FTS5 syntax interpretation.
    Tokens are joined with ``AND``.  Returns ``None`` if no tokens remain.
    """
    tokens = regex.findall(r'[\p{L}\p{N}_]+', raw)
    if not tokens:
        return None
    return " AND ".join(f'"{t.replace(chr(34), "")}"' for t in tokens)


# ---------------------------------------------------------------------------
# BM25 normalization
# ---------------------------------------------------------------------------

def bm25_rank_to_score(rank: float) -> float:
    """Convert an SQLite FTS5 ``bm25()`` rank value to a 0-1 score.

    FTS5 returns *negative* ranks where a more-negative value means higher
    relevance.  This mirrors OpenClaw's ``bm25RankToScore()``.
    """
    if not math.isfinite(rank):
        return 1 / (1 + 999)
    if rank < 0:
        relevance = -rank
        return relevance / (1 + relevance)
    return 1 / (1 + rank)


# ---------------------------------------------------------------------------
# Hybrid score fusion
# ---------------------------------------------------------------------------

def merge_hybrid_results(
    vector_results: list[tuple[str, float, dict]],
    keyword_results: list[tuple[str, float, dict]],
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
) -> list[tuple[str, float, dict]]:
    """Merge vector and keyword results using weighted score fusion.

    Each result is ``(chunk_id, score, chunk_info)``.  When the same chunk
    appears in both lists, the keyword snippet is preferred (BM25 snippets
    contain exact keyword matches, which are more useful for the agent).

    Weights are normalized to sum to 1.0.
    """
    total = vector_weight + text_weight
    if total <= 0:
        total = 1.0
    vw = vector_weight / total
    tw = text_weight / total

    merged: dict[str, tuple[float, dict]] = {}

    for chunk_id, score, info in vector_results:
        merged[chunk_id] = (vw * score, info)

    for chunk_id, score, info in keyword_results:
        if chunk_id in merged:
            old_score, _old_info = merged[chunk_id]
            # Keep keyword snippet (preferred), combine scores
            merged[chunk_id] = (old_score + tw * score, info)
        else:
            merged[chunk_id] = (tw * score, info)

    results = [(cid, sc, info) for cid, (sc, info) in merged.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Keyword extraction (BM25-only mode)
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "that", "this", "it",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "not", "but", "if", "be", "do", "does", "did", "has",
    "have", "had", "will", "would", "can", "could", "should", "may",
    "might", "shall", "must", "so", "than", "then", "there", "here",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "only", "own", "same", "too", "very",
    "just", "about", "above", "after", "again", "also", "am", "an",
    "any", "because", "been", "before", "being", "between", "during",
    "into", "its", "me", "my", "myself", "our", "ours", "out", "over",
    "through", "under", "until", "up", "we", "while", "you", "your",
}


def extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from *query* for BM25-only search.

    Tokenizes, lowercases, and removes English stop words.
    Used only in BM25-only mode to improve recall on conversational queries.
    """
    tokens = regex.findall(r'[\p{L}\p{N}_]+', query.lower())
    keywords = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return keywords if keywords else []
