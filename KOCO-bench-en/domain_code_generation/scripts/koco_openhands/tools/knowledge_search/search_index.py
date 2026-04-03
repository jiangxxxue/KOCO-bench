"""SQLite FTS5 + embedding search index.

Port of OpenClaw's ``manager-search.ts`` and ``memory-schema.ts``.
Uses Python's built-in ``sqlite3`` with FTS5 for keyword search and
in-memory cosine similarity for vector search (OpenClaw's fallback path
when ``sqlite-vec`` is unavailable — equivalent for small corpora).
"""

import os
import sqlite3

from tools.knowledge_search.chunker import Chunk, chunk_file
from tools.knowledge_search.hybrid import bm25_rank_to_score, build_fts_query

EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 64
MAX_FILE_SIZE = 512 * 1024  # 512 KB

# Known text extensions — always indexed
TEXT_EXTENSIONS = {
    # code
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".css", ".html",
    ".cpp", ".h", ".cu", ".c",
    # docs
    ".md", ".rst", ".txt",
    # config
    ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".conf",
    # scripts
    ".sh", ".bash", ".slurm",
    # templates & misc
    ".in", ".jinja", ".example",
    # notebooks
    ".ipynb",
}

# Known binary extensions — always skipped
BINARY_EXTENSIONS = {
    # images / media
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".mp4", ".wav",
    ".webm", ".bmp", ".tiff",
    # fonts
    ".ttf", ".woff", ".woff2", ".eot",
    # binary documents
    ".pdf", ".docx", ".xlsx", ".pptx",
    # data / models
    ".parquet", ".safetensors", ".onnx", ".pickle", ".pkl",
    ".bin", ".pt", ".pth", ".npz", ".npy", ".h5", ".hdf5",
    # compiled / archives
    ".pyc", ".pyo", ".so", ".dylib", ".o", ".a",
    ".jar", ".class", ".egg", ".whl",
    ".zip", ".tar", ".gz", ".bz2", ".xz",
    # other binary
    ".encrypted", ".lock",
}

# Extensionless filenames that are text — always indexed
TEXT_BASENAMES = {"Dockerfile", "Makefile", "LICENSE", "NOTICE"}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity (no numpy required)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0


class SearchIndex:
    """SQLite FTS5 + embedding index over a file corpus."""

    def __init__(self):
        self.db: sqlite3.Connection | None = None
        self.chunks: dict[str, Chunk] = {}
        self.embeddings: dict[str, list[float]] = {}
        self._embed_fn = None  # set during build if API key available

    def build(
        self,
        corpus_dirs: str | list[str],
        embed: bool = True,
    ) -> None:
        """Walk one or more directories, chunk files, build FTS5 + embedding index.

        File selection uses a three-tier strategy:
        1. White-list (``TEXT_EXTENSIONS``) — always indexed.
        2. Black-list (``BINARY_EXTENSIONS``) — always skipped.
        3. Grey zone — attempt a UTF-8 read; indexed only if decodable.

        Files larger than ``MAX_FILE_SIZE`` (512 KB) are always skipped.
        """
        if isinstance(corpus_dirs, str):
            corpus_dirs = [corpus_dirs]

        # Collect and chunk files
        all_chunks: list[Chunk] = []
        for corpus_dir in corpus_dirs:
            if not corpus_dir or not os.path.isdir(corpus_dir):
                continue
            for root, _dirs, files in os.walk(corpus_dir):
                for fname in files:
                    abs_path = os.path.join(root, fname)

                    # Size guard
                    try:
                        if os.path.getsize(abs_path) > MAX_FILE_SIZE:
                            continue
                    except OSError:
                        continue

                    ext = os.path.splitext(fname)[1].lower()

                    if ext in BINARY_EXTENSIONS:
                        continue

                    if ext not in TEXT_EXTENSIONS and fname not in TEXT_BASENAMES:
                        # Grey zone: try reading as UTF-8
                        try:
                            with open(abs_path, "r", encoding="utf-8") as f:
                                content = f.read()
                        except (OSError, UnicodeDecodeError):
                            continue
                    else:
                        try:
                            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                                content = f.read()
                        except OSError:
                            continue

                    rel_path = os.path.relpath(abs_path, corpus_dir)
                    file_chunks = chunk_file(rel_path, content=content)
                    all_chunks.extend(file_chunks)

        if not all_chunks:
            return

        # Build SQLite FTS5 index
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                path TEXT,
                start_line INTEGER,
                end_line INTEGER,
                text TEXT
            )
        """)
        self.db.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text,
                id UNINDEXED,
                path UNINDEXED,
                start_line UNINDEXED,
                end_line UNINDEXED
            )
        """)

        for chunk in all_chunks:
            self.chunks[chunk.chunk_id] = chunk
            self.db.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk.chunk_id, chunk.file_path, chunk.start_line,
                 chunk.end_line, chunk.text),
            )
            self.db.execute(
                "INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?)",
                (chunk.text, chunk.chunk_id, chunk.file_path,
                 chunk.start_line, chunk.end_line),
            )
        self.db.commit()

        # Build embeddings if API key is available
        if embed:
            self._build_embeddings(all_chunks)

    def _build_embeddings(self, chunks: list[Chunk]) -> None:
        """Compute embeddings via litellm in batches."""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return

        try:
            import litellm
        except ImportError:
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]

        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch_texts = texts[i:i + EMBED_BATCH_SIZE]
            batch_ids = ids[i:i + EMBED_BATCH_SIZE]
            try:
                response = litellm.embedding(
                    model=EMBEDDING_MODEL,
                    input=batch_texts,
                    api_key=api_key,
                )
                for j, item in enumerate(response.data):
                    self.embeddings[batch_ids[j]] = item["embedding"]
            except Exception:
                # Embedding failure is non-fatal — fall back to BM25-only
                break

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using litellm."""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        import litellm
        response = litellm.embedding(
            model=EMBEDDING_MODEL,
            input=texts,
            api_key=api_key,
        )
        return [item["embedding"] for item in response.data]

    def search_keyword(
        self, query: str, limit: int = 24,
    ) -> list[tuple[str, float, dict]]:
        """BM25 keyword search via FTS5.

        Returns ``[(chunk_id, score, info), ...]`` sorted by score descending.
        """
        if self.db is None:
            return []
        fts_query = build_fts_query(query)
        if not fts_query:
            return []
        # Column weights: 1.0 for text, 0.0 for UNINDEXED columns (id, path,
        # start_line, end_line).  Without explicit weights, FTS5 averages
        # across all columns, diluting the score from the indexed text column.
        rows = self.db.execute("""
            SELECT id, path, start_line, end_line, text,
                   bm25(chunks_fts, 1.0, 0.0, 0.0, 0.0, 0.0) AS rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
        """, (fts_query, limit)).fetchall()

        results = []
        for row in rows:
            info = {
                "path": row["path"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "snippet": row["text"],
            }
            results.append((row["id"], bm25_rank_to_score(row["rank"]), info))
        return results

    def search_vector(
        self, query: str, limit: int = 24,
    ) -> list[tuple[str, float, dict]]:
        """Semantic vector search via in-memory cosine similarity.

        Returns ``[(chunk_id, score, info), ...]`` sorted by score descending.
        """
        if not self.embeddings:
            return []
        query_embedding = self._embed([query])[0]
        scores = []
        for chunk_id, embedding in self.embeddings.items():
            sim = cosine_similarity(query_embedding, embedding)
            chunk = self.chunks[chunk_id]
            info = {
                "path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "snippet": chunk.text,
            }
            scores.append((chunk_id, sim, info))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    @property
    def has_embeddings(self) -> bool:
        return bool(self.embeddings)
