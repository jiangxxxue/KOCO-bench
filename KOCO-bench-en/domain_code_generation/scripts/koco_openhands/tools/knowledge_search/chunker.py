"""OpenClaw-style line-based file chunking.

Port of OpenClaw's ``src/memory/internal.ts`` ``chunkMarkdown()``.
Works uniformly for all file types (.py, .md, .rst, .yaml).
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A contiguous block of text from a source file."""
    text: str
    file_path: str        # relative to corpus root
    start_line: int       # 1-indexed
    end_line: int         # 1-indexed, inclusive
    chunk_id: str = field(default="")

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"{self.file_path}:{self.start_line}-{self.end_line}"


def chunk_file(
    file_path: str,
    content: str | None = None,
    tokens: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """Split a file into overlapping chunks using line-based chunking.

    Parameters:
        file_path: Relative path (used for chunk IDs and metadata).
        content: File content.  If ``None``, reads from *file_path*.
        tokens: Approximate max tokens per chunk (chars = tokens * 4).
        overlap: Overlap in tokens carried to the next chunk.

    Returns a list of :class:`Chunk` objects.
    """
    if content is None:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

    if not content.strip():
        return []

    max_chars = max(32, tokens * 4)
    overlap_chars = overlap * 4

    raw_lines = content.split("\n")

    # Long-line splitting: break lines exceeding max_chars into segments
    lines: list[tuple[str, int]] = []  # (text, original_line_number)
    for i, line in enumerate(raw_lines):
        line_num = i + 1  # 1-indexed
        if len(line) > max_chars:
            for start in range(0, len(line), max_chars):
                lines.append((line[start:start + max_chars], line_num))
        else:
            lines.append((line, line_num))

    chunks: list[Chunk] = []
    current_lines: list[tuple[str, int]] = []
    current_chars = 0

    def _flush():
        nonlocal current_lines, current_chars
        if not current_lines:
            return
        text = "\n".join(seg for seg, _ in current_lines)
        start_line = current_lines[0][1]
        end_line = current_lines[-1][1]
        chunks.append(Chunk(
            text=text,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
        ))

        # Carry overlap_chars worth of trailing lines into next chunk
        carry: list[tuple[str, int]] = []
        carry_size = 0
        for seg, ln in reversed(current_lines):
            if carry_size + len(seg) > overlap_chars and carry:
                break
            carry.append((seg, ln))
            carry_size += len(seg)
        carry.reverse()

        current_lines = carry
        current_chars = carry_size

    for seg, line_num in lines:
        seg_size = len(seg)
        if current_chars + seg_size > max_chars and current_lines:
            _flush()
        current_lines.append((seg, line_num))
        current_chars += seg_size

    # Flush remaining
    _flush()

    return chunks
