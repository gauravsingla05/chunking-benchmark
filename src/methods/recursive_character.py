from __future__ import annotations

"""
Author: Gourav Singla
Date: 2026-05
Description: Recursive character splitting in the LangChain default style.
Paper Inspiration: Most-deployed production baseline for document chunking
(LangChain `RecursiveCharacterTextSplitter`). Added in response to reviewer
feedback that the original four-method comparison omitted this baseline.

Method:
  1. Split the document into chunks by trying separators in hierarchical
     order: paragraph (``\\n\\n``) -> line (``\\n``) -> sentence boundaries
     (``. ``, ``? ``, ``! ``) -> word (`` ``) -> character.
  2. Pack chunks so each chunk is at most `chunk_chars` characters with
     `overlap_chars` of overlap between consecutive chunks. Defaults follow
     the LangChain defaults (1000 / 200).
  3. For content generation (no retrieval query), select chunks sequentially
     from the start of the document until the word budget is exhausted.
     Avoid counting the overlap region twice.

This is the natural content-generation analogue of LangChain's retrieval
default: split the document into recursive-character chunks, then take
chunks in document order up to the word budget.
"""

from .registry import register_method

DEFAULT_CHUNK_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 200
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]


def _recursive_split(text: str, chunk_chars: int, separators: list[str]) -> list[str]:
    """Split text into roughly-chunk_chars pieces using the separator hierarchy.

    Mirrors the LangChain RecursiveCharacterTextSplitter behaviour: try the
    largest separator first; if a piece is still longer than chunk_chars,
    recurse into smaller separators.
    """
    if len(text) <= chunk_chars or not separators:
        return [text]

    sep = separators[0]
    rest = separators[1:]

    if sep == "":
        # Last-resort: hard split by character.
        return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]

    parts = text.split(sep)
    # Reattach the separator to all but the last part so the join is lossless.
    if sep:
        parts = [p + sep for p in parts[:-1]] + [parts[-1]] if parts else []

    out: list[str] = []
    for p in parts:
        if not p:
            continue
        if len(p) <= chunk_chars:
            out.append(p)
        else:
            out.extend(_recursive_split(p, chunk_chars, rest))
    return out


def _merge_with_overlap(pieces: list[str], chunk_chars: int, overlap_chars: int) -> list[str]:
    """Merge small pieces into chunks of size up to chunk_chars with overlap.

    Standard LangChain pack: greedy append; when adding the next piece would
    exceed chunk_chars, emit the current chunk, then carry overlap_chars of
    tail text into the next chunk's seed.
    """
    if not pieces:
        return []

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if not cur:
            cur = piece
            continue
        if len(cur) + len(piece) <= chunk_chars:
            cur += piece
        else:
            chunks.append(cur)
            tail = cur[-overlap_chars:] if overlap_chars and overlap_chars < len(cur) else cur
            cur = tail + piece
            if len(cur) > chunk_chars:
                # The new piece itself is huge; emit the carry and start fresh.
                chunks.append(cur[:chunk_chars])
                cur = cur[chunk_chars - overlap_chars :] if overlap_chars < chunk_chars else piece
    if cur:
        chunks.append(cur)
    return chunks


@register_method(
    "recursive_character",
    description="LangChain-default recursive character splitter; chunks packed under chunk_chars with overlap; selected sequentially from start under a word budget.",
)
def recursive_character(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""

    pieces = _recursive_split(text, DEFAULT_CHUNK_CHARS, DEFAULT_SEPARATORS)
    chunks = _merge_with_overlap(pieces, DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS)
    if not chunks:
        return ""

    # Sequentially pack chunks until the word budget is hit. Avoid double-counting
    # overlap by only including the *new* portion of each chunk after the first.
    selected_words: list[str] = []
    last_chunk_tail = ""
    for chunk in chunks:
        if last_chunk_tail and chunk.startswith(last_chunk_tail):
            new_text = chunk[len(last_chunk_tail) :]
        else:
            new_text = chunk
        new_words = new_text.split()
        if len(selected_words) + len(new_words) > max_words:
            remaining = max_words - len(selected_words)
            if remaining > 0:
                selected_words.extend(new_words[:remaining])
            break
        selected_words.extend(new_words)
        last_chunk_tail = chunk[-DEFAULT_OVERLAP_CHARS:] if DEFAULT_OVERLAP_CHARS else ""

    return " ".join(selected_words)
