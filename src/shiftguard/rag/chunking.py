"""Heading-aware chunker for policy Markdown.

Policy docs are authored with an H1 title and one H2 per rule. We split on H2
headings so each chunk is a *self-contained rule* (e.g. "Overtime Threshold"),
carrying `doc`/`section` metadata for citations. Sections longer than the token
cap fall back to sentence-boundary splitting — never blind fixed-size cuts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_TOKENS = 512

_H1 = re.compile(r"^#\s+(.*)$")
_H2 = re.compile(r"^##\s+(.*)$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # stable, human-readable id e.g. "overtime:overtime-threshold"
    doc: str       # document title (H1) e.g. "Overtime Policy"
    section: str   # section heading (H2) e.g. "Overtime Threshold"
    source: str    # source filename e.g. "overtime.md"
    text: str      # embeddable/displayable content (doc + section header + body)

    def payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc": self.doc,
            "section": self.section,
            "source": self.source,
            "text": self.text,
        }


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — avoids a tokenizer dependency."""
    return max(1, len(text) // 4)


def _slug(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")


def _split_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENTENCE_SPLIT.split(text.strip())) if s]


def _pack_sentences(sentences: list[str], overhead: int, max_tokens: int) -> list[str]:
    """Greedily pack sentences into bodies that stay under the token cap."""
    bodies: list[str] = []
    current: list[str] = []
    budget = overhead
    for sentence in sentences:
        cost = estimate_tokens(sentence)
        if current and budget + cost > max_tokens:
            bodies.append(" ".join(current))
            current = [sentence]
            budget = overhead + cost
        else:
            current.append(sentence)
            budget += cost
    if current:
        bodies.append(" ".join(current))
    return bodies


def _parse_sections(md: str, source: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (doc_title, [(section_title, body), ...]) from Markdown."""
    doc_title = Path(source).stem
    sections: list[tuple[str, list[str]]] = []
    preamble: list[str] = []
    current_title: str | None = None
    current_body: list[str] = []

    for line in md.splitlines():
        h2 = _H2.match(line)
        if h2:
            if current_title is not None:
                sections.append((current_title, current_body))
            current_title = h2.group(1).strip()
            current_body = []
            continue
        h1 = _H1.match(line)
        if h1:
            doc_title = h1.group(1).strip()
            continue
        (current_body if current_title is not None else preamble).append(line)

    if current_title is not None:
        sections.append((current_title, current_body))

    ordered: list[tuple[str, str]] = []
    pre_text = "\n".join(preamble).strip()
    if pre_text:
        ordered.append(("Overview", pre_text))
    ordered.extend((title, "\n".join(body).strip()) for title, body in sections)
    return doc_title, ordered


def chunk_markdown(md: str, source: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    doc_title, sections = _parse_sections(md, source)
    stem = Path(source).stem
    chunks: list[Chunk] = []

    for section_title, body in sections:
        if not body:
            continue
        header = f"{doc_title} > {section_title}"
        prefix = f"{header}\n\n"
        overhead = estimate_tokens(prefix)
        full_text = prefix + body

        if estimate_tokens(full_text) <= max_tokens:
            bodies = [body]
        else:
            bodies = _pack_sentences(_split_sentences(body), overhead, max_tokens)

        base_id = f"{stem}:{_slug(section_title)}"
        for i, piece in enumerate(bodies):
            chunk_id = base_id if len(bodies) == 1 else f"{base_id}#{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc=doc_title,
                    section=section_title,
                    source=source,
                    text=f"{prefix}{piece}",
                )
            )
    return chunks


def chunk_file(path: Path, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    return chunk_markdown(path.read_text(encoding="utf-8"), source=path.name, max_tokens=max_tokens)


def chunk_corpus(policies_dir: Path, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(policies_dir).glob("*.md")):
        chunks.extend(chunk_file(path, max_tokens=max_tokens))
    return chunks
