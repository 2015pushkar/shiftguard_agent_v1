"""Chunk-boundary tests for the heading-aware chunker (no LLM / no Qdrant)."""

from pathlib import Path

from shiftguard.config import get_settings
from shiftguard.rag.chunking import chunk_corpus, chunk_markdown, estimate_tokens

SAMPLE = """# Overtime Policy

## Overtime Threshold
Hours over 40 in a workweek are paid at 1.5x.

## Double Time
Hours over 60 in a workweek are paid at 2.0x.
"""


def test_splits_one_chunk_per_h2_section():
    chunks = chunk_markdown(SAMPLE, source="overtime.md")
    assert [c.section for c in chunks] == ["Overtime Threshold", "Double Time"]


def test_metadata_doc_section_and_source():
    chunks = chunk_markdown(SAMPLE, source="overtime.md")
    first = chunks[0]
    assert first.doc == "Overtime Policy"
    assert first.section == "Overtime Threshold"
    assert first.source == "overtime.md"
    assert first.chunk_id == "overtime:overtime-threshold"


def test_chunk_text_contains_header_and_body():
    chunks = chunk_markdown(SAMPLE, source="overtime.md")
    text = chunks[0].text
    assert "Overtime Policy > Overtime Threshold" in text
    assert "1.5x" in text


def test_no_empty_chunks_and_unique_ids():
    chunks = chunk_markdown(SAMPLE, source="overtime.md")
    assert all(c.text.strip() for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_long_section_falls_back_to_sentence_split():
    body = " ".join(f"Sentence number {i} about overtime rules." for i in range(60))
    md = f"# Long Policy\n\n## Big Rule\n{body}\n"
    small_cap = 30
    chunks = chunk_markdown(md, source="long.md", max_tokens=small_cap)

    assert len(chunks) > 1, "oversized section should split into multiple chunks"
    assert all(estimate_tokens(c.text) <= small_cap * 2 for c in chunks)
    # Sub-chunk ids are suffixed and unique; all belong to the same section.
    assert {c.section for c in chunks} == {"Big Rule"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # No body sentence is lost across the split.
    joined = " ".join(c.text for c in chunks)
    assert "Sentence number 0 " in joined and "Sentence number 59 " in joined


def test_short_section_is_not_split():
    chunks = chunk_markdown(SAMPLE, source="overtime.md", max_tokens=512)
    assert all("#" not in c.chunk_id for c in chunks)


def test_chunks_the_real_policy_corpus():
    policies_dir = get_settings().policies_dir
    chunks = chunk_corpus(policies_dir)
    sources = {c.source for c in chunks}
    assert {"overtime.md", "timekeeping.md", "rounding.md"} <= sources
    # The authorization rule that drives ticket creation must be retrievable.
    assert any(c.section == "Overtime Authorization" for c in chunks)
    assert len(chunks) >= 9
