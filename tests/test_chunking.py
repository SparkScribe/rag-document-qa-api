"""Unit tests for the recursive character chunker."""

import pytest

from app.core.config import Settings
from app.services.chunking import RecursiveCharacterChunker, TextChunk


@pytest.fixture
def chunker() -> RecursiveCharacterChunker:
    return RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def default_chunker() -> RecursiveCharacterChunker:
    return RecursiveCharacterChunker.from_settings(Settings())


def test_empty_text_returns_no_chunks(chunker: RecursiveCharacterChunker) -> None:
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\t  ") == []


def test_short_text_returns_single_chunk(chunker: RecursiveCharacterChunker) -> None:
    text = "Hello, RAG world."
    chunks = chunker.split_text(text)

    assert len(chunks) == 1
    assert chunks[0] == TextChunk(index=0, text=text)


def test_chunks_respect_max_size(default_chunker: RecursiveCharacterChunker) -> None:
    text = "Word. " * 500
    chunks = default_chunker.split_text(text)

    assert len(chunks) > 1
    for chunk in chunks:
        # Overlap prefix may push a chunk slightly above chunk_size; body still bounded.
        assert len(chunk.text) <= default_chunker.chunk_size + default_chunker.chunk_overlap


def test_chunk_indices_are_sequential(chunker: RecursiveCharacterChunker) -> None:
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three." * 10
    chunks = chunker.split_text(text)

    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_prefers_paragraph_boundaries() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=80, chunk_overlap=10)
    text = ("First paragraph content here.\n\n" * 3) + ("Second block of text.\n\n" * 3)
    chunks = chunker.split_text(text)

    assert len(chunks) >= 2
    assert any("\n\n" in c.text or c.text.endswith(".") for c in chunks)


def test_overlap_carries_context_between_chunks() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=15)
    text = "Alpha segment. " * 20
    chunks = chunker.split_text(text)

    assert len(chunks) >= 2
    # Second chunk should begin with tail of the first chunk (overlap region).
    assert chunks[1].text.startswith(chunks[0].text[-15:])


def test_from_settings_uses_config_defaults() -> None:
    settings = Settings(chunk_size=800, chunk_overlap=120)
    chunker = RecursiveCharacterChunker.from_settings(settings)

    assert chunker.chunk_size == 800
    assert chunker.chunk_overlap == 120


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="chunk_size"):
        RecursiveCharacterChunker(chunk_size=0, chunk_overlap=0)


def test_sample_fixture_splits(sample_text: str) -> None:
    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.split_text(sample_text)

    assert len(chunks) >= 1
    combined_unique = " ".join(c.text for c in chunks)
    assert "Retrieval-Augmented Generation" in combined_unique
