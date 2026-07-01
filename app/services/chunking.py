"""LangChain-free recursive character text splitter."""

from dataclasses import dataclass

from app.core.config import Settings


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A single text segment produced by the chunker."""

    index: int
    text: str


DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ")


class RecursiveCharacterChunker:
    """Split text into overlapping chunks using recursive separator priority."""

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    @classmethod
    def from_settings(cls, settings: Settings) -> "RecursiveCharacterChunker":
        return cls(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def split_text(self, text: str) -> list[TextChunk]:
        """Split *text* into ordered chunks with configured overlap."""
        normalized = text.strip()
        if not normalized:
            return []

        raw_chunks = self._split_recursive(normalized, self.separators)
        merged = self._merge_splits(raw_chunks)
        overlapped = self._apply_overlap(merged)

        return [TextChunk(index=i, text=chunk) for i, chunk in enumerate(overlapped)]

    def _split_recursive(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text else []

        separator, *remaining = separators

        if separator:
            parts = text.split(separator)
            # Re-attach separator to all parts except the last (split consumes it).
            parts = [part + separator for part in parts[:-1]] + ([parts[-1]] if parts else [])
        else:
            # Final fallback: hard split by character count.
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = current + part
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = ""

            if len(part) <= self.chunk_size:
                current = part
                continue

            if remaining:
                chunks.extend(self._split_recursive(part, tuple(remaining)))
            else:
                chunks.extend(
                    part[i : i + self.chunk_size]
                    for i in range(0, len(part), self.chunk_size)
                )

        if current:
            chunks.append(current)

        return chunks

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Merge adjacent small splits up to chunk_size."""
        if not splits:
            return []

        merged: list[str] = []
        current = splits[0]

        for split in splits[1:]:
            if len(current) + len(split) <= self.chunk_size:
                current += split
            else:
                merged.append(current)
                current = split

        merged.append(current)
        return merged

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks

        result: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            result.append(overlap + chunks[i])

        return result
