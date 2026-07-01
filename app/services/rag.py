"""Retrieval-augmented generation query orchestration."""

import logging
import re
import unicodedata

from app.core.config import Settings
from app.schemas.query import QueryResponse, SourceCitation
from app.services.document_store import DocumentNotFoundError, DocumentStore
from app.services.embedding import EmbeddingError, OpenAIEmbeddingService
from app.services.llm import ChatError, OpenAIChatService, SYSTEM_PROMPT
from app.services.vector_store import ScoredChunk, VectorStore, VectorStoreError

logger = logging.getLogger(__name__)

INSUFFICIENT_CONTEXT_ANSWER = (
    "I do not have sufficient context in the uploaded documents to answer that question."
)


class RAGError(Exception):
    """Raised when a RAG query fails."""


class RAGService:
    """Embed questions, retrieve chunks, and synthesize cited answers."""

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        embedding_service: OpenAIEmbeddingService,
        chat_service: OpenAIChatService,
        document_store: DocumentStore,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._chat_service = chat_service
        self._document_store = document_store

    def query(
        self,
        question: str,
        *,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> QueryResponse:
        """Answer a question using retrieved document chunks."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty")

        if document_id is not None:
            self._ensure_document_exists(document_id)

        effective_top_k = top_k or self._settings.query_top_k_default

        try:
            query_vector = self._embedding_service.embed_query(normalized_question)
            chunks = self._vector_store.search_similar(
                query_vector,
                top_k=effective_top_k,
                document_id=document_id,
                score_threshold=self._settings.min_query_score,
            )
        except (EmbeddingError, VectorStoreError) as exc:
            raise RAGError(str(exc)) from exc

        if not chunks:
            return QueryResponse(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                sources=[],
                model=self._chat_service.model_name,
            )

        sources = [self._to_source(chunk) for chunk in chunks]

        try:
            answer = self._chat_service.complete(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=self._build_user_prompt(normalized_question, chunks),
            )
        except ChatError as exc:
            raise RAGError(str(exc)) from exc

        return QueryResponse(
            answer=answer,
            sources=sources,
            model=self._chat_service.model_name,
        )

    def _ensure_document_exists(self, document_id: str) -> None:
        try:
            self._document_store.get(document_id)
        except DocumentNotFoundError as exc:
            raise RAGError(f"Document not found: {document_id}") from exc

    def _to_source(self, chunk: ScoredChunk) -> SourceCitation:
        return SourceCitation(
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            score=round(chunk.score, 4),
            excerpt=sanitize_excerpt(chunk.text, self._settings.query_excerpt_max_chars),
        )

    @staticmethod
    def _build_user_prompt(question: str, chunks: list[ScoredChunk]) -> str:
        context_blocks: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"document_id: {chunk.document_id}",
                        f"chunk_index: {chunk.chunk_index}",
                        f"filename: {chunk.filename}",
                        chunk.text,
                    ]
                )
            )

        context = "\n\n".join(context_blocks)
        return (
            "Use the following sources to answer the question.\n\n"
            f"{context}\n\n"
            f"Question: {question}"
        )


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_excerpt(text: str, max_length: int) -> str:
    """Return a printable UTF-8 excerpt safe for API responses."""
    normalized = unicodedata.normalize("NFKC", text)
    without_controls = _CONTROL_CHARS.sub(" ", normalized)
    printable = "".join(
        ch if (ch.isprintable() or ch in {"\n", "\t"}) else " " for ch in without_controls
    )
    collapsed = re.sub(r"\s+", " ", printable).strip()

    if len(collapsed) <= max_length:
        return collapsed

    truncated = collapsed[: max_length - 3].rstrip()
    return f"{truncated}..."
