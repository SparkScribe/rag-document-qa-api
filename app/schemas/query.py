"""Query request and response schemas."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    document_id: str | None = Field(
        default=None,
        description="Optional document scope; searches all documents when omitted",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of chunks to retrieve (defaults to server setting)",
    )


class SourceCitation(BaseModel):
    document_id: str
    chunk_index: int
    score: float = Field(ge=0.0, le=1.0)
    excerpt: str = Field(description="Sanitized excerpt from the retrieved chunk")


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    model: str
