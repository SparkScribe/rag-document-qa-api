"""Extract plain text from uploaded document bytes."""

import io
import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".pdf"})


class UnsupportedFileTypeError(Exception):
    """Raised when the uploaded file extension is not supported."""


class EmptyDocumentError(Exception):
    """Raised when no extractable text is found in a document."""


def validate_extension(filename: str) -> str:
    """Return the normalized lowercase extension or raise."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or '(none)'}'. Allowed: txt, pdf"
        )
    return extension


def extract_text(filename: str, content: bytes) -> str:
    """Parse document bytes into plain text."""
    extension = validate_extension(filename)

    if extension == ".txt":
        return _extract_txt(content)
    if extension == ".pdf":
        return _extract_pdf(content)

    raise UnsupportedFileTypeError(f"Unsupported file type: {extension}")


def _extract_txt(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = content.decode(encoding).strip()
            if text:
                return text
        except UnicodeDecodeError:
            continue
    raise EmptyDocumentError("Text file contains no readable content")


def _extract_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text.strip())

    if not pages:
        raise EmptyDocumentError("PDF contains no extractable text")

    return "\n\n".join(pages)
