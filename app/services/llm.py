"""OpenAI chat completion client wrapper."""

import logging
from typing import Protocol

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from app.core.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided context. "
    "If the context does not contain enough information to answer, say clearly that there "
    "is insufficient information. Be concise, factual, and do not invent details."
)


class ChatError(Exception):
    """Base exception for chat completion failures."""


class ChatConfigurationError(ChatError):
    """Raised when the chat service is not properly configured."""


class ChatAPIError(ChatError):
    """Raised when the upstream chat API returns an error."""


class ChatClient(Protocol):
    """Protocol for chat backends (OpenAI or test doubles)."""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


class OpenAIChatService:
    """Generate answers via the OpenAI-compatible chat completions API."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self._settings = settings
        self._client = client

    @property
    def model_name(self) -> str:
        return self._settings.openai_chat_model

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        if not self._settings.openai_api_key:
            raise ChatConfigurationError(
                "OPENAI_API_KEY is not configured. Set it in the environment to use the LLM."
            )

        self._client = OpenAI(
            api_key=self._settings.openai_api_key,
            timeout=self._settings.openai_timeout_seconds,
        )
        return self._client

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the assistant message content for a chat completion."""
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self._settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
        except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
            logger.error("Chat API request failed: %s", exc)
            raise ChatAPIError(f"Chat request failed: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected chat API error: %s", exc)
            raise ChatAPIError(f"Chat request failed: {exc}") from exc

        if response.usage is not None:
            logger.info(
                "Chat tokens used: prompt=%s completion=%s total=%s model=%s",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
                self._settings.openai_chat_model,
            )

        choice = response.choices[0] if response.choices else None
        if choice is None or choice.message.content is None:
            raise ChatAPIError("Chat API returned an empty response")

        return choice.message.content.strip()
