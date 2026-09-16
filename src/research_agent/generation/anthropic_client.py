"""Anthropic-backed `Generator` implementation.

Default Phase 1 provider (see CLAUDE.md open decisions -- this is a
concrete, swappable choice, not a settled architectural commitment).
Credentials are read only from the environment; nothing is hardcoded, and
construction fails clearly if the API key is missing.
"""

from __future__ import annotations

import os

from research_agent.generation.base import GenerationConfigError
from research_agent.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from research_agent.models import RetrievedChunk

API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"


class AnthropicGenerator:
    def __init__(self, model_name: str = "claude-sonnet-5", max_tokens: int = 1024) -> None:
        api_key = os.environ.get(API_KEY_ENV_VAR)
        if not api_key:
            raise GenerationConfigError(
                f"{API_KEY_ENV_VAR} is not set. Export it (see .env.example) before "
                "running generation."
            )

        # Imported lazily so this module can be imported (e.g. for the error
        # class above) without the `anthropic` package being installed.
        from anthropic import Anthropic

        self.model_name = model_name
        self._max_tokens = max_tokens
        self._client = Anthropic(api_key=api_key)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            raise ValueError("cannot generate an answer with zero retrieved chunks")

        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(question, chunks)}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
