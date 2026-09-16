from __future__ import annotations

import pytest

from research_agent.generation.anthropic_client import API_KEY_ENV_VAR, AnthropicGenerator
from research_agent.generation.base import GenerationConfigError


def test_missing_api_key_raises_generation_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)

    with pytest.raises(GenerationConfigError):
        AnthropicGenerator()


def test_generate_with_no_chunks_raises_before_any_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-test-not-a-real-key")
    generator = AnthropicGenerator()

    with pytest.raises(ValueError):
        generator.generate("question", [])
