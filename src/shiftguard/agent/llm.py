"""Ollama wrapper: a schema-constrained chat call plus JSON-validation retry.

`step()` asks the model for one `AgentStep`, parses + validates it, and on bad
output re-prompts with the validation error appended (bounded by max_retries).
Swapping to Ollama's native tool-calling API would be isolated to this module.
"""

from __future__ import annotations

import json

from ollama import Client
from pydantic import ValidationError

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from .schemas import AgentStep


class LLMStepError(RuntimeError):
    """Raised when the model fails to produce a valid step within the retry budget."""


class LLMClient:
    def __init__(self, settings: Settings | None = None, client: Client | None = None):
        self.settings = settings or get_settings()
        self.client = client or Client(host=self.settings.ollama_host)
        self.log = get_logger("agent.llm")

    def _chat(self, messages: list[dict], format_schema: dict) -> str:
        response = self.client.chat(
            model=self.settings.ollama_model,
            messages=messages,
            format=format_schema,
            options={
                "num_ctx": self.settings.ollama_context_length,
                "temperature": self.settings.temperature,
            },
            stream=False,
        )
        return response["message"]["content"]

    def step(self, messages: list[dict], format_schema: dict) -> tuple[AgentStep, str]:
        """Return a validated AgentStep and its raw JSON, retrying on bad output."""
        working = list(messages)
        last_error: Exception | None = None
        attempts = self.settings.max_retries + 1

        for attempt in range(1, attempts + 1):
            content = self._chat(working, format_schema)
            try:
                return AgentStep.model_validate(json.loads(content)), content
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                self.log.warning("invalid step (attempt %d/%d): %s", attempt, attempts, e)
                working = working + [
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid ({e}). Reply with ONE valid "
                            "JSON object: a 'thought' plus exactly one of 'action' or 'final_answer'."
                        ),
                    }
                ]

        raise LLMStepError(f"no valid step after {attempts} attempts: {last_error}")
