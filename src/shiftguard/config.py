"""Central configuration for ShiftGuard.

Loaded from environment / a local `.env` (see `.env.example`) with safe
defaults, so the app runs out-of-the-box with nothing configured. Relative
paths resolve against the repo root, so commands work regardless of the
current working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives at <root>/src/shiftguard/config.py -> parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (Ollama) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    # Pinned context window. Ollama defaults to 2048 and silently truncates on
    # overflow, which can corrupt a multi-step loop; keep this >= 8192.
    ollama_context_length: int = 8192
    temperature: float = 0.0

    # --- Embeddings / RAG ---
    embed_model: str = "nomic-embed-text"
    top_k: int = 3

    # --- Agent loop ---
    max_steps: int = 8
    max_retries: int = 2

    # --- Paths (relative values resolve against PROJECT_ROOT) ---
    data_dir: Path = Path("data")
    qdrant_dir: Path = Path("data/qdrant")
    log_dir: Path = Path("logs")
    qdrant_collection: str = "policies"

    def _resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_path(self) -> Path:
        return self._resolve(self.data_dir)

    @property
    def policies_dir(self) -> Path:
        return self.data_path / "policies"

    @property
    def timecards_path(self) -> Path:
        return self.data_path / "timecards.json"

    @property
    def tickets_path(self) -> Path:
        return self.data_path / "tickets.jsonl"

    @property
    def qdrant_path(self) -> Path:
        return self._resolve(self.qdrant_dir)

    @property
    def logs_path(self) -> Path:
        return self._resolve(self.log_dir)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
