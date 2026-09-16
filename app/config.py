"""
Central configuration.

Reads the environment once, at import time, into a single Settings object.
Every other module asks `settings` for what it needs. Nothing else in the
app calls os.getenv directly.
"""

import os

from dotenv import load_dotenv

# Reads the .env file in the project root and copies each line into the
# process environment, as if you had typed `export KEY=value` in the shell.
# Does nothing if .env is missing, so production (where real env vars are
# set) and local (where .env is) behave the same.
load_dotenv()


class Settings:
    def __init__(self) -> None:
        # os.getenv returns None if the variable is missing. The "" default
        # means we always hold a string, which makes the checks below simpler.
        self.openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_base_url: str = "https://openrouter.ai/api/v1"

        # Defaults are free-tier models verified against OpenRouter on 2026-09-16.
        # Override in .env if they change. The embed model must never change
        # after jobs are indexed without re-indexing: vectors from two models
        # live in different spaces and cannot be compared.
        self.embed_model: str = os.getenv("OPENROUTER_EMBED_MODEL") or "nvidia/nemotron-3-embed-1b:free"
        # Free models queue unpredictably: on 2026-09-16 nemotron-3.5-lightning took
        # 37s for one word while nemotron-3-super answered in 1s. Override in .env
        # if the default is slow on the day.
        self.chat_model: str = os.getenv("OPENROUTER_CHAT_MODEL") or "nvidia/nemotron-3-super-120b-a12b:free"

    def require_api_key(self) -> None:
        """Fail loudly at startup rather than quietly on the first API call."""
        if not self.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Copy .env.example to .env and fill it in."
            )


# One instance, created when this module is first imported. Python caches
# imported modules, so every `from app.config import settings` gets this
# same object.
settings = Settings()


if __name__ == "__main__":
    # Quick self-check: run this file directly to see whether the key loaded.
    # We never print the key itself, only whether it exists and its length.
    settings.require_api_key()
    key = settings.openrouter_api_key
    print(f"API key loaded: {key[:6]}... ({len(key)} characters)")
