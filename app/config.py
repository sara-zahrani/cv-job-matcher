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
