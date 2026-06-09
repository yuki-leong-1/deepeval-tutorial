"""
Shared helper used by every tutorial.

It loads environment variables from a `.env` file (so your OPENAI_API_KEY is
picked up automatically) and gives a friendly error if the key is missing.

You normally do NOT need to read this file to follow the tutorials — it just
keeps the other files short. The one DeepEval-specific thing to know:

    DeepEval reads OPENAI_API_KEY at import time. Most metrics are
    "LLM-as-a-judge", i.e. they call an LLM to grade your LLM. By default
    that judge is an OpenAI model, so a key is required for almost everything.
"""

import os
import sys

from dotenv import load_dotenv

# DeepEval prints its report with `rich`, which emits Unicode (✓, ⚠, 🎉). The
# default Windows console is cp1252 and CRASHES on those characters, so force
# the streams to UTF-8 before any evaluation prints. Harmless on macOS/Linux.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # Python 3.7+
    except (AttributeError, ValueError):
        pass

# Looks for a `.env` file in the project root and loads it into os.environ.
load_dotenv()


def require_openai_key() -> None:
    """Print a clear message and exit if no judge model is configured."""
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "\n[!] OPENAI_API_KEY is not set.\n"
            "    Copy .env.example to .env and paste your key, e.g.:\n"
            "        OPENAI_API_KEY=sk-...\n"
            "    (Or use a local model - see the README 'Using a different judge' section.)\n"
        )
