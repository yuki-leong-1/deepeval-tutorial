"""
Shared helpers for the pipeline metric files (translation / STT / TTS / system).

Provides:
  • UTF-8 console fix (so DeepEval's rich report doesn't crash on Windows)
  • .env loading + OPENAI_API_KEY check
  • tiny text utilities used by the deterministic medical metrics
  • a couple of small lexicons (units, medical terms) you should replace with
    your real domain lists.
"""

import os
import re
import sys

from dotenv import load_dotenv

# --- Windows console: force UTF-8 so ✓ / ⚠ / 🎉 in reports don't crash --------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

load_dotenv()


def require_openai_key() -> None:
    """Stop with a clear message if no LLM-judge key is configured."""
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "\n[!] OPENAI_API_KEY is not set (needed by the GEval/LLM-judge metrics).\n"
            "    Copy .env.example to .env and add your key, then re-run.\n"
            "    NOTE: the STT file (02) and most of TTS/system files run WITHOUT a key.\n"
        )


# --- Text utilities ----------------------------------------------------------

# A tiny unit lexicon. Replace with your real clinical unit list.
UNIT_LEXICON = {
    "mg", "g", "kg", "mcg", "µg", "ug", "ml", "l", "dl", "cc",
    "mmhg", "mmol", "mol", "iu", "u", "units", "unit", "%", "bpm",
    "mg/dl", "meq", "ng", "puff", "puffs", "tablet", "tablets", "drop", "drops",
}

# A tiny medical-term lexicon for "medical term WER". Replace with your own.
MEDICAL_TERM_LEXICON = {
    "insulin", "warfarin", "metformin", "ibuprofen", "penicillin", "morphine",
    "allergy", "allergic", "diabetes", "hypertension", "asthma", "pregnant",
    "anticoagulant", "dose", "dosage", "intravenous", "subcutaneous", "oral",
}

_WORD_RE = re.compile(r"[^\w%/°µ]+", flags=re.UNICODE)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase + split on non-word chars (keeps %, /, °, µ for units)."""
    return [t for t in _WORD_RE.split(text.lower()) if t]


def extract_numbers(text: str) -> list[str]:
    """All numeric tokens, with ',' normalized to '.' (e.g. '1,5' -> '1.5')."""
    return [n.replace(",", ".") for n in _NUM_RE.findall(text)]


def extract_units(text: str) -> list[str]:
    """Tokens that look like clinical units (from UNIT_LEXICON)."""
    return [t for t in tokenize(text) if t in UNIT_LEXICON]


def keep_only(text: str, vocab: set[str]) -> str:
    """Reduce a string to just the tokens present in `vocab` (order kept)."""
    return " ".join(t for t in tokenize(text) if t in vocab)
