"""
TTS / audio metrics — ⚠ MIXED: one fully measurable, two "bring-your-own-signal"
================================================================================
From the image:  ASR loopback WER · pronunciation flags · intelligibility

CORE CONCEPT
------------
These judge SPOKEN audio, but DeepEval (and any text metric) only sees TEXT or
NUMBERS. So the split is about WHERE the measurement happens:

  ✅ ASR loopback WER — FULLY MEASURABLE here.
        Pipeline: TTS speaks the text -> an ASR model transcribes that audio
        back -> we compare the transcript to the intended text with WER.
        The audio round-trip happens in YOUR pipeline; the WER step is just
        `jiwer` on (intended_text vs loopback_transcript). No key needed.

  ⚠ pronunciation flags / intelligibility — the SCORE is computed by an AUDIO
        model (forced aligner, MOS/STOI predictor, ASR confidence), which lives
        OUTSIDE DeepEval. You cannot derive them from text. BUT once your audio
        stack produces those numbers, you can still fold them into the SAME
        DeepEval run as pass/fail gates by passing them in via `metadata`.

So: all three can be reported together in one `evaluate()`, but only the
loopback WER is *calculated* by this file — the other two read precomputed
signals your TTS/ASR system must supply.

RUN (no key required)
---------------------
    uv run python pipeline_metrics/03_tts_audio_metrics.py
"""

import re

import jiwer

import _common  # noqa: F401  — imported for its UTF-8 console fix (Windows-safe rich output)
from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

_PUNCT = re.compile(r"[^\w\s]")


def _norm(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


# ---- ✅ Fully measurable: ASR loopback WER ---------------------------------
class ASRLoopbackWERMetric(BaseMetric):
    """
    WER between the text we asked TTS to speak (expected_output) and what an ASR
    model heard when transcribing the synthesized audio (actual_output).
    High loopback WER => the TTS voice is hard to recognize / mispronounces.
    """

    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold

    def measure(self, tc: LLMTestCase, *a, **k) -> float:
        self.score = jiwer.wer(_norm(tc.expected_output), _norm(tc.actual_output))
        self.success = self.score <= self.threshold
        self.reason = f"loopback WER = {self.score:.1%}"
        return self.score

    async def a_measure(self, tc, *a, **k):
        return self.measure(tc)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "ASR loopback WER"


# ---- ⚠ Bring-your-own-signal: read precomputed audio scores from metadata --
class PronunciationFlagRateMetric(BaseMetric):
    """
    Your audio stack flags mispronounced words; you pass the count in via
    metadata={'pronunciation_flags': N, 'word_count': M}. We score the flag
    RATE (lower is better). The DETECTION is not done here — only the gating.
    """

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def measure(self, tc: LLMTestCase, *a, **k) -> float:
        meta = tc.metadata or {}
        flags = meta.get("pronunciation_flags", 0)
        words = meta.get("word_count") or max(len(_norm(tc.expected_output).split()), 1)
        self.score = flags / words
        self.success = self.score <= self.threshold
        self.reason = f"{flags} flagged / {words} words = {self.score:.1%}"
        return self.score

    async def a_measure(self, tc, *a, **k):
        return self.measure(tc)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Pronunciation flag rate"


class IntelligibilityMetric(BaseMetric):
    """
    Intelligibility/MOS is predicted by an audio model (e.g. NISQA, STOI). You
    pass the 0-1 score in via metadata={'intelligibility': 0.82}. Higher is
    better, so this gate passes when score >= threshold.
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def measure(self, tc: LLMTestCase, *a, **k) -> float:
        meta = tc.metadata or {}
        self.score = float(meta.get("intelligibility", 0.0))
        self.success = self.score >= self.threshold
        self.reason = f"predicted intelligibility = {self.score:.2f}"
        return self.score

    async def a_measure(self, tc, *a, **k):
        return self.measure(tc)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Intelligibility"


def main() -> None:
    # One synthesized utterance. `actual_output` is what ASR heard from the TTS
    # audio; `metadata` carries the precomputed audio-model signals.
    test_case = LLMTestCase(
        input="<text sent to the TTS engine>",
        expected_output="take fifteen milligrams of warfarin once daily",
        actual_output="take fifty milligrams of warfarin once daily",  # ASR misheard
        metadata={
            "pronunciation_flags": 1,   # from your forced aligner
            "word_count": 7,
            "intelligibility": 0.82,    # from your MOS/STOI predictor
        },
    )

    # All three reported together; only the first is *computed* from text here.
    evaluate(
        test_cases=[test_case],
        metrics=[
            ASRLoopbackWERMetric(threshold=0.10),
            PronunciationFlagRateMetric(threshold=0.05),
            IntelligibilityMetric(threshold=0.7),
        ],
    )


if __name__ == "__main__":
    main()
