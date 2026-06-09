"""
STT (Speech-to-Text) metrics  — ✅ ALL MEASURABLE, ✅ MEASURED TOGETHER
======================================================================
From the image:  WER · CER · medical term WER · number WER

CORE CONCEPT
------------
These are DETERMINISTIC, reference-based metrics: you compare the recognized
transcript (`actual_output`) against the human ground-truth transcript
(`expected_output`). No LLM judge, so NO API KEY is needed.

DeepEval has no built-in WER, but DeepEval lets you wrap ANY scorer as a custom
metric by subclassing `BaseMetric`. Once wrapped, all four run together in ONE
`evaluate()` call over the SAME test case — because they all read the same
(actual_output vs expected_output) pair. That's the rule for "can be measured
together": same test case + all are single-turn metrics.

We use the `jiwer` library to compute the edit-distance error rates:
    WER             = word errors / words            (overall)
    CER             = character errors / characters  (catches spelling)
    medical term WER= WER computed only over medical terms (clinical safety)
    number WER      = WER computed only over numeric tokens (dosages!)

Lower is better, so `is_successful()` is True when score <= threshold.

RUN (no key required)
---------------------
    uv run python pipeline_metrics/02_stt_metrics.py
"""

import re

import jiwer

from _common import MEDICAL_TERM_LEXICON, keep_only
from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

_PUNCT = re.compile(r"[^\w\s]")


def _norm(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — standard ASR scoring prep."""
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


class _ErrorRateMetric(BaseMetric):
    """Base class: an error-rate metric passes when score <= threshold."""

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class WERMetric(_ErrorRateMetric):
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        ref, hyp = _norm(test_case.expected_output), _norm(test_case.actual_output)
        self.score = jiwer.wer(ref, hyp)
        self.success = self.score <= self.threshold
        self.reason = f"{self.score:.1%} of words wrong (ref='{ref}' vs hyp='{hyp}')"
        return self.score

    @property
    def __name__(self):
        return "WER"


class CERMetric(_ErrorRateMetric):
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        ref, hyp = _norm(test_case.expected_output), _norm(test_case.actual_output)
        self.score = jiwer.cer(ref, hyp)
        self.success = self.score <= self.threshold
        self.reason = f"{self.score:.1%} of characters wrong"
        return self.score

    @property
    def __name__(self):
        return "CER"


class MedicalTermWERMetric(_ErrorRateMetric):
    """WER computed over ONLY the medical terms — a wrong drug name is critical."""

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        ref = keep_only(test_case.expected_output, MEDICAL_TERM_LEXICON)
        hyp = keep_only(test_case.actual_output, MEDICAL_TERM_LEXICON)
        # No medical terms in the reference -> nothing to get wrong -> perfect.
        self.score = jiwer.wer(ref, hyp) if ref else 0.0
        self.success = self.score <= self.threshold
        self.reason = f"medical terms ref='{ref or '∅'}' vs hyp='{hyp or '∅'}' -> {self.score:.1%} WER"
        return self.score

    @property
    def __name__(self):
        return "Medical-term WER"


class NumberWERMetric(_ErrorRateMetric):
    """WER over ONLY numeric tokens — a misheard dosage can be dangerous."""

    _NUM = re.compile(r"\d+(?:[.,]\d+)?")

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        ref = " ".join(self._NUM.findall(test_case.expected_output))
        hyp = " ".join(self._NUM.findall(test_case.actual_output))
        self.score = jiwer.wer(ref, hyp) if ref else 0.0
        self.success = self.score <= self.threshold
        self.reason = f"numbers ref='{ref or '∅'}' vs hyp='{hyp or '∅'}' -> {self.score:.1%} WER"
        return self.score

    @property
    def __name__(self):
        return "Number WER"


def main() -> None:
    # A transcript where the ASR misheard the DOSAGE ("15" -> "50") and a term.
    test_case = LLMTestCase(
        input="<audio of clinician speaking>",
        actual_output="the patient takes 50 mg of warfarin twice daily for diabetes",
        expected_output="the patient takes 15 mg of warfarin twice daily for diabetes",
    )

    # All four error-rate metrics run together in one call (same test case).
    evaluate(
        test_cases=[test_case],
        metrics=[
            WERMetric(threshold=0.10),
            CERMetric(threshold=0.10),
            MedicalTermWERMetric(threshold=0.0),   # zero tolerance for drug errors
            NumberWERMetric(threshold=0.0),         # zero tolerance for dosage errors
        ],
    )


if __name__ == "__main__":
    main()
