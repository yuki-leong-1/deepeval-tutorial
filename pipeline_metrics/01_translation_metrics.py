"""
Translation + Medical-safety metrics — ✅ MEASURABLE, ✅ MEASURED TOGETHER
=========================================================================
From the image:
  Translation metrics:     COMET / XCOMET · chrF++ · LLM judge via DeepEval
  Medical safety metrics:  number · unit · drug · negation · frequency ·
                           critical-fact accuracy

CORE CONCEPT
------------
All of these judge the SAME thing — one translation — so they share ONE
LLMTestCase and run together in ONE `evaluate()` call:
    input            = source clinical text
    actual_output    = your machine translation
    expected_output  = human reference translation

They come in three "flavours", but DeepEval unifies them because each is wrapped
as a `BaseMetric`:

  • Deterministic, NO API key:
        chrF++           (character n-gram F-score, via `sacrebleu`)
        number accuracy  (did every dose/value survive?)
        unit accuracy    (mg vs ml etc.)
  • Neural, NO API key but a heavy model download (OPTIONAL — see USE_COMET):
        COMET / XCOMET   (learned MT quality, via `unbabel-comet`)
  • LLM-as-judge, NEEDS OPENAI_API_KEY (DeepEval's GEval):
        LLM judge (overall adequacy+fluency)
        drug / negation / frequency / critical-fact accuracy
        (these are semantic and language-pair-agnostic, so an LLM judge is the
         robust choice over brittle string rules)

Because GEval is in the mix, THIS FILE NEEDS A KEY overall. (If you only want
the keyless metrics, see 02_stt_metrics.py which is fully offline.)

Direction note: chrF++ and the accuracy metrics are "higher = better"
(pass when score >= threshold); that's the opposite of the WER metrics in file 02.

RUN
---
    uv run python pipeline_metrics/01_translation_metrics.py        # needs OPENAI_API_KEY
    USE_COMET=1 uv run python pipeline_metrics/01_translation_metrics.py  # also run COMET
"""

import importlib.util
import os

import sacrebleu

from _common import extract_numbers, extract_units, require_openai_key
from deepeval import evaluate
from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

JUDGE = "gpt-4o-mini"


# ============================================================================
# Deterministic metrics (no API key) — wrapped as DeepEval BaseMetric
# ============================================================================
class ChrfppMetric(BaseMetric):
    """chrF++ via sacrebleu. Score normalized to 0-1; higher is better."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, tc: LLMTestCase, *a, **k) -> float:
        raw = sacrebleu.sentence_chrf(
            tc.actual_output, [tc.expected_output], word_order=2  # word_order=2 -> chrF++
        ).score
        self.score = raw / 100.0
        self.success = self.score >= self.threshold
        self.reason = f"chrF++ = {raw:.1f}/100"
        return self.score

    async def a_measure(self, tc, *a, **k):
        return self.measure(tc)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "chrF++"


class _AccuracyMetric(BaseMetric):
    """Base for 'fraction of expected items preserved in the output' metrics."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold  # default: require 100% for clinical safety

    async def a_measure(self, tc, *a, **k):
        return self.measure(tc)

    def is_successful(self):
        return self.success

    def _score_items(self, expected: list[str], actual: list[str]) -> float:
        if not expected:
            return 1.0
        remaining = list(actual)
        matched = 0
        for item in expected:
            if item in remaining:
                remaining.remove(item)  # multiset match (handles repeats)
                matched += 1
        return matched / len(expected)


class NumberAccuracyMetric(_AccuracyMetric):
    def measure(self, tc: LLMTestCase, *a, **k) -> float:
        exp, act = extract_numbers(tc.expected_output), extract_numbers(tc.actual_output)
        self.score = self._score_items(exp, act)
        self.success = self.score >= self.threshold
        self.reason = f"expected numbers {exp or '∅'} -> {self.score:.0%} preserved (got {act or '∅'})"
        return self.score

    @property
    def __name__(self):
        return "Number accuracy"


class UnitAccuracyMetric(_AccuracyMetric):
    def measure(self, tc: LLMTestCase, *a, **k) -> float:
        exp, act = extract_units(tc.expected_output), extract_units(tc.actual_output)
        self.score = self._score_items(exp, act)
        self.success = self.score >= self.threshold
        self.reason = f"expected units {exp or '∅'} -> {self.score:.0%} preserved (got {act or '∅'})"
        return self.score

    @property
    def __name__(self):
        return "Unit accuracy"


# ============================================================================
# Optional neural metric: COMET / XCOMET (heavy dep, no key, big download)
# ============================================================================
class CometMetric(BaseMetric):
    """
    Learned MT-quality estimator. Requires `unbabel-comet` (pulls in torch) and
    downloads a ~2 GB model on first run, so it is OFF by default. Enable with
    the USE_COMET=1 env var. This shows that even heavy third-party scorers fold
    into the same evaluate() once wrapped as a BaseMetric.
    """

    _model = None

    def __init__(self, threshold: float = 0.5, model_name: str = "Unbabel/wmt22-comet-da"):
        self.threshold = threshold
        self.model_name = model_name

    def _load(self):
        if CometMetric._model is None:
            from comet import download_model, load_from_checkpoint  # lazy import
            CometMetric._model = load_from_checkpoint(download_model(self.model_name))
        return CometMetric._model

    def measure(self, tc: LLMTestCase, *a, **k) -> float:
        model = self._load()
        data = [{"src": tc.input, "mt": tc.actual_output, "ref": tc.expected_output}]
        self.score = float(model.predict(data, progress_bar=False).system_score)
        self.success = self.score >= self.threshold
        self.reason = f"COMET = {self.score:.3f}"
        return self.score

    async def a_measure(self, tc, *a, **k):
        return self.measure(tc)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "COMET"


# ============================================================================
# LLM-as-judge metrics (DeepEval GEval) — NEED OPENAI_API_KEY
# ============================================================================
def build_geval_metrics() -> list[GEval]:
    params = [
        SingleTurnParams.INPUT,            # source
        SingleTurnParams.ACTUAL_OUTPUT,    # machine translation
        SingleTurnParams.EXPECTED_OUTPUT,  # reference translation
    ]
    return [
        GEval(
            name="LLM judge (adequacy+fluency)",
            criteria=(
                "Judge whether the actual output is a fluent, accurate translation "
                "of the input that conveys the same meaning as the expected output."
            ),
            evaluation_params=params, model=JUDGE,
        ),
        GEval(
            name="Drug accuracy",
            evaluation_steps=[
                "Identify every medication/drug name in the input and expected output.",
                "Verify each appears in the actual output with no substitution, omission or misspelling.",
                "Output 0 if any drug name is wrong or missing; 1 if all are correct.",
            ],
            evaluation_params=params, model=JUDGE,
        ),
        GEval(
            name="Negation accuracy",
            criteria=(
                "Are all negations (no, not, without, denies, absence of) preserved so the "
                "actual output never flips the clinical polarity of the expected output?"
            ),
            evaluation_params=params, model=JUDGE,
        ),
        GEval(
            name="Frequency accuracy",
            criteria=(
                "Is the dosing frequency/schedule (e.g. 'twice daily', 'every 8 hours', "
                "'BID', 'as needed') translated with the same meaning?"
            ),
            evaluation_params=params, model=JUDGE,
        ),
        GEval(
            name="Critical-fact accuracy",
            criteria=(
                "Is any clinically critical fact (allergy, dosage, drug, route, "
                "contraindication) altered, added or omitted? Score 1 only if none are."
            ),
            evaluation_params=params, model=JUDGE,
        ),
    ]


def main() -> None:
    require_openai_key()  # GEval metrics need a key; checked before constructing them

    # Source (EN) -> reference (ES). The machine translation got the DOSE wrong
    # (15 -> 50) and dropped a unit, which the safety metrics should catch.
    test_case = LLMTestCase(
        input="Take 15 mg of warfarin once daily. Patient has no penicillin allergy.",
        actual_output="Tome 50 mg de warfarina una vez al día. El paciente tiene alergia a la penicilina.",
        expected_output="Tome 15 mg de warfarina una vez al día. El paciente no tiene alergia a la penicilina.",
    )

    # Build the metric list. Deterministic + GEval all run in ONE evaluate().
    metrics: list[BaseMetric] = [
        ChrfppMetric(threshold=0.4),
        NumberAccuracyMetric(threshold=1.0),  # zero tolerance: every dose must survive
        UnitAccuracyMetric(threshold=1.0),
        *build_geval_metrics(),
    ]

    # COMET only if explicitly enabled AND installed.
    if os.getenv("USE_COMET") == "1":
        if importlib.util.find_spec("comet") is not None:
            metrics.insert(1, CometMetric(threshold=0.5))
        else:
            print("USE_COMET=1 but `unbabel-comet` isn't installed; skipping COMET. "
                  "Install with:  uv add unbabel-comet\n")

    evaluate(test_cases=[test_case], metrics=metrics)


if __name__ == "__main__":
    main()
