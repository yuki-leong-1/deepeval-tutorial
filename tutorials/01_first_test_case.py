"""
Tutorial 01 — Your First Test Case
==================================

CORE CONCEPT
------------
DeepEval evaluates an LLM the way unit tests evaluate code. Two building blocks:

  1. LLMTestCase  — one interaction with your app. Required fields:
        input         : what the user asked
        actual_output : what your LLM actually answered
     ...plus optional fields (expected_output, context, retrieval_context,
     tools_called) used by different metrics.

  2. Metric       — the scoring logic. Given a test case it produces:
        .score          : a float in [0, 1]
        .reason         : a natural-language explanation of the score
        .is_successful(): True when score >= threshold

In this file we use **GEval**, a research-backed "LLM-as-a-judge" metric that
can grade ANY criterion you describe in plain English. We grade *Correctness*
by comparing `actual_output` against `expected_output`.

`metric.measure(test_case)` runs ONE metric on ONE test case — the simplest
possible evaluation, perfect for learning what a metric does.

RUN
---
    uv run python tutorials/01_first_test_case.py

Requires OPENAI_API_KEY (GEval calls an OpenAI model to act as the judge).
"""

from _setup import require_openai_key
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval


def main() -> None:
    require_openai_key()

    # ---- Step 1: define a metric -------------------------------------------
    # GEval turns a plain-English `criteria` into a reusable grader.
    # `evaluation_params` tells it WHICH fields of the test case to look at.
    correctness = GEval(
        name="Correctness",
        criteria=(
            "Determine whether the 'actual output' is factually correct and "
            "covers the key points of the 'expected output'. Minor wording "
            "differences are fine; missing or wrong medical advice is not."
        ),
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,          # score >= 0.5 counts as a pass
        model="gpt-4o-mini",    # the judge model (any OpenAI model name works)
    )

    # ---- Step 2: build a test case -----------------------------------------
    # In a real app `actual_output` would come from your own LLM call.
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        actual_output=(
            "A persistent cough and fever could be a viral infection or "
            "something more serious. See a doctor if symptoms worsen or "
            "don't improve in a few days."
        ),
        expected_output=(
            "A persistent cough and fever could indicate anything from a mild "
            "viral infection to more serious conditions like pneumonia or "
            "COVID-19. Seek medical attention if symptoms worsen, last more "
            "than a few days, or come with difficulty breathing or chest pain."
        ),
    )

    # ---- Step 3: measure ----------------------------------------------------
    correctness.measure(test_case)

    print(f"Score   : {correctness.score:.2f}")
    print(f"Success : {correctness.is_successful()}")
    print(f"Reason  : {correctness.reason}")


if __name__ == "__main__":
    main()
