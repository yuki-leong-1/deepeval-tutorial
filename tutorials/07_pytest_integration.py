"""
Tutorial 07 — Pytest Integration (CI/CD)
========================================

CORE CONCEPT
------------
Everything so far used `evaluate()` (prints a report). To FAIL A BUILD when
quality drops, you instead write real tests and let DeepEval drive pytest.

    assert_test(test_case, [metrics])  — like `assert`, but for LLM quality.
                                         Raises (test fails) if any metric's
                                         score is below its threshold.

You run these with DeepEval's own test runner, NOT `python`:

    uv run deepeval test run tutorials/07_pytest_integration.py

That wraps pytest and adds LLM-aware flags, e.g.:
    -n 4   run 4 test cases in parallel (faster)
    -c     use cached results (skip unchanged cases)
    -i     ignore metric errors (don't crash on one bad judge response)
    -r 2   repeat each test case twice (catch flaky/non-deterministic output)

Because it's pytest under the hood, this drops straight into GitHub Actions /
GitLab CI: a failing metric = a failing pipeline = the regression never ships.

Parametrize over a dataset to test many cases with one function (shown below).

RUN
---
    uv run deepeval test run tutorials/07_pytest_integration.py
    # add flags:  uv run deepeval test run tutorials/07_pytest_integration.py -n 2 -i

Requires OPENAI_API_KEY.
"""

import pytest

from _setup import require_openai_key  # noqa: F401  (ensures .env is loaded)
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval, AnswerRelevancyMetric

JUDGE = "gpt-4o-mini"


def my_llm_app(question: str) -> str:
    """Replace with your real app. Canned answers keep the demo self-contained."""
    return {
        "What is your return window?":
            "You can return any item within 30 days of delivery.",
        "Do you offer free shipping?":
            "Yes, orders over $50 ship free.",
    }.get(question, "Sorry, I don't know.")


# A tiny inline dataset: (input, expected_output) pairs.
CASES = [
    ("What is your return window?", "Returns are accepted within 30 days."),
    ("Do you offer free shipping?", "Free shipping on orders over $50."),
]


@pytest.mark.parametrize("question,expected", CASES)
def test_support_answers(question: str, expected: str) -> None:
    """One pytest test per case in CASES (thanks to @parametrize)."""
    correctness = GEval(
        name="Correctness",
        criteria="Is the actual output consistent with the expected output?",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model=JUDGE,
    )
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=JUDGE)

    test_case = LLMTestCase(
        input=question,
        actual_output=my_llm_app(question),
        expected_output=expected,
    )

    # Fails the test if EITHER metric scores below its threshold.
    assert_test(test_case, [correctness, relevancy])
