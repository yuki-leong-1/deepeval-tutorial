"""
Tutorial 03 — Custom Metrics with GEval
=======================================

CORE CONCEPT
------------
The built-in metrics (Tut 02) cover common cases, but every product has its
OWN definition of "good". GEval lets you invent a metric from a description —
no training, no code, just English.

Two ways to define what GEval grades:
    • criteria         — a one-liner; GEval auto-expands it into grading steps.
    • evaluation_steps — you spell out the exact checklist. More control and
                         more reproducible. Prefer this for metrics you'll
                         reuse or put in CI.

Optional knobs shown here:
    • rubric      — pin specific score ranges to descriptions (0-2 = bad, etc.)
                    so scores mean the same thing every run.
    • strict_mode — force a binary 0/1 verdict (pass or fail, no partial credit).

GEval only looks at the test-case fields you list in `evaluation_params`, so
the SAME metric can grade tone (actual_output only), correctness
(actual_output + expected_output), groundedness (+ retrieval_context), etc.

RUN
---
    uv run python tutorials/03_custom_geval.py

Requires OPENAI_API_KEY.
"""

from _setup import require_openai_key
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval.utils import Rubric

JUDGE = "gpt-4o-mini"


def main() -> None:
    require_openai_key()

    # ---- Metric A: tone, defined with a single `criteria` ------------------
    professionalism = GEval(
        name="Professionalism",
        criteria=(
            "Is the response polite, empathetic and free of slang or blame "
            "toward the customer?"
        ),
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],  # tone needs only the answer
        model=JUDGE,
    )

    # ---- Metric B: explicit steps + a rubric for stable scoring ------------
    helpfulness = GEval(
        name="Helpfulness",
        # Spelling out steps makes the score far more reproducible than a
        # vague one-liner — GEval follows this checklist literally.
        evaluation_steps=[
            "Check whether the response directly answers the user's question.",
            "Check whether it gives a concrete next step the user can take.",
            "Penalize vague deflections like 'contact support' with no detail.",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        rubric=[
            Rubric(score_range=(0, 3), expected_outcome="Unhelpful or evasive."),
            Rubric(score_range=(4, 7), expected_outcome="Partially helpful."),
            Rubric(score_range=(8, 10), expected_outcome="Directly solves the problem."),
        ],
        model=JUDGE,
    )

    # ---- Two test cases: one good answer, one deflecting answer ------------
    good = LLMTestCase(
        input="My order hasn't arrived and it's been two weeks. What do I do?",
        actual_output=(
            "I'm sorry for the wait! I've checked and your parcel is delayed in "
            "transit. I've requested a reshipment that will arrive in 3-5 days, "
            "and emailed you a tracking link. Let me know if you'd prefer a refund."
        ),
    )
    evasive = LLMTestCase(
        input="My order hasn't arrived and it's been two weeks. What do I do?",
        actual_output="Please contact support.",
    )

    # Run both custom metrics over both cases.
    evaluate(test_cases=[good, evasive], metrics=[professionalism, helpfulness])


if __name__ == "__main__":
    main()
