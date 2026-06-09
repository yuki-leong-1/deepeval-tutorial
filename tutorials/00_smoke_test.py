"""
Tutorial 00 — Smoke Test (no API key needed)
============================================

CORE CONCEPT
------------
Before spending tokens on LLM judges, confirm the install works. Building an
LLMTestCase touches no network at all, so that part always runs offline and
shows you the anatomy of a test case.

Note: *constructing* a metric (e.g. GEval) already initializes its judge
client, so it needs OPENAI_API_KEY even though no LLM is CALLED until
`.measure()`. This file builds the test case unconditionally, then tries to
construct a metric and tells you whether your key is configured.

RUN
---
    uv run python tutorials/00_smoke_test.py

No OPENAI_API_KEY required.
"""

import deepeval
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import SingleTurnParams


def main() -> None:
    print(f"DeepEval version: {deepeval.__version__}\n")

    # Build a fully-featured test case (no LLM is called by doing this).
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund on unworn shoes.",
        expected_output="You're eligible for a 30-day refund.",
        context=["All customers get a 30-day full refund at no extra cost."],
        retrieval_context=["Only unworn shoes can be refunded within 30 days."],
        tools_called=[ToolCall(name="search_policy")],
    )
    print("Constructed LLMTestCase:")
    print(f"  input            : {test_case.input}")
    print(f"  actual_output    : {test_case.actual_output}")
    print(f"  expected_output  : {test_case.expected_output}")
    print(f"  retrieval_context: {test_case.retrieval_context}")
    print(f"  tools_called     : {[t.name for t in test_case.tools_called]}\n")

    # Try to construct a metric. This initializes the judge client, so it only
    # succeeds once OPENAI_API_KEY is set (no LLM is actually called yet).
    import os

    if os.getenv("OPENAI_API_KEY"):
        GEval(
            name="Correctness",
            criteria="Is the actual output consistent with the expected output?",
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
        )
        AnswerRelevancyMetric()
        print("Judge client constructed OK — your key works. Environment is ready.")
        print("Next: run  uv run python tutorials/01_first_test_case.py")
    else:
        print("OPENAI_API_KEY not set - install is fine, but metrics need a key.")
        print("Next: copy .env.example to .env, add your key, then re-run this file.")


if __name__ == "__main__":
    main()
