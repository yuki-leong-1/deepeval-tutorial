"""
Tutorial 02 — Evaluating a RAG Pipeline
=======================================

CORE CONCEPT
------------
A RAG (Retrieval-Augmented Generation) app does two jobs:
    RETRIEVER  → fetches chunks of context for the question
    GENERATOR  → writes an answer using those chunks

DeepEval ships 5 metrics that pinpoint *which half* is failing. They split
neatly into "generator" vs "retriever" quality:

  GENERATOR metrics (is the answer good given what was retrieved?)
    • AnswerRelevancyMetric  — does the answer actually address the question?
    • FaithfulnessMetric     — is every claim grounded in retrieval_context
                               (i.e. no hallucination)?

  RETRIEVER metrics (did we fetch the right context?)
    • ContextualRelevancyMetric — is the retrieved context on-topic?
    • ContextualRecallMetric    — does it contain everything needed for the
                                  expected_output? (catches "missing chunk")
    • ContextualPrecisionMetric — are the most relevant chunks ranked first?

KEY FIELDS each metric reads on the LLMTestCase:
    AnswerRelevancy      : input, actual_output
    Faithfulness         : input, actual_output, retrieval_context
    ContextualRelevancy  : input,                retrieval_context
    ContextualRecall     : input, expected_output, retrieval_context
    ContextualPrecision  : input, expected_output, retrieval_context

`evaluate()` runs MANY metrics over MANY test cases at once and prints a table
— this is the normal way to evaluate (vs. the single `.measure()` in Tut 01).

RUN
---
    uv run python tutorials/02_rag_metrics.py

Requires OPENAI_API_KEY.
"""

from _setup import require_openai_key
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

JUDGE = "gpt-4o-mini"  # cheap, capable judge used for every metric below


def main() -> None:
    require_openai_key()

    # ---- Step 1: simulate a RAG result -------------------------------------
    # `retrieval_context` is what YOUR retriever returned. `actual_output` is
    # what YOUR generator wrote. Here we hard-code them so the example is
    # self-contained; in production you'd capture them from your real pipeline.
    test_case = LLMTestCase(
        input="What is the return window and who pays return shipping?",
        actual_output=(
            "You can return items within 30 days for a full refund, and "
            "return shipping is free."
        ),
        # The ground-truth answer (needed by recall & precision).
        expected_output=(
            "Returns are accepted within 30 days and the customer pays return "
            "shipping."
        ),
        # The chunks the retriever fetched.
        retrieval_context=[
            "Our return policy allows returns within 30 days of delivery.",
            "Customers are responsible for return shipping costs.",
            "Gift cards are non-refundable.",  # an off-topic chunk on purpose
        ],
    )

    # ---- Step 2: instantiate the five RAG metrics --------------------------
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=JUDGE),
        FaithfulnessMetric(threshold=0.7, model=JUDGE),
        ContextualRelevancyMetric(threshold=0.7, model=JUDGE),
        ContextualRecallMetric(threshold=0.7, model=JUDGE),
        ContextualPrecisionMetric(threshold=0.7, model=JUDGE),
    ]

    # ---- Step 3: run them all and print a report ---------------------------
    # Note our actual_output claims "return shipping is free" but the context
    # says the customer pays — watch Faithfulness flag that contradiction.
    evaluate(test_cases=[test_case], metrics=metrics)


if __name__ == "__main__":
    main()
