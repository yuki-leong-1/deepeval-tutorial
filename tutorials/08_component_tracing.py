"""
Tutorial 08 — Component-Level Evaluation with Tracing
=====================================================

CORE CONCEPT
------------
Tutorials 01-07 do END-TO-END evaluation: feed an input, judge the final
output, treat the app as a black box. That's great for "is the product good?"
but useless for "WHICH step broke?" in a multi-step agent or RAG pipeline.

COMPONENT-LEVEL evaluation fixes that by tracing the internals:

    @observe(...)        — decorate any function to record it as a "span"
                           (one node in the execution trace). Give it a `type`
                           ("retriever", "llm", "tool", "agent") and, optionally,
                           the `metrics` that should grade THAT span.

    update_current_span( — inside a span, report the data its metrics need
        input=..., output=..., retrieval_context=...)   (input/output/context).

    dataset.evals_iterator(metrics=...) — yields each golden; you call your
                           traced app on it. DeepEval collects every span's
                           trace and scores the per-span metrics, so you get a
                           breakdown like "retriever 0.9, generator 0.4".

Below, a mini RAG app has two inner spans — a retriever and a generator — each
with its own metric. We then run the whole dataset through it and DeepEval
reports each component's score separately.

RUN
---
    uv run python tutorials/08_component_tracing.py

Requires OPENAI_API_KEY.
"""

from _setup import require_openai_key
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_span
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric

# The metrics below live inside @observe(...) decorators, which run at IMPORT
# time — so we check for the key here, before those decorators construct their
# judge clients. (In tutorials 01-07 the check sits inside main() instead.)
require_openai_key()

JUDGE = "gpt-4o-mini"

# A toy "knowledge base" our retriever searches over.
KB = {
    "return": "Our return policy allows returns within 30 days of delivery.",
    "shipping": "Standard shipping takes 3-5 business days.",
    "warranty": "All products come with a 1-year limited warranty.",
}


# ---- Span 1: the retriever --------------------------------------------------
# `type="retriever"` + ContextualRelevancyMetric grades whether the fetched
# context is on-topic for the query.
@observe(type="retriever", metrics=[ContextualRelevancyMetric(threshold=0.5, model=JUDGE)])
def retrieve(query: str) -> list[str]:
    # Dumb keyword retrieval — returns every KB entry whose key is in the query.
    chunks = [text for key, text in KB.items() if key in query.lower()]
    if not chunks:
        chunks = list(KB.values())  # fall back to everything
    # Tell the retriever's metric what the input/output of this span were.
    update_current_span(input=query, retrieval_context=chunks)
    return chunks


# ---- Span 2: the generator --------------------------------------------------
# `type="llm"` + AnswerRelevancyMetric grades whether the answer addresses the
# query. (Swap the body for a real OpenAI/Anthropic call in your app.)
@observe(type="llm", metrics=[AnswerRelevancyMetric(threshold=0.5, model=JUDGE)])
def generate(query: str, chunks: list[str]) -> str:
    answer = " ".join(chunks)  # a real app would prompt an LLM with the chunks
    update_current_span(input=query, output=answer, retrieval_context=chunks)
    return answer


# ---- Root span: the agent that ties them together ---------------------------
@observe(type="agent")
def rag_app(query: str) -> str:
    chunks = retrieve(query)
    return generate(query, chunks)


def main() -> None:
    # A dataset of inputs only — component metrics live on the spans, so the
    # goldens just need an `input` to drive the app.
    dataset = EvaluationDataset(goldens=[
        Golden(input="What is your return policy?"),
        Golden(input="How long does shipping take?"),
    ])

    # For each golden: run the traced app. After the loop DeepEval scores every
    # span's metrics and prints a per-component report.
    for golden in dataset.evals_iterator():
        rag_app(golden.input)


if __name__ == "__main__":
    main()
