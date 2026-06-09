"""
Tutorial 05 — Multi-Turn (Conversational) Evaluation
====================================================

CORE CONCEPT
------------
A chatbot isn't a single Q&A — it's a *conversation*. Evaluating only the last
reply misses things like "did it forget what the user said three turns ago?"
or "did it stay in character?".

DeepEval models this with:

    Turn                  — one message: Turn(role="user"|"assistant",
                            content="..."). Optionally carries retrieval_context
                            or tools_called for that turn.

    ConversationalTestCase— an ordered list of Turns (the whole dialogue),
                            plus optional `chatbot_role` / `scenario`.

    Conversational metrics— grade the dialogue as a whole. You CANNOT use a
                            single-turn metric here, and vice versa.

Useful built-in conversational metrics:
    • TurnRelevancyMetric          — is each assistant turn relevant in context?
    • ConversationCompletenessMetric — were the user's goals actually met?
    • RoleAdherenceMetric          — did the assistant stay in `chatbot_role`?
    • KnowledgeRetentionMetric     — did it remember earlier-stated facts?

And **ConversationalGEval** — the multi-turn twin of GEval — lets you grade any
custom conversational criterion in plain English (used below).

RUN
---
    uv run python tutorials/05_conversational.py

Requires OPENAI_API_KEY.
"""

from _setup import require_openai_key
from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase, Turn, MultiTurnParams
from deepeval.metrics import ConversationalGEval, TurnRelevancyMetric

JUDGE = "gpt-4o-mini"


def main() -> None:
    require_openai_key()

    # ---- Step 1: model the conversation as a list of Turns -----------------
    conversation = ConversationalTestCase(
        chatbot_role="a calm, professional airline support agent",
        turns=[
            Turn(role="user", content="Hi, my flight AA123 got cancelled. I'm furious."),
            Turn(role="assistant", content=(
                "I'm really sorry about the cancellation of AA123 — that's "
                "frustrating. I can rebook you on the next available flight or "
                "process a full refund. Which would you prefer?"
            )),
            Turn(role="user", content="Rebook me, and remind me of my seat preference."),
            Turn(role="assistant", content=(
                "You're rebooked on AA456 departing 6pm. I've kept your usual "
                "window seat, 14A, as on your original booking."
            )),
        ],
    )

    # ---- Step 2: pick conversational metrics -------------------------------
    # A custom criterion expressed in English (note: ConversationalGEval).
    professionalism = ConversationalGEval(
        name="Professionalism",
        criteria=(
            "Across all turns, does the assistant stay calm, empathetic and "
            "professional even when the user is angry?"
        ),
        evaluation_params=[MultiTurnParams.CONTENT],
        model=JUDGE,
    )

    # A built-in metric: is every assistant turn relevant given the dialogue?
    relevancy = TurnRelevancyMetric(threshold=0.7, model=JUDGE)

    # ---- Step 3: evaluate the conversation ---------------------------------
    evaluate(test_cases=[conversation], metrics=[professionalism, relevancy])


if __name__ == "__main__":
    main()
