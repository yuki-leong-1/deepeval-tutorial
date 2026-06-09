"""
Tutorial 04 — Datasets & Goldens
================================

CORE CONCEPT
------------
Testing one case at a time doesn't scale. A **Dataset** is a reusable
collection you run every time you change a prompt or model — that's how you
catch regressions.

The key vocabulary:

    Golden   — a "test case minus the output". It stores the fixed inputs
               (input, expected_output, context...) but NOT actual_output,
               because that is generated fresh on every run by your app.

    TestCase — a Golden once you've filled in actual_output from your LLM.

    Dataset  — a collection of Goldens (the stable benchmark) that you
               convert into TestCases at evaluation time.

The standard loop is therefore:
    load goldens  →  for each golden, call your app to get actual_output
                  →  wrap in an LLMTestCase  →  evaluate() them all.

Because the goldens stay fixed, comparing two runs is apples-to-apples.

RUN
---
    uv run python tutorials/04_datasets_and_goldens.py

Requires OPENAI_API_KEY.
"""

from pathlib import Path

from _setup import require_openai_key
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval

DATA = Path(__file__).resolve().parent.parent / "data" / "goldens.csv"
JUDGE = "gpt-4o-mini"


def my_llm_app(question: str) -> str:
    """
    Stand-in for YOUR real LLM application. Replace the body with your own
    OpenAI/Anthropic/RAG call. We fake a few canned answers so the tutorial
    runs without extra setup.
    """
    canned = {
        "What is your return window?":
            "We accept returns within 30 days of delivery.",
        "Do you ship internationally?":
            "Yes! We ship worldwide to 50+ countries.",
        "How long does delivery take?":
            "Most orders arrive in 3 to 5 business days.",
        "Can I cancel an order after placing it?":
            "Orders can be cancelled within the first hour after checkout.",
    }
    return canned.get(question, "I'm not sure, let me check that for you.")


def main() -> None:
    require_openai_key()

    # ---- Step 1: build a dataset of goldens --------------------------------
    # Option A: load from the CSV in /data (columns: input, expected_output).
    dataset = EvaluationDataset()
    dataset.add_goldens_from_csv_file(file_path=str(DATA))

    # Option B (equivalent) — define goldens inline in code:
    #   dataset = EvaluationDataset(goldens=[
    #       Golden(input="What is your return window?",
    #              expected_output="You can return items within 30 days."),
    #   ])
    #   dataset.add_golden(Golden(input="Do you ship internationally?"))

    print(f"Loaded {len(dataset.goldens)} goldens from {DATA.name}\n")

    # ---- Step 2: turn each golden into a test case -------------------------
    # This is where your app actually runs. Each golden -> one LLM call.
    test_cases = []
    for golden in dataset.goldens:
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=my_llm_app(golden.input),  # <-- your app's answer
                expected_output=golden.expected_output,  # carried from the golden
            )
        )

    # ---- Step 3: evaluate the whole dataset at once ------------------------
    correctness = GEval(
        name="Correctness",
        criteria="Is the actual output consistent with the expected output?",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=JUDGE,
    )

    evaluate(test_cases=test_cases, metrics=[correctness])

    # TIP: with a Confident AI account you can store the dataset in the cloud
    # and pull it anywhere, keeping goldens out of your codebase:
    #     dataset.push(alias="Support FAQ")          # upload once
    #     dataset.pull(alias="Support FAQ")          # fetch in CI / teammates


if __name__ == "__main__":
    main()
