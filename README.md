# DeepEval Tutorial — Learn LLM Evaluation Step by Step

A hands-on, progressive tutorial for [**DeepEval**](https://deepeval.com/docs/introduction),
the open-source "Pytest for LLMs". Every lesson is a small, runnable Python file
with a **Core Concept** header explaining the idea before the code.

Built with **[uv](https://docs.astral.sh/uv/)** for a fast, reproducible environment.
Verified against **DeepEval 4.0.5 / Python 3.12**.

---

## What is DeepEval? (the 60-second version)

DeepEval lets you **measure the quality of LLM outputs** the same way unit tests
measure code. You assert things like *"this answer is faithful to the retrieved
context"* or *"this chatbot stayed professional"*, and DeepEval scores it.

Four concepts hold the whole framework together:

| Concept | What it is | Analogy |
|---|---|---|
| **Test Case** | One interaction: `input` + `actual_output` (+ optional `expected_output`, `context`, …) | A single unit-test input/output |
| **Metric** | Scoring logic returning `score` (0–1), `reason`, and `is_successful()` | An assertion |
| **Dataset / Golden** | A reusable collection of fixed inputs (a *Golden* is a test case minus the output) | A test suite / fixtures |
| **Evaluation** | Running metrics over test cases — end-to-end *or* per-component via tracing | The test run |

Most metrics are **"LLM-as-a-judge"**: they call an LLM (OpenAI by default) to
grade your LLM's output, which is why you need an API key for almost everything.

---

## Setup (with uv)

### 1. Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed (`uv --version`)
- An OpenAI API key (for the metrics that use an LLM judge)

### 2. Install dependencies
The project is already initialized. From the project root:

```bash
uv sync
```

This creates a `.venv` pinned to Python 3.12 and installs `deepeval` + `python-dotenv`
exactly as locked in `uv.lock`. (If you're starting from scratch, the equivalent is
`uv init --python 3.12` then `uv add deepeval python-dotenv`.)

### 3. Add your API key
```bash
# copy the template, then edit .env and paste your key
cp .env.example .env      # PowerShell: Copy-Item .env.example .env
```
`.env`:
```
OPENAI_API_KEY=sk-your-key-here
```
The tutorials auto-load this via `python-dotenv` — no `export` needed.

### 4. Verify the install (no key required)
```bash
uv run python tutorials/00_smoke_test.py
```
You should see the DeepEval version and a constructed test case.

> **Why `uv run`?** It executes the command inside the project's virtual
> environment without you having to activate it manually.

---

## The Tutorials

Work through them in order — each builds on the last. Run any file with
`uv run python tutorials/<file>.py`.

| # | File | Core concept you'll learn |
|---|------|---------------------------|
| 00 | `00_smoke_test.py` | Anatomy of an `LLMTestCase`; verify install (offline) |
| 01 | `01_first_test_case.py` | `LLMTestCase` + a metric + `.measure()` — the smallest eval |
| 02 | `02_rag_metrics.py` | The 5 RAG metrics: pinpoint retriever vs generator failures |
| 03 | `03_custom_geval.py` | **GEval** — invent your own metric from plain English |
| 04 | `04_datasets_and_goldens.py` | **Datasets & Goldens** — evaluate at scale, catch regressions |
| 05 | `05_conversational.py` | Multi-turn / chatbot evaluation with `ConversationalTestCase` |
| 06 | `06_safety_metrics.py` | Toxicity, Bias & Hallucination (lower score = safer) |
| 07 | `07_pytest_integration.py` | `assert_test` + `deepeval test run` for CI/CD gates |
| 08 | `08_component_tracing.py` | `@observe` tracing — score each step of an agent/RAG pipeline |

### 01 — Your first test case
**Concept:** a test case pairs an `input` with your app's `actual_output`; a
metric scores it. We use **GEval** to grade *Correctness* against an
`expected_output`, then call `.measure()` to get `score` + `reason`.
```bash
uv run python tutorials/01_first_test_case.py
```

### 02 — RAG metrics
**Concept:** a RAG app = retriever + generator. DeepEval's five RAG metrics tell
you *which half* is broken:
- **Generator:** `AnswerRelevancyMetric`, `FaithfulnessMetric`
- **Retriever:** `ContextualRelevancyMetric`, `ContextualRecallMetric`, `ContextualPrecisionMetric`

The example deliberately plants a hallucinated claim — watch **Faithfulness**
catch it. Uses `evaluate()` to run all five at once.
```bash
uv run python tutorials/02_rag_metrics.py
```

### 03 — Custom metrics with GEval
**Concept:** when built-in metrics don't fit, describe your own. `criteria` gives
a one-liner; `evaluation_steps` gives an explicit, reproducible checklist; a
`rubric` pins score ranges to meanings.
```bash
uv run python tutorials/03_custom_geval.py
```

### 04 — Datasets & Goldens
**Concept:** a **Golden** is "a test case minus the output". A **Dataset** is a
fixed collection of goldens you re-run on every change to catch regressions. The
loop is: *load goldens → generate `actual_output` with your app → wrap as test
cases → `evaluate()`*. Loads from `data/goldens.csv`.
```bash
uv run python tutorials/04_datasets_and_goldens.py
```

### 05 — Conversational (multi-turn)
**Concept:** chatbots are dialogues, not single answers. A
`ConversationalTestCase` holds a list of `Turn`s; conversational metrics
(`ConversationalGEval`, `TurnRelevancyMetric`, …) judge the whole conversation —
e.g. did it stay professional across all turns?
```bash
uv run python tutorials/05_conversational.py
```

### 06 — Safety & hallucination
**Concept:** `ToxicityMetric`, `BiasMetric`, `HallucinationMetric` flag unsafe or
untrue output. **Gotcha:** for these, a *higher* score is *worse* — they pass
when `score <= threshold`. Hallucination compares the answer against
ground-truth `context` (not `retrieval_context`).
```bash
uv run python tutorials/06_safety_metrics.py
```

### 07 — Pytest integration (CI/CD)
**Concept:** swap `evaluate()` for `assert_test()` to make a failing metric fail a
build. Run with DeepEval's pytest-based runner (note: **not** `python`):
```bash
uv run deepeval test run tutorials/07_pytest_integration.py
# faster + resilient: parallel, ignore judge errors
uv run deepeval test run tutorials/07_pytest_integration.py -n 2 -i
```
Drop this into GitHub Actions and a quality regression never ships.

### 08 — Component-level evaluation (tracing)
**Concept:** end-to-end evals treat your app as a black box. `@observe` traces
each internal function as a "span" (retriever, llm, tool, agent), attaches
per-span metrics, and `dataset.evals_iterator()` runs them — so you see
*"retriever 0.9, generator 0.4"* instead of one opaque number.
```bash
uv run python tutorials/08_component_tracing.py
```

---

## Cheat sheet

```python
# One metric, one case — quick experiments
metric.measure(test_case)
print(metric.score, metric.reason, metric.is_successful())

# Many metrics, many cases — the normal report
from deepeval import evaluate
evaluate(test_cases=[...], metrics=[...])

# Make it a CI gate — raises if any metric fails
from deepeval import assert_test
assert_test(test_case, [metric1, metric2])
#   run with:  uv run deepeval test run your_file.py
```

**Pick few metrics.** The docs recommend **≤ 5 total** — ~2–3 system metrics
(e.g. relevancy, faithfulness) plus 1–2 custom GEval metrics for what uniquely
matters to your product. More than that and signal drowns in noise.

**Reference vs referenceless.** Metrics needing `expected_output`/`context`
(Correctness, Contextual Recall, Hallucination) are for *development*. Only
referenceless ones (Answer Relevancy, Faithfulness, Toxicity) work in
*production* monitoring where there's no ground truth.

---

## Using a different judge (no OpenAI)

Every metric takes a `model=` argument. Options:

```python
# A different OpenAI model
GEval(..., model="gpt-4o")

# A local model via Ollama (no API key, runs on your machine)
#   1) ollama pull llama3.1     2) then in code:
from deepeval.models import OllamaModel
GEval(..., model=OllamaModel(model="llama3.1"))
```
DeepEval also supports Azure OpenAI, Anthropic, Gemini, and fully custom judges
(subclass `DeepEvalBaseLLM`). See the docs:
https://deepeval.com/docs/metrics-introduction#using-a-custom-llm

---

## Going further

- **Synthetic data** — auto-generate goldens from your documents with the
  `Synthesizer` / `dataset.generate_goldens_from_docs(...)`.
- **Red teaming / safety at scale** — the companion project
  [DeepTeam](https://www.trydeepteam.com/) probes for jailbreaks, PII leakage, bias.
- **Confident AI** — run `uv run deepeval login` to push results, datasets, and
  regression history to a team dashboard (`evaluate(...)` then `deepeval view`).

## Project layout

```
DeepEval-Tutorial/
├── README.md              ← you are here
├── pyproject.toml         ← uv project + deps
├── uv.lock                ← pinned, reproducible versions
├── .env.example           ← copy to .env and add your key
├── data/
│   └── goldens.csv        ← sample dataset for tutorial 04
└── tutorials/
    ├── _setup.py          ← shared .env loader + key check
    ├── 00_smoke_test.py
    ├── 01_first_test_case.py
    ├── 02_rag_metrics.py
    ├── 03_custom_geval.py
    ├── 04_datasets_and_goldens.py
    ├── 05_conversational.py
    ├── 06_safety_metrics.py
    ├── 07_pytest_integration.py
    └── 08_component_tracing.py
```

## Troubleshooting

- **`OPENAI_API_KEY is not set`** — create `.env` from `.env.example` and add your key.
- **`deepeval: command not found`** — prefix with uv: `uv run deepeval ...`.
- **Rate limits / 429s** — pass `AsyncConfig(max_concurrent=3, throttle_value=2)` to
  `evaluate(...)`, or add `-n 1` to `deepeval test run`.
- **Costs** — the tutorials default to the cheap `gpt-4o-mini` judge. Each run is a
  few cents at most.

---

*Sources: [DeepEval Docs](https://deepeval.com/docs/introduction) · [Metrics](https://deepeval.com/docs/metrics-introduction) · [Test Cases](https://deepeval.com/docs/evaluation-test-cases) · [Datasets](https://deepeval.com/docs/evaluation-datasets)*
