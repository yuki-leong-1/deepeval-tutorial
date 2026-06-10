# Pipeline Metrics — What Can Be Measured, and Can They Run Together?

Covers every metric from the "Metric Runner" diagram: **can it be measured, how, and can it share a single `evaluate()` call?**

## TL;DR

- **Rule for running metrics together:** inside one `evaluate(test_cases=[tc], metrics=[...])`, every metric runs against the **same `LLMTestCase`**. As long as each metric is a single-turn `BaseMetric` and reads `actual_output` / `expected_output` with the same meaning, they can be batched together.
- **Split by pipeline stage:** translation / STT / TTS / system each get their own file. Metrics within a stage can run together; **cross-stage they cannot** — the same `actual_output` field means "translated text" in the translation stage and "transcribed text" in the STT stage, so mixing them produces meaningless scores.
- DeepEval has no built-in WER / chrF++ / COMET, but **any scorer can be wrapped as a custom `BaseMetric`** and then runs alongside GEval in a single `evaluate()`.
- **System metrics** (latency / cost / failure rate) are not "score the output content" metrics. Most require instrumentation; per-turn threshold checks and aggregate calculations are handled separately.

## Metric Reference

| Metric | Measurable? | Method / Library | File (grouped with) | API key? |
|---|---|---|---|---|
| **COMET / XCOMET** | ✅ yes (heavy) | `unbabel-comet` (neural, ~2 GB model download) → `CometMetric` | `01_translation_metrics.py` (off by default; enable with `USE_COMET=1`) | No (but requires heavy dep) |
| **chrF++** | ✅ yes | `sacrebleu.sentence_chrf(word_order=2)` → `ChrfppMetric` | `01` | No |
| **LLM judge via DeepEval** | ✅ yes | DeepEval native `GEval` | `01` | **Yes** |
| **number accuracy** | ✅ yes | Regex digit extraction + comparison → `NumberAccuracyMetric` (deterministic) | `01` | No |
| **unit accuracy** | ✅ yes | Unit lexicon lookup → `UnitAccuracyMetric` (deterministic) | `01` | No |
| **drug accuracy** | ✅ yes | Semantic judgement → `GEval` (clinical judge) | `01` | **Yes** |
| **negation accuracy** | ✅ yes | Semantic judgement → `GEval` | `01` | **Yes** |
| **frequency accuracy** | ✅ yes | Semantic judgement → `GEval` | `01` | **Yes** |
| **critical fact accuracy** | ✅ yes | Semantic judgement → `GEval` | `01` | **Yes** |
| **WER** | ✅ yes | `jiwer.wer` → `WERMetric` | `02_stt_metrics.py` | No |
| **CER** | ✅ yes | `jiwer.cer` → `CERMetric` | `02` | No |
| **medical term WER** | ✅ yes | Filter to medical terms then compute WER → `MedicalTermWERMetric` | `02` | No |
| **number WER** | ✅ yes | Filter to digits then compute WER → `NumberWERMetric` | `02` | No |
| **ASR loopback WER** | ✅ yes | TTS → ASR round-trip then `jiwer.wer` → `ASRLoopbackWERMetric` | `03_tts_audio_metrics.py` | No |
| **pronunciation flags** | ⚠️ partial | Requires an **audio model / forced alignment** outside DeepEval; pass your pre-computed score in via `metadata` | `03` (reads `metadata`) | No |
| **intelligibility** | ⚠️ partial | Requires a **MOS / STOI prediction model**; pass the 0–1 score in via `metadata` | `03` (reads `metadata`) | No |
| **time to first audio** | ✅ yes (threshold) | Instrument timing → `metadata['ttfa_s']` → `TimeToFirstAudioMetric` | `04_system_metrics.py` (part A) | No |
| **end-to-end turn latency** | ✅ yes (threshold) | Instrument timing → `LLMTestCase.completion_time` → `MaxLatencyMetric` | `04` (part A) | No |
| **cost per minute** | ❌ not a per-turn metric | **Aggregate** = total cost / total audio minutes; compute in plain Python | `04` (part B, **separate**) | No |
| **failure rate** | ❌ not a per-turn metric | **Aggregate** = failed turns / total turns; compute in plain Python | `04` (part B, **separate**) | No |

## Why These Groups (Together vs. Separate)

1. **Translation + medical safety metrics** → both judge **the same translated text** (`input` = source, `actual_output` = machine translation, `expected_output` = reference translation). They share one `LLMTestCase` and run together in `01`.
2. **STT metrics** → judge **transcribed text** (`expected_output` = human transcript, `actual_output` = ASR output). The `actual_output` means something different from translation → must live in `02`. All deterministic, **no key required**.
3. **TTS / audio metrics** → judge **synthesized audio**. Loopback WER can be computed from text; pronunciation / intelligibility scores come from external audio models and are passed in via `metadata` → grouped in `03`.
4. **System metrics** → not scoring "output content":
   - **Per-turn** values (latency, time-to-first-audio) → can be wrapped as DeepEval threshold metrics and run together in `04` part A.
   - **Cross-turn aggregates** (cost per minute, failure rate) → no meaningful per-test-case score exists, so they are **not** DeepEval metrics; computed in plain Python in `04` part B.

> **Key constraint:** `evaluate()` runs **every metric against every test case**. If you mix translation and STT test cases in one call, `WERMetric` will also run on the translation case and produce a meaningless score. Splitting by stage is the fix.

## Score Direction

| Type | Examples | Pass condition |
|---|---|---|
| Higher is better (quality / accuracy) | chrF++, COMET, number/unit accuracy, GEval, intelligibility | `score >= threshold` |
| Lower is better (error rate) | WER, CER, medical-term/number WER, loopback WER, pronunciation flag rate, latency | `score <= threshold` |

## Running the Scripts

```bash
# Install dependencies (already in pyproject): deepeval, jiwer, sacrebleu, python-dotenv
uv sync

# No API key required:
uv run python pipeline_metrics/02_stt_metrics.py
uv run python pipeline_metrics/03_tts_audio_metrics.py
uv run python pipeline_metrics/04_system_metrics.py

# Requires OPENAI_API_KEY (GEval clinical judge) — add key to .env first:
uv run python pipeline_metrics/01_translation_metrics.py

# Also run COMET (requires `uv add unbabel-comet`, downloads a large model):
USE_COMET=1 uv run python pipeline_metrics/01_translation_metrics.py
```

> The lexicons in `_common.py` (`UNIT_LEXICON`, `MEDICAL_TERM_LEXICON`) are small placeholder samples — replace them with your own clinical terminology. The semantic metrics (drug / negation / frequency / critical-fact accuracy) use GEval rather than hard-coded rules because it is more robust and language-pair-agnostic.
