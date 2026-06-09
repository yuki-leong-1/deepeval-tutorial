"""
System / operational metrics — ❌ NOT LLM-judge metrics → mostly SEPARATE
=========================================================================
From the image:  time to first audio · end-to-end turn latency ·
                 cost per minute · failure rate

CORE CONCEPT
------------
These are NOT about the CONTENT of an output, so there is nothing for an
LLM judge to score. They're operational telemetry you MEASURE by instrumenting
your app (timers, token counters, error counts). That gives two kinds:

  (A) PER-TURN gates — a value exists for each turn (latency, time-to-first-
      audio, per-turn cost). DeepEval CAN host these: stash the measured value
      on the test case (`completion_time`, `token_cost`, or `metadata`) and wrap
      a threshold check as a BaseMetric. Then they run together in `evaluate()`
      just like quality metrics — handy as CI gates ("fail if p95 latency > 2s").

  (B) AGGREGATE rates — only defined OVER MANY turns (cost PER MINUTE,
      failure RATE). There is no per-test-case value, so they are NOT DeepEval
      metrics. You compute them with plain Python from your collected records.
      => kept SEPARATE on purpose (bottom of this file).

So the honest answer for this group: latency / time-to-first-audio are
measurable as DeepEval gates; cost-per-minute and failure-rate are measured
separately as aggregates.

RUN (no key required)
---------------------
    uv run python pipeline_metrics/04_system_metrics.py
"""

from dataclasses import dataclass

import _common  # noqa: F401  — imported for its UTF-8 console fix (Windows-safe rich output)
from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


# ============================================================================
# (A) Per-turn gates as DeepEval metrics
# ============================================================================
class MaxLatencyMetric(BaseMetric):
    """Pass if the turn's end-to-end latency (LLMTestCase.completion_time) <= max."""

    def __init__(self, max_seconds: float = 2.0):
        self.threshold = max_seconds

    def measure(self, tc: LLMTestCase, *a, **k) -> float:
        self.score = float(tc.completion_time or 0.0)
        self.success = self.score <= self.threshold
        self.reason = f"turn latency {self.score:.2f}s (limit {self.threshold:.2f}s)"
        return self.score

    async def a_measure(self, tc, *a, **k):
        return self.measure(tc)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Max turn latency"


class TimeToFirstAudioMetric(BaseMetric):
    """Pass if time-to-first-audio (carried in metadata['ttfa_s']) <= max."""

    def __init__(self, max_seconds: float = 1.0):
        self.threshold = max_seconds

    def measure(self, tc: LLMTestCase, *a, **k) -> float:
        self.score = float((tc.metadata or {}).get("ttfa_s", 0.0))
        self.success = self.score <= self.threshold
        self.reason = f"time to first audio {self.score:.2f}s (limit {self.threshold:.2f}s)"
        return self.score

    async def a_measure(self, tc, *a, **k):
        return self.measure(tc)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Time to first audio"


# ============================================================================
# (B) Aggregate rates — plain Python, NOT DeepEval metrics
# ============================================================================
@dataclass
class TurnRecord:
    text: str
    latency_s: float
    ttfa_s: float
    cost_usd: float
    audio_seconds: float
    failed: bool


def cost_per_minute(records: list[TurnRecord]) -> float:
    total_cost = sum(r.cost_usd for r in records)
    total_minutes = sum(r.audio_seconds for r in records) / 60.0
    return total_cost / total_minutes if total_minutes else 0.0


def failure_rate(records: list[TurnRecord]) -> float:
    return sum(r.failed for r in records) / len(records) if records else 0.0


def main() -> None:
    # Pretend these came from instrumenting a few live turns.
    records = [
        TurnRecord("turn 1", latency_s=1.4, ttfa_s=0.6, cost_usd=0.012, audio_seconds=8, failed=False),
        TurnRecord("turn 2", latency_s=2.7, ttfa_s=1.3, cost_usd=0.020, audio_seconds=11, failed=False),
        TurnRecord("turn 3", latency_s=0.0, ttfa_s=0.0, cost_usd=0.000, audio_seconds=0, failed=True),
    ]

    # ---- (A) per-turn gates via DeepEval ----------------------------------
    # Map each record onto a test case, putting the measured numbers where the
    # metrics look for them (completion_time + metadata).
    test_cases = [
        LLMTestCase(
            input=r.text,
            actual_output="<assistant audio reply>",
            completion_time=r.latency_s,
            token_cost=r.cost_usd,
            metadata={"ttfa_s": r.ttfa_s},
        )
        for r in records if not r.failed   # failed turns have no latency to gate
    ]
    print(">>> (A) Per-turn latency / time-to-first-audio gates (DeepEval):\n")
    evaluate(
        test_cases=test_cases,
        metrics=[MaxLatencyMetric(max_seconds=2.0), TimeToFirstAudioMetric(max_seconds=1.0)],
    )

    # ---- (B) aggregate rates computed separately --------------------------
    print("\n>>> (B) Aggregate metrics (plain Python, NOT a DeepEval metric):")
    print(f"    cost per minute : ${cost_per_minute(records):.4f}")
    print(f"    failure rate    : {failure_rate(records):.1%}")


if __name__ == "__main__":
    main()
