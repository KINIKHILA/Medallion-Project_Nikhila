from types import SimpleNamespace

import pytest

from core import performance


def test_metrics_summary_tracks_agents_calls_memory_and_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(performance, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(performance, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(performance, "OPENAI_MODEL", "test-model")
    monkeypatch.setattr(performance, "LLM_INPUT_COST_PER_1M_TOKENS", "2.0")
    monkeypatch.setattr(performance, "LLM_OUTPUT_COST_PER_1M_TOKENS", "4.0")

    message = SimpleNamespace(
        usage_metadata={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
        response_metadata={"model_name": "test-model"},
    )
    usage = performance.record_api_call("run-1", "profiler", message)
    performance.record_metric("run-1", "profiler", "agent_duration_seconds", 2.5, "seconds")
    performance.record_metric("run-1", "profiler", "memory_peak_mb", 128.0, "MB")
    performance.record_metric(
        "run-1", "profiler", "cache_lookup", 1, details={"status": "hit"}
    )
    performance.record_metric(
        "run-1", "sttm_generator", "cache_lookup", 1, details={"status": "miss"}
    )

    summary = performance.summarize_run_metrics("run-1")

    assert usage["estimated_cost_usd"] == pytest.approx(0.004)
    assert summary["api_calls"] == 1
    assert summary["total_tokens"] == 1500
    assert summary["peak_memory_mb"] == 128.0
    assert summary["agent_durations_seconds"]["profiler"] == 2.5
    assert summary["cache"] == {"hits": 1, "misses": 1, "lookups": 2}
    assert summary["estimated_cost_usd"] == pytest.approx(0.004)


def test_cost_is_unavailable_without_configured_rates(tmp_path, monkeypatch):
    monkeypatch.setattr(performance, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(performance, "LLM_INPUT_COST_PER_1M_TOKENS", "")
    monkeypatch.setattr(performance, "LLM_OUTPUT_COST_PER_1M_TOKENS", "")

    usage = performance.record_api_call(
        "run-2",
        "reporter",
        SimpleNamespace(usage_metadata={"input_tokens": 20, "output_tokens": 10}),
    )

    assert usage["estimated_cost_usd"] is None
    assert performance.summarize_run_metrics("run-2")["estimated_cost_usd"] is None
