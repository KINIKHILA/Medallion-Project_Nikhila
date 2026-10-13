import json
import re
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import (
    GEMINI_MODEL,
    GITHUB_MODEL,
    GROQ_MODEL,
    LLM_INPUT_COST_PER_1M_TOKENS,
    LLM_OUTPUT_COST_PER_1M_TOKENS,
    LLM_PROVIDER,
    METRICS_DIR,
    OPENAI_MODEL,
)
from core.user_workspace import user_directory


def _metrics_path(run_id: str) -> Path:
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]", "_", run_id)
    metrics_dir = user_directory("metrics", METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir / f"{safe_run_id}.jsonl"


def record_metric(
    run_id: str,
    agent: str,
    metric: str,
    value: Any,
    unit: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "agent": agent,
        "metric": metric,
        "value": value,
        "unit": unit,
        "details": details or {},
    }
    try:
        with _metrics_path(str(run_id)).open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
    except OSError:
        return


def _process_memory_mb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None


class AgentMemorySampler:
    """Sample process RSS while an agent runs and record its observed peak."""

    def __init__(self, run_id: str, agent: str, interval_seconds: float = 0.5) -> None:
        self.run_id = run_id
        self.agent = agent
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.peak_memory_mb = _process_memory_mb()
        self.thread: Optional[threading.Thread] = None
        self.stopped = False
        if self.peak_memory_mb is not None:
            record_metric(
                run_id,
                agent,
                "memory_start_mb",
                self.peak_memory_mb,
                unit="MB",
            )
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            memory_mb = _process_memory_mb()
            if memory_mb is not None:
                self.peak_memory_mb = max(self.peak_memory_mb or memory_mb, memory_mb)

    def stop(self) -> Optional[float]:
        if self.stopped:
            return self.peak_memory_mb
        self.stopped = True
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=max(1.0, self.interval_seconds * 2))
            final_memory = _process_memory_mb()
            if final_memory is not None:
                self.peak_memory_mb = max(self.peak_memory_mb or final_memory, final_memory)
            if self.peak_memory_mb is not None:
                record_metric(
                    self.run_id,
                    self.agent,
                    "memory_peak_mb",
                    self.peak_memory_mb,
                    unit="MB",
                )
        return self.peak_memory_mb


def _configured_model() -> str:
    return {
        "groq": GROQ_MODEL,
        "gemini": GEMINI_MODEL,
        "openai": OPENAI_MODEL,
        "github": GITHUB_MODEL,
    }.get(LLM_PROVIDER, "unknown")


def _optional_rate(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    return rate if rate >= 0 else None


def record_api_call(run_id: str, agent: str, message: Any) -> Dict[str, Any]:
    usage = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    if not usage:
        usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}

    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    input_rate = _optional_rate(LLM_INPUT_COST_PER_1M_TOKENS)
    output_rate = _optional_rate(LLM_OUTPUT_COST_PER_1M_TOKENS)
    estimated_cost = None
    if input_rate is not None and output_rate is not None:
        estimated_cost = round(
            (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000,
            8,
        )

    details = {
        "provider": LLM_PROVIDER,
        "model": response_metadata.get("model_name") or _configured_model(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
    }
    record_metric(
        run_id,
        agent,
        "api_call",
        1,
        unit="call",
        details=details,
    )
    return details


def get_run_metrics(run_id: str) -> List[Dict[str, Any]]:
    metrics_path = _metrics_path(str(run_id))
    if not metrics_path.exists():
        return []
    entries = []
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        for line in metrics_file:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def summarize_run_metrics(run_id: str) -> Dict[str, Any]:
    events = get_run_metrics(run_id)
    durations = defaultdict(float)
    api_events = [event for event in events if event.get("metric") == "api_call"]
    cache_events = [event for event in events if event.get("metric") == "cache_lookup"]
    cache_hits = sum(
        event.get("details", {}).get("status") == "hit" for event in cache_events
    )
    cache_misses = sum(
        event.get("details", {}).get("status") == "miss" for event in cache_events
    )
    memory_events = [
        event
        for event in events
        if event.get("metric") == "memory_peak_mb" and event.get("value") is not None
    ]
    input_tokens = sum(int(event.get("details", {}).get("input_tokens", 0)) for event in api_events)
    output_tokens = sum(int(event.get("details", {}).get("output_tokens", 0)) for event in api_events)
    cost_values = [
        event["details"]["estimated_cost_usd"]
        for event in api_events
        if event.get("details", {}).get("estimated_cost_usd") is not None
    ]
    for event in events:
        if event.get("metric") == "agent_duration_seconds":
            durations[event.get("agent", "unknown")] += float(event.get("value", 0))

    return {
        "run_id": str(run_id),
        "agent_durations_seconds": dict(durations),
        "api_calls": len(api_events),
        "cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "lookups": len(cache_events),
        },
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "peak_memory_mb": max((float(event["value"]) for event in memory_events), default=None),
        "estimated_cost_usd": round(sum(cost_values), 8) if len(cost_values) == len(api_events) and api_events else None,
        "cost_estimate_configured": bool(api_events) and len(cost_values) == len(api_events),
        "events": events,
    }
