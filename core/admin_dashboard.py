import hmac
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import ADMIN_PASSWORD, ADMIN_USERNAME, DATA_DIR, RUNS_DIR

_RUN_ID_PATTERN = re.compile(r"[^A-Za-z0-9._-]")


def admin_auth_configured() -> bool:
    return bool(ADMIN_USERNAME and ADMIN_PASSWORD)


def verify_admin_credentials(username: str, password: str) -> bool:
    if not admin_auth_configured():
        return False
    username_matches = hmac.compare_digest(
        username.encode("utf-8"), ADMIN_USERNAME.encode("utf-8")
    )
    password_matches = hmac.compare_digest(
        password.encode("utf-8"), ADMIN_PASSWORD.encode("utf-8")
    )
    return username_matches and password_matches


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as source:
            data = json.load(source)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _run_sources(data_dir: Path, runs_dir: Path) -> List[tuple[str, Path, Path]]:
    sources = [("default", runs_dir, data_dir / "metrics")]
    users_dir = data_dir / "users"
    if users_dir.is_dir():
        for user_dir in users_dir.iterdir():
            if user_dir.is_dir():
                sources.append((user_dir.name, user_dir / "runs", user_dir / "metrics"))
    return sources


def _read_metrics(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except OSError:
        return []
    return events


def collect_admin_statistics(
    data_dir: Path = DATA_DIR,
    runs_dir: Path = RUNS_DIR,
) -> Dict[str, Any]:
    runs = []
    agent_durations: Dict[str, List[float]] = defaultdict(list)
    agent_api_calls: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    total_api_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for owner, owner_runs_dir, metrics_dir in _run_sources(data_dir, runs_dir):
        if not owner_runs_dir.is_dir():
            continue
        for state_path in owner_runs_dir.glob("*.json"):
            state = _read_json(state_path)
            if state is None:
                continue
            run_id = str(state.get("run_id") or state_path.stem)
            status = str(state.get("status") or "unknown").lower()
            updated_at = datetime.fromtimestamp(
                state_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            error_details = state.get("error_details") or {}
            if not isinstance(error_details, dict):
                error_details = {}
            error_code = error_details.get("code") or error_details.get("exception_type")
            failed_phase = state.get("failed_phase") or error_details.get("phase")
            runs.append(
                {
                    "Run ID": run_id,
                    "User": owner,
                    "Status": status,
                    "Updated": updated_at,
                    "Failed phase": failed_phase or "",
                    "Error code": error_code or "",
                    "_updated_timestamp": state_path.stat().st_mtime,
                }
            )
            statuses[status] += 1
            if status == "failed":
                failure_category = str(error_code or failed_phase or "unknown failure")
                failures[failure_category] += 1

            safe_run_id = _RUN_ID_PATTERN.sub("_", run_id)
            for event in _read_metrics(metrics_dir / f"{safe_run_id}.jsonl"):
                agent = str(event.get("agent") or "unknown")
                metric = event.get("metric")
                if metric == "agent_duration_seconds":
                    try:
                        agent_durations[agent].append(float(event.get("value", 0)))
                    except (TypeError, ValueError):
                        continue
                elif metric == "api_call":
                    agent_api_calls[agent] += 1
                    total_api_calls += 1
                    details = event.get("details") or {}
                    total_input_tokens += int(details.get("input_tokens", 0) or 0)
                    total_output_tokens += int(details.get("output_tokens", 0) or 0)

    runs.sort(key=lambda run: run["_updated_timestamp"], reverse=True)
    for run in runs:
        run.pop("_updated_timestamp", None)

    agent_metrics = []
    for agent, samples in agent_durations.items():
        ordered = sorted(samples)
        p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
        agent_metrics.append(
            {
                "Agent": agent,
                "Runs": len(samples),
                "Avg duration (s)": round(sum(samples) / len(samples), 3),
                "P95 duration (s)": round(ordered[p95_index], 3),
                "API calls": agent_api_calls[agent],
            }
        )
    agent_metrics.sort(key=lambda item: item["Avg duration (s)"], reverse=True)

    total_runs = len(runs)
    failed_runs = statuses["failed"]
    return {
        "runs": runs,
        "status_counts": dict(statuses),
        "total_runs": total_runs,
        "failed_runs": failed_runs,
        "active_runs": sum(
            count for status, count in statuses.items() if status in {"initialized", "running"}
        ),
        "failure_rate": failed_runs / total_runs if total_runs else 0.0,
        "common_failures": [
            {"Failure": failure, "Count": count}
            for failure, count in failures.most_common(10)
        ],
        "agent_metrics": agent_metrics,
        "api_calls": total_api_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }


def get_system_health(data_dir: Path = DATA_DIR) -> Dict[str, Any]:
    health: Dict[str, Any] = {"sampled_at": datetime.now(timezone.utc).isoformat()}
    try:
        import psutil

        memory = psutil.virtual_memory()
        health.update(
            {
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_percent": memory.percent,
            }
        )
    except (ImportError, OSError):
        health["memory_error"] = "System memory metrics unavailable."

    try:
        disk = shutil.disk_usage(data_dir)
        health.update(
            {
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_used_percent": (disk.used / disk.total * 100) if disk.total else 0,
            }
        )
    except OSError:
        health["disk_error"] = "Disk metrics unavailable."
    return health
