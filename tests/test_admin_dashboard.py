import json

from core import admin_dashboard


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_collects_runs_metrics_and_failures_across_workspaces(tmp_path):
    data_dir = tmp_path / "data"
    runs_dir = data_dir / "run_states"
    _write_json(
        runs_dir / "run-failed.json",
        {
            "run_id": "run-failed",
            "status": "failed",
            "failed_phase": "phase1_profile",
            "error_details": {"code": "bad_input", "phase": "phase1_profile"},
        },
    )
    _write_json(
        data_dir / "users" / "alice" / "runs" / "run-complete.json",
        {"run_id": "run-complete", "status": "completed"},
    )
    metrics_path = data_dir / "users" / "alice" / "metrics" / "run-complete.jsonl"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "agent": "profiler",
                "metric": "agent_duration_seconds",
                "value": 2.5,
            }
        )
        + "\n"
        + json.dumps(
            {
                "agent": "profiler",
                "metric": "api_call",
                "details": {"input_tokens": 100, "output_tokens": 20},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dashboard = admin_dashboard.collect_admin_statistics(data_dir, runs_dir)

    assert dashboard["total_runs"] == 2
    assert dashboard["failed_runs"] == 1
    assert dashboard["failure_rate"] == 0.5
    assert dashboard["status_counts"] == {"failed": 1, "completed": 1}
    assert dashboard["common_failures"] == [{"Failure": "bad_input", "Count": 1}]
    assert dashboard["agent_metrics"][0]["Agent"] == "profiler"
    assert dashboard["agent_metrics"][0]["API calls"] == 1
    assert dashboard["input_tokens"] == 100
    assert dashboard["output_tokens"] == 20


def test_admin_auth_requires_configured_matching_credentials(monkeypatch):
    monkeypatch.setattr(admin_dashboard, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(admin_dashboard, "ADMIN_PASSWORD", "pässword")

    assert admin_dashboard.verify_admin_credentials("admin", "pässword")
    assert not admin_dashboard.verify_admin_credentials("admin", "wrong")
    assert not admin_dashboard.verify_admin_credentials("other", "pässword")

    monkeypatch.setattr(admin_dashboard, "ADMIN_PASSWORD", "")
    assert not admin_dashboard.verify_admin_credentials("admin", "pässword")
