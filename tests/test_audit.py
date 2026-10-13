import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.audit as audit_module
from core.audit import AuditLogger


class TestAuditLogger:
	def test_log_creates_file(self, tmp_path, monkeypatch):
		monkeypatch.setattr(audit_module, "AUDIT_DIR", tmp_path)
		logger = AuditLogger("test-run-123")

		logger.log("profiler", "profile_started", file_count=2)

		assert logger.log_path == tmp_path / "test-run-123.jsonl"
		assert logger.log_path.exists()
		logs = logger.get_logs()
		assert len(logs) == 1
		assert logs[0]["run_id"] == "test-run-123"
		assert logs[0]["agent"] == "profiler"
		assert logs[0]["action"] == "profile_started"
		assert logs[0]["file_count"] == 2
		assert "timestamp" in logs[0]
		with logger.log_path.open("r", encoding="utf-8") as log_file:
			assert json.loads(log_file.readline()) == logs[0]

	def test_multiple_logs_appended(self, tmp_path, monkeypatch):
		monkeypatch.setattr(audit_module, "AUDIT_DIR", tmp_path)
		logger = AuditLogger("append-run")

		logger.log("bronze", "started")
		logger.log("bronze", "completed", row_count=12)

		logs = logger.get_logs()
		assert len(logs) == 2
		assert [entry["action"] for entry in logs] == ["started", "completed"]
		assert logs[1]["row_count"] == 12

	def test_get_logs_empty_file(self, tmp_path, monkeypatch):
		monkeypatch.setattr(audit_module, "AUDIT_DIR", tmp_path)
		logger = AuditLogger("missing-run")

		assert not logger.log_path.exists()
		assert logger.get_logs() == []