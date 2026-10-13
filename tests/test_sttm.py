"""Tests for STTM context preparation and extracting generated mapping rows."""

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import sttm_generator
from core import sttm_versions, user_workspace

STTM_COLUMNS = [
    "source_schema",
    "source_table",
    "source_column",
    "target_schema",
    "target_table",
    "target_column",
    "transformation_type",
    "transformation_logic",
]

BRONZE_ROW = {
    "source_schema": "landing",
    "source_table": "orders",
    "source_column": "order_id",
    "target_schema": "bronze",
    "target_table": "orders",
    "target_column": "order_id",
    "transformation_type": "copy",
    "transformation_logic": "Copy source value.",
}
SILVER_ROW = {
    "source_schema": "bronze",
    "source_table": "orders_bronze",
    "source_column": "order_id",
    "target_schema": "silver",
    "target_table": "orders",
    "target_column": "order_id",
    "transformation_type": "copy",
    "transformation_logic": "Keep approved identifier.",
}
GOLD_ROW = {
    "source_schema": "silver",
    "source_table": "orders",
    "source_column": "order_id",
    "target_schema": "gold",
    "target_table": "order_summary",
    "target_column": "order_id",
    "transformation_type": "group_key",
    "transformation_logic": "Group by order_id.",
}


def _write_sttm(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    sttm_path = tmp_path / name
    with sttm_path.open("w", newline="", encoding="utf-8") as sttm_file:
        writer = csv.DictWriter(sttm_file, fieldnames=STTM_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in STTM_COLUMNS}
            for row in rows
        )
    return sttm_path


def _write_parquet(tmp_path: Path, name: str, dataframe: pd.DataFrame) -> Path:
    parquet_path = tmp_path / name
    dataframe.to_parquet(parquet_path, index=False)
    return parquet_path


def _mock_agent(contents: list[str]):
    """Return a simple agent double that exposes messages for parser tests."""
    class MockAgent:
        def invoke(self, request=None):
            return {"messages": [SimpleNamespace(content=content) for content in contents]}

    return MockAgent()


class TestPrepareBronzeContext:
    def test_returns_profile_dict(self, tmp_path):
        profile = {
            "inspection": {
                "files": [
                    {
                        "path": str(tmp_path / "orders.csv"),
                        "shape": [2, 2],
                        "columns": ["order_id", "amount"],
                    }
                ]
            },
            "statistics": {"files": []},
        }
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")

        context = sttm_generator._prepare_bronze_context(profile_path)

        assert context["layer"] == "bronze"
        assert context["profile"] == profile
        assert context["source_files"][0]["table"] == "orders"
        assert context["source_files"][0]["columns"] == ["order_id", "amount"]

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sttm_generator._prepare_bronze_context(tmp_path / "missing-profile.json")


class TestSttmVersionHistory:
    def test_versions_compare_restore_and_audit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(user_workspace, "DATA_DIR", tmp_path / "data")
        audit_events = []

        class FakeAuditLogger:
            def log(self, agent, action, **details):
                audit_events.append({"agent": agent, "action": action, **details})

        with user_workspace.use_user_workspace("alice"):
            first = sttm_versions.record_sttm_version(
                "bronze", [BRONZE_ROW], "run-1", "generated", FakeAuditLogger()
            )
            unchanged = sttm_versions.record_sttm_version(
                "bronze", [BRONZE_ROW], "run-2", "approved", FakeAuditLogger()
            )
            changed_row = {**BRONZE_ROW, "transformation_logic": "Normalize order IDs."}
            second = sttm_versions.record_sttm_version(
                "bronze", [changed_row], "run-2", "approved", FakeAuditLogger()
            )
            diff = sttm_versions.compare_sttm_versions(
                "bronze", first["version_id"], second["version_id"]
            )
            restored = sttm_versions.record_sttm_version(
                "bronze",
                first["rows"],
                "run-3",
                "restored",
                FakeAuditLogger(),
                restored_from=first["version_id"],
            )
            version_count = len(sttm_versions.list_sttm_versions("bronze"))

        assert unchanged["version_id"] == first["version_id"]
        assert version_count == 3
        assert any(line.startswith("-") for line in diff)
        assert any(line.startswith("+") for line in diff)
        assert second["previous_version_id"] == first["version_id"]
        assert restored["previous_version_id"] == second["version_id"]
        assert restored["restored_from"] == first["version_id"]
        assert [event["action"] for event in audit_events] == [
            "version_created",
            "version_unchanged",
            "version_created",
            "version_created",
        ]


class TestPrepareSilverContext:
    def test_filters_to_approved_columns(self, tmp_path):
        bronze_path = _write_parquet(
            tmp_path,
            "orders_bronze.parquet",
            pd.DataFrame({"order_id": [1], "amount": [10.0], "internal_note": ["x"]}),
        )
        sttm_path = _write_sttm(tmp_path, "bronze_sttm.csv", [
            {**BRONZE_ROW, "target_column": "order_id"},
            {**BRONZE_ROW, "source_column": "amount", "target_column": "amount"},
        ])

        context = sttm_generator._prepare_silver_context([bronze_path], sttm_path)

        bronze_file = context["bronze_files"][0]
        assert [column["name"] for column in bronze_file["columns"]] == [
            "order_id",
            "amount",
        ]
        assert bronze_file["excluded_columns"] == ["internal_note"]

    def test_returns_filename_not_full_path(self, tmp_path):
        bronze_path = _write_parquet(
            tmp_path,
            "orders_bronze.parquet",
            pd.DataFrame({"order_id": [1]}),
        )
        sttm_path = _write_sttm(tmp_path, "bronze_sttm.csv", [BRONZE_ROW])

        context = sttm_generator._prepare_silver_context([bronze_path], sttm_path)

        bronze_file = context["bronze_files"][0]
        assert bronze_file["file_name"] == "orders_bronze.parquet"
        assert not Path(bronze_file["file_name"]).is_absolute()


class TestPrepareGoldContext:
    def test_filters_to_approved_columns(self, tmp_path):
        silver_path = _write_parquet(
            tmp_path,
            "orders_silver.parquet",
            pd.DataFrame({"order_id": [1], "customer_id": [8], "debug_value": ["x"]}),
        )
        silver_sttm = _write_sttm(tmp_path, "silver_sttm.csv", [
            {**SILVER_ROW, "target_column": "order_id"},
            {**SILVER_ROW, "source_column": "customer_id", "target_column": "customer_id"},
        ])

        context = sttm_generator._prepare_gold_context([silver_path], silver_sttm)

        silver_file = context["silver_files"][0]
        assert [column["name"] for column in silver_file["columns"]] == [
            "order_id",
            "customer_id",
        ]
        assert silver_file["excluded_columns"] == ["debug_value"]


class TestExtractSttmRows:
    def test_sttm_cache_context_ignores_run_metadata(self):
        first = {
            "profile_path": "profiles/first.json",
            "profile": {"run_id": "first", "created_at": "today", "columns": ["id"]},
        }
        second = {
            "profile_path": "profiles/second.json",
            "profile": {"run_id": "second", "created_at": "tomorrow", "columns": ["id"]},
        }

        assert sttm_generator._stable_cache_context(first) == sttm_generator._stable_cache_context(second)

    def test_extracts_plain_json_array(self):
        rows = [BRONZE_ROW, SILVER_ROW]
        agent = _mock_agent([json.dumps(rows)])

        assert sttm_generator._extract_sttm_rows(agent.invoke({})["messages"]) == rows

    def test_extracts_json_with_fences(self):
        fenced = "```json\n" + json.dumps([GOLD_ROW]) + "\n```"
        agent = _mock_agent([fenced])

        assert sttm_generator._extract_sttm_rows(agent.invoke({})["messages"]) == [GOLD_ROW]

    def test_returns_empty_list_when_no_array(self):
        agent = _mock_agent(["I could not identify any mapping rules."])

        assert sttm_generator._extract_sttm_rows(agent.invoke({})["messages"]) == []

    def test_prefers_last_message(self):
        agent = _mock_agent(
            [
                json.dumps([BRONZE_ROW]),
                "Final rules:\n" + json.dumps([GOLD_ROW]),
            ]
        )

        assert sttm_generator._extract_sttm_rows(agent.invoke({})["messages"]) == [GOLD_ROW]
