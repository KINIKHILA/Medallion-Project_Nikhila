"""Tests for Gold table generation and the Gold agent's execution boundary."""

import csv
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import gold_agent

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


def _make_gold_sttm(tmp_path: Path, rows: list[dict]) -> Path:
    sttm_path = tmp_path / "gold_sttm.csv"
    with sttm_path.open("w", newline="", encoding="utf-8") as sttm_file:
        writer = csv.DictWriter(sttm_file, fieldnames=STTM_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in STTM_COLUMNS}
            for row in rows
        )
    return sttm_path


def _passthrough_sttm(
    tmp_path: Path, source_table: str, columns: list[str]
) -> Path:
    target_table = f"{source_table}_gold"
    rows = [
        {
            "source_schema": "silver",
            "source_table": source_table,
            "source_column": column,
            "target_schema": "gold",
            "target_table": target_table,
            "target_column": column,
            "transformation_type": "copy",
            "transformation_logic": "Copy without changes.",
        }
        for column in columns
    ]
    return _make_gold_sttm(tmp_path, rows)


def _mock_agent_that_calls_tool(tool):
    """Return a mock agent that invokes the supplied tool or ordered tools."""
    from langchain_core.messages import ToolMessage

    tools = list(tool) if isinstance(tool, (list, tuple)) else [tool]

    class MockAgent:
        def invoke(self, request):
            messages = []
            for tool_index, current_tool in enumerate(tools):
                tool_name = getattr(current_tool, "name", None) or getattr(
                    current_tool, "__name__", "tool"
                )
                result = (
                    current_tool.invoke({})
                    if hasattr(current_tool, "invoke")
                    else current_tool()
                )
                messages.append(
                    ToolMessage(
                        content=str(result),
                        name=tool_name,
                        tool_call_id=f"mock-tool-call-{tool_index}",
                    )
                )
            return {"messages": messages}

    return MockAgent()


def _write_silver_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    parquet_path = tmp_path / "sales_silver.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return parquet_path


def _copy_rule(source_table: str, target_table: str, column: str) -> dict:
    return {
        "source_schema": "silver",
        "source_table": source_table,
        "source_column": column,
        "target_schema": "gold",
        "target_table": target_table,
        "target_column": column,
        "transformation_type": "copy",
        "transformation_logic": "Copy without changes.",
    }


class TestApplyGoldRules:
    def _apply(self, tmp_path, monkeypatch, input_files, sttm_path):
        monkeypatch.setattr(gold_agent, "GOLD_DIR", tmp_path / "gold")
        return gold_agent._apply_gold_rules(input_files, sttm_path, "test-run")

    def test_output_written_for_each_target_table(self, tmp_path, monkeypatch):
        input_path = _write_silver_parquet(
            tmp_path, [{"customer_id": 1, "amount": 25.0}]
        )
        sttm_path = _passthrough_sttm(
            tmp_path, "sales", ["customer_id", "amount"]
        )

        output_paths = self._apply(tmp_path, monkeypatch, [input_path], sttm_path)

        assert len(output_paths) == 1
        assert Path(output_paths[0]).name == "sales_gold.parquet"
        assert Path(output_paths[0]).is_file()

    def test_output_is_valid_parquet(self, tmp_path, monkeypatch):
        input_path = _write_silver_parquet(
            tmp_path, [{"customer_id": 1, "amount": 25.0}]
        )
        sttm_path = _passthrough_sttm(
            tmp_path, "sales", ["customer_id", "amount"]
        )

        output_path = self._apply(tmp_path, monkeypatch, [input_path], sttm_path)[0]
        dataframe = pd.read_parquet(output_path)

        assert len(dataframe) == 1
        assert dataframe["customer_id"].tolist() == [1]
        assert dataframe["amount"].tolist() == [25.0]

    def test_multiple_target_tables(self, tmp_path, monkeypatch):
        input_path = _write_silver_parquet(
            tmp_path,
            [
                {"customer_id": 1, "amount": 25.0, "category": "Books"},
                {"customer_id": 2, "amount": 40.0, "category": "Games"},
            ],
        )
        rules = [
            _copy_rule("sales", "customer_sales", "customer_id"),
            _copy_rule("sales", "customer_sales", "amount"),
            _copy_rule("sales", "category_list", "category"),
        ]
        sttm_path = _make_gold_sttm(tmp_path, rules)

        output_paths = self._apply(tmp_path, monkeypatch, [input_path], sttm_path)

        assert {Path(path).name for path in output_paths} == {
            "customer_sales.parquet",
            "category_list.parquet",
        }
        assert len(output_paths) == 2

    def test_column_subset_applied(self, tmp_path, monkeypatch):
        input_path = _write_silver_parquet(
            tmp_path,
            [{"customer_id": 7, "amount": 99.0, "internal_note": "private"}],
        )
        sttm_path = _passthrough_sttm(
            tmp_path, "sales", ["customer_id", "amount"]
        )

        output_path = self._apply(tmp_path, monkeypatch, [input_path], sttm_path)[0]
        dataframe = pd.read_parquet(output_path)

        assert list(dataframe.columns) == ["pk_gold_id", "customer_id", "amount"]
        assert "internal_note" not in dataframe.columns

    def test_postgres_export_creates_or_replaces_table(self, monkeypatch):
        captured = {}

        class FakeEngine:
            def dispose(self):
                captured["disposed"] = True

        sqlalchemy = types.ModuleType("sqlalchemy")
        sqlalchemy.create_engine = lambda url, **kwargs: captured.update(
            {"url": url, **kwargs}
        ) or FakeEngine()
        monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy)
        monkeypatch.setattr(
            gold_agent,
            "DATABASE_URL",
            "postgresql+psycopg://user:password@localhost:5432/analytics",
        )
        monkeypatch.setattr(gold_agent, "GOLD_DB_SCHEMA", "analytics")

        class FakeDataFrame:
            def to_sql(self, **kwargs):
                captured["to_sql"] = kwargs

        assert gold_agent._export_gold_to_postgres(FakeDataFrame(), "sales_summary")
        assert captured["to_sql"]["name"] == "sales_summary"
        assert captured["to_sql"]["schema"] == "analytics"
        assert captured["to_sql"]["if_exists"] == "replace"
        assert captured["to_sql"]["index"] is False
        assert captured["disposed"] is True


class TestExecuteGold:
    def test_execute_gold_returns_paths(self, tmp_path, monkeypatch):
        input_path = _write_silver_parquet(
            tmp_path, [{"customer_id": 1, "amount": 25.0}]
        )
        sttm_path = _passthrough_sttm(
            tmp_path, "sales", ["customer_id", "amount"]
        )
        monkeypatch.setattr(gold_agent, "GOLD_DIR", tmp_path / "gold")
        monkeypatch.setattr(gold_agent, "_make_llm", lambda: object())
        monkeypatch.setattr(gold_agent, "AgentTrace", _FakeTrace)

        with patch("agents.gold_agent.AuditLogger") as audit_logger, patch(
            "langgraph.prebuilt.create_react_agent",
            side_effect=lambda llm, tools, prompt: _mock_agent_that_calls_tool(tools),
        ):
            output_paths = gold_agent.execute_gold(
                [input_path],
                sttm_path,
                "test-run-456",
                "Create a sales summary table.",
            )

        assert len(output_paths) == 1
        assert Path(output_paths[0]).is_file()
        assert len(pd.read_parquet(output_paths[0])) == 1
        audit_logger.assert_called_once_with("test-run-456")


class _FakeTrace:
    def set_input(self, **kwargs):
        return self

    def set_plan(self, *args):
        return self

    def extract_from_messages(self, *args):
        return self

    def set_output(self, **kwargs):
        return self

    def complete(self):
        return None

    def fail(self, error):
        raise error
