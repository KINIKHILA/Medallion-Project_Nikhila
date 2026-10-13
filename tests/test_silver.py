"""Tests for Silver cleansing and the Silver agent's execution boundary."""

import csv
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import silver_agent

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


def _make_silver_sttm(tmp_path: Path, rows: list[dict]) -> Path:
    sttm_path = tmp_path / "silver_sttm.csv"
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
    rows = [
        {
            "source_table": source_table,
            "source_column": column,
            "target_column": column,
            "transformation_type": "copy",
            "transformation_logic": "Copy without changes.",
        }
        for column in columns
    ]
    return _make_silver_sttm(tmp_path, rows)


def _mock_agent_that_calls_tool(tool):
    """Return an agent double that invokes the supplied tool or ordered tools."""
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


def _write_bronze_parquet(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    parquet_path = tmp_path / name
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return parquet_path


class TestApplySilverRules:
    def _run_silver(self, tmp_path, monkeypatch, input_files, sttm_path):
        monkeypatch.setattr(silver_agent, "SILVER_DIR", tmp_path / "silver")
        return silver_agent._apply_silver_rules(input_files, sttm_path, "test-run")

    def test_output_written_for_each_input(self, tmp_path, monkeypatch):
        first = _write_bronze_parquet(
            tmp_path, "sales_bronze.parquet", [{"id": 1, "value": "a"}]
        )
        second = _write_bronze_parquet(
            tmp_path, "returns_bronze.parquet", [{"id": 2, "value": "b"}]
        )
        sttm_path = _passthrough_sttm(tmp_path, "", ["id", "value"])

        outputs = self._run_silver(tmp_path, monkeypatch, [first, second], sttm_path)

        assert len(outputs) == 2
        assert {Path(path).name for path in outputs} == {
            "sales_silver.parquet",
            "returns_silver.parquet",
        }
        assert all(Path(path).is_file() for path in outputs)

    def test_null_handling_fill_mean(self, tmp_path, monkeypatch):
        input_path = _write_bronze_parquet(
            tmp_path,
            "sales_bronze.parquet",
            [{"amount": 10.0}, {"amount": None}, {"amount": 30.0}],
        )
        sttm_path = _make_silver_sttm(
            tmp_path,
            [
                {
                    "source_table": "sales_bronze",
                    "source_column": "amount",
                    "target_column": "amount",
                    "transformation_type": "fillna",
                    "transformation_logic": "fill null values with mean",
                }
            ],
        )

        output_path = self._run_silver(tmp_path, monkeypatch, [input_path], sttm_path)[0]
        dataframe = pd.read_parquet(output_path)

        assert dataframe["amount"].tolist() == pytest.approx([10.0, 20.0, 30.0])

    def test_dedup_removes_duplicate_rows(self, tmp_path, monkeypatch):
        input_path = _write_bronze_parquet(
            tmp_path,
            "sales_bronze.parquet",
            [{"id": 1, "value": "a"}, {"id": 1, "value": "a"}, {"id": 2, "value": "b"}],
        )
        sttm_path = _passthrough_sttm(tmp_path, "sales_bronze", ["id", "value"])

        output_path = self._run_silver(tmp_path, monkeypatch, [input_path], sttm_path)[0]
        dataframe = pd.read_parquet(output_path)

        assert len(dataframe) == 2
        assert dataframe["pk_sales_silver_id"].tolist() == [1, 2]

    def test_column_rename_applied(self, tmp_path, monkeypatch):
        input_path = _write_bronze_parquet(
            tmp_path, "sales_bronze.parquet", [{"customer_name": "Ada"}]
        )
        sttm_path = _make_silver_sttm(
            tmp_path,
            [
                {
                    "source_table": "sales_bronze",
                    "source_column": "customer_name",
                    "target_column": "customer",
                    "transformation_type": "rename",
                    "transformation_logic": "Rename customer_name to customer.",
                }
            ],
        )

        output_path = self._run_silver(tmp_path, monkeypatch, [input_path], sttm_path)[0]
        dataframe = pd.read_parquet(output_path)

        assert list(dataframe.columns)[0] == "pk_sales_silver_id"
        assert "customer" in dataframe.columns
        assert "customer_name" not in dataframe.columns
        assert dataframe["customer"].tolist() == ["Ada"]

    def test_output_is_valid_parquet(self, tmp_path, monkeypatch):
        input_path = _write_bronze_parquet(
            tmp_path, "sales_bronze.parquet", [{"id": 1, "value": "a"}]
        )
        sttm_path = _passthrough_sttm(tmp_path, "sales_bronze", ["id", "value"])

        output_path = self._run_silver(tmp_path, monkeypatch, [input_path], sttm_path)[0]
        dataframe = pd.read_parquet(output_path)

        assert Path(output_path).is_file()
        assert dataframe["id"].tolist() == [1]
        assert dataframe["value"].tolist() == ["a"]


class TestExecuteSilver:
    def test_execute_silver_returns_paths(self, tmp_path, monkeypatch):
        input_path = _write_bronze_parquet(
            tmp_path, "sales_bronze.parquet", [{"id": 1, "value": "a"}]
        )
        sttm_path = _passthrough_sttm(tmp_path, "sales_bronze", ["id", "value"])
        monkeypatch.setattr(silver_agent, "SILVER_DIR", tmp_path / "silver")
        monkeypatch.setattr(silver_agent, "_make_llm", lambda: object())
        monkeypatch.setattr(silver_agent, "AgentTrace", _FakeTrace)

        with patch("agents.silver_agent.AuditLogger") as audit_logger, patch(
            "langgraph.prebuilt.create_react_agent",
            side_effect=lambda llm, tools, prompt: _mock_agent_that_calls_tool(tools),
        ):
            output_paths = silver_agent.execute_silver(
                [input_path],
                sttm_path,
                "test-run-123",
                "Clean the sales records for analytics.",
            )

        assert len(output_paths) == 1
        assert Path(output_paths[0]).is_file()
        assert len(pd.read_parquet(output_paths[0])) == 1
        audit_logger.assert_called_once_with("test-run-123")


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
