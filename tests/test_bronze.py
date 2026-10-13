"""Test Bronze ingestion with deterministic CSV/Parquet checks.

_apply_bronze_rules is a pure Python data helper, so its transformations can be
checked directly. execute_bronze is the AI entry point, so its agent is mocked
while the underlying ingestion tool still writes a real Parquet output.
"""

import csv
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import bronze_agent


def _make_sttm_csv(
	tmp_path: Path, extra_rows: list[dict] | None = None
) -> Path:
	columns = [
		"source_schema",
		"source_table",
		"source_column",
		"target_schema",
		"target_table",
		"target_column",
		"transformation_type",
		"transformation_logic",
	]
	rows = [
		{
			"source_schema": "landing",
			"source_table": "",
			"source_column": "source_id",
			"target_schema": "bronze",
			"target_table": "sales",
			"target_column": "record_id",
			"transformation_type": "rename",
			"transformation_logic": "Rename source_id to record_id.",
		},
		{
			"source_schema": "landing",
			"source_table": "",
			"source_column": "amount",
			"target_schema": "bronze",
			"target_table": "sales",
			"target_column": "amount",
			"transformation_type": "copy",
			"transformation_logic": "Copy without changes.",
		},
	]
	if extra_rows:
		rows.extend(extra_rows)

	sttm_path = tmp_path / "bronze_sttm.csv"
	with sttm_path.open("w", newline="", encoding="utf-8") as sttm_file:
		writer = csv.DictWriter(sttm_file, fieldnames=columns)
		writer.writeheader()
		writer.writerows(rows)
	return sttm_path


def _make_input_csv(tmp_path: Path, name: str = "sales.csv", row_count: int = 2) -> Path:
	csv_path = tmp_path / name
	with csv_path.open("w", newline="", encoding="utf-8") as input_file:
		writer = csv.DictWriter(
			input_file,
			fieldnames=["source_id", "amount"],
		)
		writer.writeheader()
		writer.writerows(
			{"source_id": index, "amount": index * 10}
			for index in range(1, row_count + 1)
		)
	return csv_path


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
				if hasattr(current_tool, "invoke"):
					result = current_tool.invoke({})
				else:
					result = current_tool()
				messages.append(
					ToolMessage(
						content=str(result),
						name=tool_name,
						tool_call_id=f"mock-tool-call-{tool_index}",
					)
				)
			return {"messages": messages}

	return MockAgent()


class TestApplyBronzeRules:
	def _run_ingestion(
		self,
		tmp_path: Path,
		monkeypatch: pytest.MonkeyPatch,
		input_files: list[Path],
	):
		monkeypatch.setattr(bronze_agent, "BRONZE_DIR", tmp_path / "bronze")
		sttm_path = _make_sttm_csv(tmp_path)
		return bronze_agent._apply_bronze_rules(input_files, sttm_path, "test-run")

	def test_metadata_columns_added(self, tmp_path, monkeypatch):
		input_path = _make_input_csv(tmp_path)
		output_path = self._run_ingestion(tmp_path, monkeypatch, [input_path])[0]
		dataframe = pd.read_parquet(output_path)

		assert "_load_timestamp" in dataframe.columns
		assert "_source_file" in dataframe.columns
		assert dataframe["_load_timestamp"].notna().all()
		assert dataframe["_source_file"].tolist() == [str(input_path)] * 2

	def test_row_count_preserved(self, tmp_path, monkeypatch):
		input_path = _make_input_csv(tmp_path, row_count=3)
		output_path = self._run_ingestion(tmp_path, monkeypatch, [input_path])[0]

		assert len(pd.read_parquet(output_path)) == 3

	def test_column_rename_applied(self, tmp_path, monkeypatch):
		input_path = _make_input_csv(tmp_path)
		output_path = self._run_ingestion(tmp_path, monkeypatch, [input_path])[0]
		dataframe = pd.read_parquet(output_path)

		assert "record_id" in dataframe.columns
		assert "source_id" not in dataframe.columns
		assert dataframe["record_id"].tolist() == [1, 2]

	def test_multiple_files_produce_multiple_outputs(self, tmp_path, monkeypatch):
		first_file = _make_input_csv(tmp_path, "sales.csv")
		second_file = _make_input_csv(tmp_path, "returns.csv")
		output_paths = self._run_ingestion(
			tmp_path,
			monkeypatch,
			[first_file, second_file],
		)

		assert len(output_paths) == 2
		assert {Path(path).name for path in output_paths} == {
			"sales_bronze.parquet",
			"returns_bronze.parquet",
		}
		assert all(Path(path).is_file() for path in output_paths)

	def test_output_file_is_valid_parquet(self, tmp_path, monkeypatch):
		input_path = _make_input_csv(tmp_path)
		output_path = self._run_ingestion(tmp_path, monkeypatch, [input_path])[0]

		dataframe = pd.read_parquet(output_path)
		assert list(dataframe["record_id"]) == [1, 2]
		assert list(dataframe["amount"]) == [10, 20]


class TestExecuteBronze:
	def test_execute_bronze_returns_paths(self, tmp_path, monkeypatch):
		input_path = _make_input_csv(tmp_path)
		sttm_path = _make_sttm_csv(tmp_path)
		monkeypatch.setattr(bronze_agent, "BRONZE_DIR", tmp_path / "bronze")
		monkeypatch.setattr(bronze_agent, "_make_llm", lambda: object())
		monkeypatch.setattr(bronze_agent, "AgentTrace", _FakeTrace)

		with patch("agents.bronze_agent.AuditLogger") as audit_logger, patch(
			"langgraph.prebuilt.create_react_agent",
			side_effect=lambda llm, tools, prompt: _mock_agent_that_calls_tool(tools),
		):
			output_paths = bronze_agent.execute_bronze(
				[input_path],
				sttm_path,
				"test-run-123",
				"Load the uploaded sales file into Bronze.",
			)

		assert len(output_paths) == 1
		assert Path(output_paths[0]).is_file()
		assert len(pd.read_parquet(output_paths[0])) == 2
		audit_logger.assert_called_once_with("test-run-123")


class _FakeTrace:
	def __init__(self, *args):
		self.output = None

	def set_input(self, **kwargs):
		return self

	def set_plan(self, *args):
		return self

	def extract_from_messages(self, *args):
		return self

	def set_output(self, **kwargs):
		self.output = kwargs
		return self

	def complete(self):
		return None

	def fail(self, error):
		raise error
