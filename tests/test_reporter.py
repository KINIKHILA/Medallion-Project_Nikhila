"""Tests for reporter chart generation and DuckDB-backed tools."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import reporter

ANALYSIS_JSON = {
    "direct_answer": "Revenue increased during the period.",
    "charts": [
        {
            "type": "bar",
            "title": "Revenue by month",
            "x_column": "month",
            "y_column": "revenue",
        }
    ],
    "detailed_analysis": "The latest month had the highest revenue.",
}


def _mock_reporter_agent(tools, sql_query="SELECT month, revenue FROM gold_sales"):
    """Return an agent double that executes the reporter tools in order."""
    from langchain_core.messages import ToolMessage

    class MockReporterAgent:
        def invoke(self, request):
            tool_inputs = [
                {},
                {},
                {"sql_query": sql_query},
                {"analysis_json": json.dumps(ANALYSIS_JSON)},
            ]
            messages = []
            for index, (tool, tool_input) in enumerate(zip(tools, tool_inputs)):
                result = tool.invoke(tool_input)
                messages.append(
                    ToolMessage(
                        content=str(result),
                        name=tool.name,
                        tool_call_id=f"mock-reporter-call-{index}",
                    )
                )
            return {"messages": messages}

    return MockReporterAgent()


def _make_gold_parquet(tmp_path: Path) -> Path:
    parquet_path = tmp_path / "gold_sales.parquet"
    pd.DataFrame(
        {
            "month": ["2024-01", "2024-02", "2024-03"],
            "revenue": [100.0, 150.0, 180.0],
        }
    ).to_parquet(parquet_path, index=False)
    return parquet_path


class TestGenerateChartFromSpec:
    def test_bar_chart_returns_html(self):
        dataframe = pd.DataFrame({"month": ["Jan", "Feb"], "revenue": [10, 20]})
        chart_html = reporter.generate_chart_from_spec(
            dataframe,
            {"type": "bar", "title": "Revenue", "x_column": "month", "y_column": "revenue"},
            "bar-chart",
        )

        assert "<div" in chart_html
        assert "plotly" in chart_html.lower()

    def test_pie_chart_returns_html(self):
        dataframe = pd.DataFrame({"category": ["Books", "Games"], "sales": [3, 7]})
        chart_html = reporter.generate_chart_from_spec(
            dataframe,
            {"type": "pie", "title": "Sales mix", "x_column": "category", "y_column": "sales"},
            "pie-chart",
        )

        assert "<div" in chart_html
        assert "plotly" in chart_html.lower()

    def test_unknown_chart_type_returns_empty(self):
        dataframe = pd.DataFrame({"x": [1], "y": [2]})

        assert reporter.generate_chart_from_spec(
            dataframe,
            {"type": "heatmap", "title": "Unsupported", "x_column": "x", "y_column": "y"},
            "unknown-chart",
        ) == ""

    def test_bad_column_does_not_raise(self):
        dataframe = pd.DataFrame({"x": [1], "y": [2]})

        chart_html = reporter.generate_chart_from_spec(
            dataframe,
            {"type": "bar", "title": "Invalid columns", "x_column": "missing", "y_column": "y"},
            "bad-column-chart",
        )

        assert chart_html == ""


class TestMakeReporterTools:
    def test_load_tool_returns_catalog(self, tmp_path):
        gold_file = _make_gold_parquet(tmp_path)
        with patch("agents.reporter.AuditLogger"):
            tools, scratchpad, connection = reporter._make_reporter_tools(
                [gold_file], "catalog-run"
            )
        try:
            catalog = json.loads(tools[1].invoke({}))
            assert catalog[0]["table_name"] == "gold_sales"
            assert catalog[0]["columns"] == ["month", "revenue"]
            assert scratchpad["catalog"] == catalog
            registered_tables = connection.execute("SHOW TABLES").fetchall()
            assert ("gold_sales",) in registered_tables
        finally:
            connection.close()

    def test_execute_tool_stores_result_in_scratchpad(self, tmp_path):
        gold_file = _make_gold_parquet(tmp_path)
        with patch("agents.reporter.AuditLogger"):
            tools, scratchpad, connection = reporter._make_reporter_tools(
                [gold_file], "query-run"
            )
        try:
            agent = _mock_reporter_agent(tools)
            agent.invoke({"messages": []})
            assert len(scratchpad["query_results"]) == 1
            query_result = scratchpad["query_results"][0]
            assert query_result["columns"] == ["month", "revenue"]
            assert query_result["row_count"] == 3
            assert scratchpad["analysis"] == ANALYSIS_JSON
        finally:
            connection.close()

    def test_execute_tool_returns_error_on_bad_sql(self, tmp_path):
        gold_file = _make_gold_parquet(tmp_path)
        with patch("agents.reporter.AuditLogger"):
            tools, scratchpad, connection = reporter._make_reporter_tools(
                [gold_file], "bad-query-run"
            )
        try:
            tools[1].invoke({})
            result = json.loads(
                tools[2].invoke({"sql_query": "SELECT * FROM missing_gold_table"})
            )
            assert "error" in result
            assert "missing_gold_table" in result["error"]
            assert scratchpad["query_results"] == []
        finally:
            connection.close()
