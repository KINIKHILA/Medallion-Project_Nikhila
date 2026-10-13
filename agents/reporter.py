import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from core.audit import AuditLogger
from core.config import (
	GEMINI_MODEL,
	GOOGLE_API_KEY,
	GITHUB_BASE_URL,
	GITHUB_MODEL,
	GITHUB_TOKEN,
	LLM_PROVIDER,
	GOLD_DIR,
	GROQ_API_KEY,
	GROQ_MODEL,
	OPENAI_API_KEY,
	OPENAI_MODEL,
	REPORTS_DIR,
)
from core.observability import AgentTrace
from core.user_workspace import user_directory

REPORTER_AGENT_PROMPT = """
You are an analytics reporter who answers the supplied business question using
only the available Gold tables.
Call tools in this exact order:
1. inspect_gold_tables_tool
2. load_gold_data_tool
3. execute_query_tool (you may run more than one SQL query if needed)
4. submit_analysis_tool
Do not skip tools, stop after planning, or submit analysis before querying data.
After reviewing query results, call submit_analysis_tool with a JSON string that
has direct_answer, charts, and detailed_analysis keys. The charts value must be a
list of objects with type, title, x_column, and y_column fields. Supported chart
types are bar, line, pie, and scatter. Use only columns returned by your SQL.
""".strip()

FilePath = Union[str, Path]


def _json_default(value: Any) -> Any:
	if hasattr(value, "item"):
		return value.item()
	if hasattr(value, "isoformat"):
		return value.isoformat()
	return str(value)


def _table_name(file_path: FilePath) -> str:
	name = re.sub(r"\W+", "_", Path(file_path).stem).strip("_").lower()
	if not name or name[0].isdigit():
		name = f"gold_{name}"
	return name


def _inspect_gold_tables(gold_files: Sequence[FilePath]) -> Dict[str, Any]:
	import pyarrow.parquet as parquet

	tables = []
	for file_path in gold_files:
		path = Path(file_path)
		try:
			parquet_file = parquet.ParquetFile(path)
			schema = parquet_file.schema_arrow
			sample_rows = []
			if parquet_file.num_row_groups:
				sample_rows = parquet_file.read_row_group(0).slice(0, 5).to_pylist()
			tables.append(
				{
					"path": str(path),
					"table_name": _table_name(path),
					"row_count": parquet_file.metadata.num_rows,
					"columns": [
						{"name": field.name, "dtype": str(field.type)} for field in schema
					],
					"sample_rows": sample_rows,
				}
			)
		except Exception as error:
			tables.append({"path": str(path), "error": str(error)})
	return {"tables": tables}


def generate_chart_from_spec(df: Any, chart_spec: Dict[str, Any], chart_id: str) -> str:
	import plotly.express as px

	chart_type = str(chart_spec.get("type", "")).strip().lower()
	title = str(chart_spec.get("title", ""))
	x_column = chart_spec.get("x_column")
	y_column = chart_spec.get("y_column")
	if chart_type not in {"bar", "line", "pie", "scatter"}:
		return ""
	if x_column not in df.columns or y_column not in df.columns:
		return ""

	chart_functions = {
		"bar": px.bar,
		"line": px.line,
		"pie": px.pie,
		"scatter": px.scatter,
	}
	try:
		if chart_type == "pie":
			figure = chart_functions[chart_type](
				df, names=x_column, values=y_column, title=title
			)
		else:
			figure = chart_functions[chart_type](
				df, x=x_column, y=y_column, title=title
			)
		figure.update_layout(
			template="plotly_white", margin={"l": 40, "r": 24, "t": 64, "b": 40}
		)
		return figure.to_html(
			full_html=False,
			include_plotlyjs=False,
			div_id=re.sub(r"[^A-Za-z0-9_-]", "_", chart_id),
		)
	except Exception:
		return ""


def _make_reporter_tools(
	gold_files: Sequence[FilePath], run_id: str
) -> tuple[List[Any], Dict[str, Any], Any]:
	import duckdb
	from langchain_core.tools import tool

	connection = duckdb.connect(database=":memory:")
	scratchpad: Dict[str, Any] = {
		"catalog": [],
		"query_results": [],
		"analysis": None,
	}
	audit_logger = AuditLogger(run_id)
	paths = [str(Path(file_path)) for file_path in gold_files]

	@tool("inspect_gold_tables_tool")
	def inspect_gold_tables_tool() -> str:
		"""Preview Gold table schemas and sample rows without querying DuckDB."""
		result = _inspect_gold_tables(gold_files)
		audit_logger.log("reporter_agent", "inspect_gold_tables", gold_files=paths)
		return json.dumps(result, default=_json_default, ensure_ascii=False)

	@tool("load_gold_data_tool")
	def load_gold_data_tool() -> str:
		"""Load all supplied Gold Parquet files into DuckDB and return the catalog."""
		catalog = []
		seen_names = set()
		for file_path in gold_files:
			path = Path(file_path)
			table_name = _table_name(path)
			if table_name in seen_names:
				raise ValueError(f"Gold files resolve to duplicate table name '{table_name}'.")
			seen_names.add(table_name)
			relation = connection.read_parquet(str(path))
			relation.create_view(table_name, replace=True)
			catalog.append(
				{
					"table_name": table_name,
					"path": str(path),
					"columns": list(relation.columns),
				}
			)
		scratchpad["catalog"] = catalog
		audit_logger.log("reporter_agent", "load_gold_data", catalog=catalog)
		return json.dumps(catalog, ensure_ascii=False)

	@tool("execute_query_tool")
	def execute_query_tool(sql_query: str) -> str:
		"""Run one read-only SELECT query against the loaded Gold tables."""
		query = str(sql_query or "").strip().rstrip(";").strip()
		try:
			if not query:
				raise ValueError("SQL query cannot be empty.")
			if ";" in query or re.match(r"^(select|with)\b", query, flags=re.IGNORECASE) is None:
				raise ValueError("Only one read-only SELECT or WITH query is allowed.")
			if re.search(
				r"\b(insert|update|delete|drop|create|alter|copy|attach|detach|pragma|call)\b",
				query,
				flags=re.IGNORECASE,
			):
				raise ValueError("Data-changing SQL statements are not allowed.")
			if not scratchpad["catalog"]:
				raise RuntimeError("Call load_gold_data_tool before querying Gold tables.")

			dataframe = connection.execute(query).fetchdf()
			records = json.loads(dataframe.to_json(orient="records", date_format="iso"))
			query_result = {
				"sql": query,
				"columns": [str(column) for column in dataframe.columns],
				"row_count": len(dataframe),
				"rows": records,
				"dataframe": dataframe,
			}
			scratchpad["query_results"].append(query_result)
			audit_logger.log(
				"reporter_agent",
				"execute_query",
				sql=query,
				row_count=len(dataframe),
			)
			return json.dumps(
				{
					"sql": query,
					"columns": query_result["columns"],
					"row_count": query_result["row_count"],
					"rows": records[:100],
					"truncated": len(records) > 100,
				},
				ensure_ascii=False,
			)
		except Exception as error:
			audit_logger.log("reporter_agent", "query_error", sql=query, error=str(error))
			return json.dumps({"sql": query, "error": str(error)}, ensure_ascii=False)

	@tool("submit_analysis_tool")
	def submit_analysis_tool(analysis_json: str) -> str:
		"""Submit final JSON analysis with direct_answer, charts, and detailed_analysis."""
		try:
			analysis = json.loads(analysis_json)
		except json.JSONDecodeError as error:
			raise ValueError("Analysis must be valid JSON.") from error
		if not isinstance(analysis, dict):
			raise ValueError("Analysis JSON must be an object.")
		for field in ("direct_answer", "charts", "detailed_analysis"):
			if field not in analysis:
				raise ValueError(f"Analysis JSON is missing '{field}'.")
		if not isinstance(analysis["charts"], list):
			raise ValueError("Analysis 'charts' must be a list.")
		scratchpad["analysis"] = analysis
		audit_logger.log("reporter_agent", "submit_analysis")
		return "Analysis accepted."

	return [
		inspect_gold_tables_tool,
		load_gold_data_tool,
		execute_query_tool,
		submit_analysis_tool,
	], scratchpad, connection


def _make_llm() -> Any:
	if LLM_PROVIDER == "groq":
		if not GROQ_API_KEY:
			raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER is 'groq'.")
		from langchain_groq import ChatGroq

		return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)

	if LLM_PROVIDER == "openai":
		if not OPENAI_API_KEY:
			raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'.")
		from langchain_openai import ChatOpenAI

		return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)

	if LLM_PROVIDER == "github":
		if not GITHUB_TOKEN:
			raise ValueError("GITHUB_TOKEN is required when LLM_PROVIDER is 'github'.")
		from langchain_openai import ChatOpenAI

		return ChatOpenAI(
			model=GITHUB_MODEL,
			api_key=GITHUB_TOKEN,
			base_url=GITHUB_BASE_URL,
			temperature=0,
		)

	if LLM_PROVIDER == "gemini":
		if not GOOGLE_API_KEY:
			raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER is 'gemini'.")
		from langchain_google_genai import ChatGoogleGenerativeAI

		return ChatGoogleGenerativeAI(
			model=GEMINI_MODEL,
			google_api_key=GOOGLE_API_KEY,
			temperature=0,
		)

	raise ValueError(
		"No supported LLM provider is configured. Set LLM_PROVIDER and its API key."
	)


def _analysis_html(value: Any) -> str:
	if isinstance(value, list):
		return "<ul>" + "".join(f"<li>{html.escape(str(item))}</li>" for item in value) + "</ul>"
	if isinstance(value, dict):
		return f"<pre>{html.escape(json.dumps(value, indent=2, default=_json_default))}</pre>"
	return f"<p>{html.escape(str(value))}</p>"


def _build_report_html(
	business_intent: str,
	analysis: Dict[str, Any],
	queries: Sequence[Dict[str, Any]],
	chart_html: Sequence[str],
) -> str:
	sql_text = "\n\n".join(query["sql"] for query in queries) or "No SQL queries were recorded."
	charts_section = "".join(chart_html) or "<p>No charts were requested.</p>"
	return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Business Analytics Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #16212b; background: #f3f7f8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at top right, #d8f0ed 0, transparent 34rem), linear-gradient(135deg, #f5f8f7, #eef2f6); }}
    header {{ padding: 3.5rem max(1.25rem, calc((100vw - 1040px) / 2)); color: #fff; background: linear-gradient(115deg, #123c48, #167c80 58%, #41a98f); }}
    header .eyebrow {{ color: #b7ece0; font-size: .78rem; font-weight: 700; text-transform: uppercase; }}
    h1 {{ max-width: 900px; margin: .6rem 0 0; font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.08; }}
    main {{ width: min(1040px, calc(100% - 2rem)); margin: 2rem auto 4rem; }}
    section {{ margin: 1rem 0; padding: 1.5rem; border: 1px solid #dce5e7; border-radius: 8px; background: rgba(255,255,255,.9); box-shadow: 0 8px 28px rgba(25,55,65,.06); }}
    h2 {{ margin: 0 0 1rem; color: #164b55; font-size: 1.2rem; }}
    .answer {{ font-size: 1.12rem; line-height: 1.65; }}
    pre {{ overflow-x: auto; margin: 0; padding: 1rem; border-radius: 6px; color: #e1f5f2; background: #152b36; font: .88rem/1.6 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; }}
    .chart {{ margin: 1.5rem 0; }}
    footer {{ padding: 1.25rem; color: #d7e6e8; text-align: center; background: #183842; }}
    @media (max-width: 600px) {{ header {{ padding: 2.5rem 1rem; }} section {{ padding: 1rem; }} }}
  </style>
</head>
<body>
  <header><div class="eyebrow">Gold-layer analytics</div><h1>{html.escape(business_intent)}</h1></header>
  <main>
    <section><h2>Answer</h2><div class="answer">{_analysis_html(analysis.get("direct_answer", ""))}</div></section>
    <section><h2>Approach</h2><pre><code>{html.escape(sql_text)}</code></pre></section>
    <section><h2>Detailed analysis</h2>{_analysis_html(analysis.get("detailed_analysis", ""))}</section>
    <section><h2>Charts</h2>{charts_section}</section>
  </main>
  <footer>Generated by IDAMP</footer>
</body>
</html>"""


def generate_report(
	gold_files: Sequence[FilePath],
	business_intent: str,
	run_id: str,
	task_description: str,
) -> str:
	from langchain_core.messages import ToolMessage
	from langgraph.prebuilt import create_react_agent

	paths = [Path(file_path) for file_path in gold_files]
	if not paths:
		raise ValueError("At least one Gold Parquet file is required.")

	trace = AgentTrace("reporter", run_id)
	trace.set_input(
		gold_files=[str(path) for path in paths],
		business_intent=business_intent,
		task_description=task_description,
	)
	trace.set_plan("Inspect Gold tables, load them into DuckDB, query, and submit analysis.")
	connection = None
	try:
		tools, scratchpad, connection = _make_reporter_tools(paths, run_id)
		agent = create_react_agent(_make_llm(), tools, prompt=REPORTER_AGENT_PROMPT)
		response = agent.invoke(
			{
				"messages": [
					{
						"role": "user",
						"content": (
							task_description
							+ "\nBusiness question: "
							+ business_intent
							+ "\nGold files: "
							+ ", ".join(str(path) for path in paths)
						),
					}
				]
			}
		)
		messages = response.get("messages", [])
		trace.extract_from_messages(messages)
		tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
		tool_names = [message.name for message in tool_messages]
		expected_order = [
			"inspect_gold_tables_tool",
			"load_gold_data_tool",
			"execute_query_tool",
			"submit_analysis_tool",
		]
		positions = []
		last_position = -1
		for name in expected_order:
			position = next(
				(index for index in range(last_position + 1, len(tool_names)) if tool_names[index] == name),
				-1,
			)
			if position < 0:
				raise RuntimeError(f"The agent did not call required tool '{name}'.")
			positions.append(position)
			last_position = position
		if positions != sorted(positions) or positions[-1] != len(tool_names) - 1:
			raise RuntimeError("Reporter tools must run in order, with submit_analysis_tool last.")

		analysis = scratchpad.get("analysis")
		query_results = scratchpad.get("query_results", [])
		if not analysis or not query_results:
			raise RuntimeError("The reporter must query Gold data and submit its analysis.")

		chart_html = []
		chart_errors = []
		for chart_index, chart_spec in enumerate(analysis.get("charts", []), start=1):
			if not isinstance(chart_spec, dict):
				chart_errors.append("Skipped a chart specification that was not an object.")
				continue
			chart_frame = next(
				(
					query["dataframe"]
					for query in reversed(query_results)
					if chart_spec.get("x_column") in query["dataframe"].columns
					and chart_spec.get("y_column") in query["dataframe"].columns
				),
				None,
			)
			if chart_frame is None:
				chart_errors.append(
					f"No query result contains the columns for chart '{chart_spec.get('title', chart_index)}'."
				)
				continue
			try:
				chart_markup = generate_chart_from_spec(
					chart_frame, chart_spec, f"chart-{chart_index}"
				)
				if chart_markup:
					chart_html.append(f'<div class="chart">{chart_markup}</div>')
			except Exception as error:
				chart_errors.append(str(error))

		reports_dir = user_directory("reports", REPORTS_DIR)
		reports_dir.mkdir(parents=True, exist_ok=True)
		safe_run_id = re.sub(r"[^A-Za-z0-9._-]", "_", run_id)
		timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
		base_path = reports_dir / f"report_{safe_run_id}_{timestamp}"
		html_path = base_path.with_suffix(".html")
		json_path = base_path.with_suffix(".json")
		html_report = _build_report_html(
			business_intent,
			analysis,
			query_results,
			chart_html,
		)
		report_data = {
			"run_id": run_id,
			"generated_at": datetime.now(timezone.utc).isoformat(),
			"business_intent": business_intent,
			"analysis": analysis,
			"catalog": scratchpad["catalog"],
			"queries": [
				{
					"sql": result["sql"],
					"columns": result["columns"],
					"row_count": result["row_count"],
					"rows": result["rows"][:1000],
					"truncated": result["row_count"] > 1000,
				}
				for result in query_results
			],
			"chart_errors": chart_errors,
			"html_path": str(html_path),
		}
		html_path.write_text(html_report, encoding="utf-8")
		json_path.write_text(
			json.dumps(report_data, indent=2, default=_json_default, ensure_ascii=False)
			+ "\n",
			encoding="utf-8",
		)
		trace.set_output(html_path=str(html_path), json_path=str(json_path))
		trace.complete()
		return str(html_path)
	except Exception as error:
		trace.fail(error)
		raise
	finally:
		if connection is not None:
			connection.close()