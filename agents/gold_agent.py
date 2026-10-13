import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from core.audit import AuditLogger
from core.config import (
	GEMINI_MODEL,
	GOLD_DIR,
	GOOGLE_API_KEY,
	GITHUB_BASE_URL,
	GITHUB_MODEL,
	GITHUB_TOKEN,
	DATABASE_URL,
	GOLD_DB_SCHEMA,
	GROQ_API_KEY,
	GROQ_MODEL,
	LLM_PROVIDER,
	OPENAI_API_KEY,
	OPENAI_MODEL,
)
from core.custom_transformations import (
	CUSTOM_TRANSFORMATION_TYPES,
	apply_custom_transformation,
)
from core.observability import AgentTrace
from core.user_workspace import current_username, user_directory

GOLD_AGENT_PROMPT = """
You are a careful Gold-layer analytics agent. You shape Silver data into
business-ready tables by joining sources and applying STTM aggregations.
You MUST call both tools in this order: inspect_task_tool, then
gold_aggregation_tool. Do not stop after inspecting or planning; planning without
executing is FAILURE.
Use the inspected Gold STTM grouped by target_table. Use matching *_id columns
for outer joins, apply the specified renames and aggregations, and return the
output paths produced by gold_aggregation_tool.
""".strip()

STTM_FIELDS = (
	"source_schema",
	"source_table",
	"source_column",
	"target_schema",
	"target_table",
	"target_column",
	"transformation_type",
	"transformation_logic",
)
FilePath = Union[str, Path]


def _normalize_sttm_rows(sttm_path: FilePath) -> List[Dict[str, str]]:
	aliases = {
		"sourcecol": "source_column",
		"sourcefield": "source_column",
		"targetcol": "target_column",
		"targetfield": "target_column",
		"transformation": "transformation_type",
		"logic": "transformation_logic",
	}
	with Path(sttm_path).open("r", encoding="utf-8", newline="") as sttm_file:
		reader = csv.DictReader(sttm_file)
		rows = []
		for raw_row in reader:
			rule = {}
			for raw_key, value in raw_row.items():
				key = re.sub(r"[^a-z0-9]+", "_", (raw_key or "").strip().lower()).strip("_")
				key = aliases.get(key.replace("_", ""), key)
				if key:
					rule[key] = "" if value is None else str(value).strip()
			for field in STTM_FIELDS:
				rule.setdefault(field, "")
			rows.append(rule)
	return rows


def _group_rules(
	rules: Sequence[Dict[str, str]]
) -> Dict[str, List[Dict[str, str]]]:
	grouped: Dict[str, List[Dict[str, str]]] = {}
	for rule in rules:
		target_table = rule.get("target_table", "").strip()
		if target_table:
			grouped.setdefault(target_table, []).append(rule)
	return grouped


def _inspect_task(
	input_files: Sequence[FilePath], sttm_path: FilePath
) -> Dict[str, Any]:
	import pyarrow.parquet as parquet

	files = []
	for file_path in input_files:
		path = Path(file_path)
		try:
			schema = parquet.read_schema(path)
			files.append(
				{
					"path": str(path),
					"source_name": _source_name(path),
					"columns": [
						{"name": field.name, "dtype": str(field.type)} for field in schema
					],
				}
			)
		except Exception as error:
			files.append({"path": str(path), "error": str(error)})

	try:
		rules = _normalize_sttm_rows(sttm_path)
		sttm_context: Dict[str, Any] = {
			"path": str(sttm_path),
			"by_target_table": _group_rules(rules),
		}
	except Exception as error:
		sttm_context = {"path": str(sttm_path), "error": str(error), "by_target_table": {}}
	return {"files": files, "sttm": sttm_context}


def _source_name(file_path: FilePath) -> str:
	name = Path(file_path).stem
	return name[: -len("_silver")] if name.endswith("_silver") else name


def _resolve_source_name(source_table: str, source_frames: Dict[str, Any]) -> str:
	if source_table in source_frames:
		return source_table
	if source_table.endswith("_silver") and source_table[: -len("_silver")] in source_frames:
		return source_table[: -len("_silver")]
	if f"{source_table}_silver" in source_frames:
		return f"{source_table}_silver"
	raise ValueError(f"No Silver input file matches source table '{source_table}'.")


def _join_sources(source_items: Sequence[tuple[str, Any]]) -> Any:
	import pandas as pd

	if not source_items:
		raise ValueError("A Gold target table must reference at least one source table.")
	merged = source_items[0][1].copy()
	for source_name, source_frame in source_items[1:]:
		join_columns = [
			column
			for column in merged.columns
			if column in source_frame.columns and str(column).lower().endswith("_id")
		]
		if not join_columns:
			raise ValueError(
				f"Cannot outer-join source '{source_name}': no shared *_id columns."
			)
		merged = pd.merge(
			merged,
			source_frame,
			how="outer",
			on=join_columns,
			suffixes=("", f"_{source_name}"),
			sort=False,
		)
	return merged


def _find_source_column(
	dataframe: Any, source_column: str, source_table: str
) -> str:
	if source_table:
		candidates = [f"{source_column}_{source_table}"]
		if source_table.endswith("_silver"):
			candidates.append(f"{source_column}_{source_table[:-7]}")
		candidates.append(source_column)
	else:
		candidates = [source_column]
	return next((column for column in candidates if column in dataframe.columns), "")


def _aggregation_method(rule: Dict[str, str]) -> str:
	if rule.get("transformation_type", "").strip().lower() in CUSTOM_TRANSFORMATION_TYPES:
		return ""
	text = " ".join(
		(rule.get(field, "") for field in ("transformation_type", "transformation_logic"))
	).lower()
	match = re.search(r"\b(sum|avg|average|mean|count|max|min)\b", text)
	if not match:
		return ""
	return {"avg": "mean", "average": "mean"}.get(match.group(1), match.group(1))


def _group_columns(
	dataframe: Any, rules: Sequence[Dict[str, str]], aggregate_columns: set[str]
) -> List[str]:
	group_columns = []
	for rule in rules:
		if rule.get("transformation_type", "").strip().lower() in CUSTOM_TRANSFORMATION_TYPES:
			continue
		text = " ".join(
			(rule.get(field, "") for field in ("transformation_type", "transformation_logic"))
		).lower()
		column = rule.get("target_column", "")
		if (
			column in dataframe.columns
			and column not in aggregate_columns
			and any(marker in text for marker in ("groupby", "group by", "dimension", "group_key"))
			and column not in group_columns
		):
			group_columns.append(column)

		match = re.search(r"group\s*by\s+([a-zA-Z0-9_, ]+)", text)
		if match:
			for name in (part.strip() for part in match.group(1).split(",")):
				if name in dataframe.columns and name not in aggregate_columns and name not in group_columns:
					group_columns.append(name)

	if not group_columns:
		group_columns = [column for column in dataframe.columns if column not in aggregate_columns]
	return group_columns


def _apply_aggregations(dataframe: Any, rules: Sequence[Dict[str, str]]) -> Any:
	import pandas as pd

	aggregations = []
	for rule in rules:
		method = _aggregation_method(rule)
		if not method:
			continue
		column = rule.get("target_column", "")
		if column not in dataframe.columns:
			column = rule.get("source_column", "")
		if column in dataframe.columns:
			aggregations.append((column, rule.get("target_column") or column, method))
	if not aggregations:
		return dataframe

	aggregate_columns = {source for source, _, _ in aggregations}
	group_columns = _group_columns(dataframe, rules, aggregate_columns)
	if group_columns:
		grouped = dataframe.groupby(group_columns, dropna=False)
		result = grouped.size().to_frame(name="__row_count__").reset_index().drop(
			columns=["__row_count__"]
		)
		for source_column, target_column, method in aggregations:
			aggregated = getattr(grouped[source_column], method)().reset_index(name=target_column)
			if target_column in result.columns:
				result = result.drop(columns=[target_column])
			result = result.merge(aggregated, on=group_columns, how="left", sort=False)
		return result

	result = {}
	for source_column, target_column, method in aggregations:
		result[target_column] = [getattr(dataframe[source_column], method)()]
	return pd.DataFrame(result)


def _export_gold_to_postgres(dataframe: Any, table_name: str) -> bool:
	if not DATABASE_URL:
		return False

	from sqlalchemy import create_engine

	engine = create_engine(DATABASE_URL, pool_pre_ping=True)
	try:
		dataframe.to_sql(
			name=table_name,
			con=engine,
			schema=GOLD_DB_SCHEMA,
			if_exists="replace",
			index=False,
			method="multi",
			chunksize=1000,
		)
	finally:
		engine.dispose()
	return True


def _apply_gold_rules(
	input_files: Sequence[FilePath], sttm_path: FilePath, run_id: str
) -> List[str]:
	import pandas as pd

	rules = _normalize_sttm_rows(sttm_path)
	rules_by_target = _group_rules(rules)
	if not rules_by_target:
		raise ValueError("Gold STTM must contain at least one target_table rule.")

	source_frames = {}
	for file_path in input_files:
		path = Path(file_path)
		name = _source_name(path)
		if name in source_frames:
			raise ValueError(f"Multiple Silver files resolve to the source name '{name}'.")
		source_frames[name] = pd.read_parquet(path)

	gold_dir = user_directory("gold_layer", GOLD_DIR)
	gold_dir.mkdir(parents=True, exist_ok=True)
	output_paths = []
	for target_table, target_rules in rules_by_target.items():
		source_names = []
		for rule in target_rules:
			source_table = rule.get("source_table", "").strip()
			if source_table:
				resolved = _resolve_source_name(source_table, source_frames)
				if resolved not in source_names:
					source_names.append(resolved)
		if not source_names:
			source_names = list(source_frames)
		source_items = [(name, source_frames[name]) for name in source_names]
		dataframe = _join_sources(source_items)

		rename_map = {}
		for rule in target_rules:
			source_column = rule.get("source_column", "")
			target_column = rule.get("target_column", "")
			if not source_column or not target_column:
				continue
			source_table = rule.get("source_table", "")
			column = _find_source_column(dataframe, source_column, source_table)
			if column and column != target_column:
				rename_map[column] = target_column
		if rename_map:
			dataframe = dataframe.rename(columns=rename_map)
		for rule in target_rules:
			if rule.get("transformation_type", "").strip().lower() in CUSTOM_TRANSFORMATION_TYPES:
				target_column = rule.get("target_column", "")
				if target_column:
					dataframe[target_column] = apply_custom_transformation(dataframe, rule)

		dataframe = _apply_aggregations(dataframe, target_rules)
		approved_columns = {
			rule["target_column"] for rule in target_rules if rule.get("target_column")
		}
		approved_columns.discard("pk_gold_id")
		columns_to_keep = [
			column for column in dataframe.columns if column in approved_columns
		]
		dataframe = dataframe.loc[:, columns_to_keep]
		dataframe.insert(0, "pk_gold_id", range(1, len(dataframe) + 1))

		safe_target = re.sub(r"[^A-Za-z0-9._-]", "_", target_table)
		output_path = gold_dir / f"{safe_target}.parquet"
		dataframe.to_parquet(output_path, index=False)
		database_table = safe_target.lower()
		username = current_username()
		if username:
			database_table = f"{username}_{database_table}"[:63]
		_export_gold_to_postgres(dataframe, database_table)
		output_paths.append(str(output_path))

	return output_paths


def _make_gold_tools(
	input_files: Sequence[FilePath], sttm_path: FilePath, run_id: str
) -> List[Any]:
	from langchain_core.tools import tool

	audit_logger = AuditLogger(run_id)
	paths = [str(path) for path in input_files]

	@tool("inspect_task_tool")
	def inspect_task_tool() -> str:
		"""Preview Silver schemas and Gold rules grouped by target table."""
		context = _inspect_task(input_files, sttm_path)
		audit_logger.log(
			"gold_agent",
			"inspect_task",
			input_files=paths,
			sttm_path=str(sttm_path),
		)
		return json.dumps(context, default=str, ensure_ascii=False)

	@tool("gold_aggregation_tool")
	def gold_aggregation_tool() -> str:
		"""Join Silver sources, apply Gold aggregations, and write Parquet tables."""
		output_paths = _apply_gold_rules(input_files, sttm_path, run_id)
		audit_logger.log(
			"gold_agent",
			"gold_aggregation",
			input_files=paths,
			output_paths=output_paths,
		)
		return json.dumps(output_paths)

	return [inspect_task_tool, gold_aggregation_tool]


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


def execute_gold(
	input_files: Sequence[FilePath],
	sttm_path: FilePath,
	run_id: str,
	task_description: str,
) -> List[str]:
	from langchain_core.messages import ToolMessage
	from langgraph.prebuilt import create_react_agent

	paths = [Path(file_path) for file_path in input_files]
	if not paths:
		raise ValueError("At least one Silver Parquet file is required.")

	trace = AgentTrace("gold_agent", run_id)
	trace.set_input(
		input_files=[str(path) for path in paths],
		sttm_path=str(sttm_path),
		task_description=task_description,
	)
	trace.set_plan("Inspect Silver schemas and Gold rules, then build analytics-ready tables.")

	try:
		agent = create_react_agent(
			_make_llm(),
			_make_gold_tools(paths, sttm_path, run_id),
			prompt=GOLD_AGENT_PROMPT,
		)
		response = agent.invoke(
			{
				"messages": [
					{
						"role": "user",
						"content": task_description
						+ "\nSilver files: "
						+ ", ".join(str(path) for path in paths)
						+ f"\nGold STTM: {sttm_path}",
					}
				]
			}
		)
		messages = response.get("messages", [])
		trace.extract_from_messages(messages)
		tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
		if len(tool_messages) < 2 or [message.name for message in tool_messages[:2]] != [
			"inspect_task_tool",
			"gold_aggregation_tool",
		]:
			raise RuntimeError(
				"The agent must call inspect_task_tool, then gold_aggregation_tool."
			)

		try:
			output_paths = json.loads(tool_messages[1].content)
		except (json.JSONDecodeError, TypeError) as error:
			raise RuntimeError("gold_aggregation_tool returned invalid output paths.") from error
		if not isinstance(output_paths, list) or not output_paths:
			raise RuntimeError("Gold aggregation did not return any output paths.")
		if any(not Path(path).is_file() for path in output_paths):
			raise RuntimeError("One or more Gold output files were not created.")

		trace.set_output(output_paths=output_paths)
		trace.complete()
		return [str(path) for path in output_paths]
	except Exception as error:
		trace.fail(error)
		raise