import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Union

from core.audit import AuditLogger
from core.cache import (
	build_cache_key,
	configured_model_signature,
	fingerprint_file,
	get_cached_json,
	set_cached_json,
)
from core.config import PROFILES_DIR, STTM_DIR
from core.observability import AgentTrace
from core.performance import record_metric
from core.sttm_versions import record_sttm_version
from core.user_workspace import user_directory

STTM_AGENT_PROMPT = """
You generate Source-to-Target Mapping (STTM) rules for Bronze, Silver, or Gold layers.
First call inspect_context_tool to inspect the supplied context.
Then call exactly once the generation tool for the current layer. Do not call the
generation tools for other layers.
Every generated rule must have these fields: source_schema, source_table,
source_column, target_schema, target_table, target_column, transformation_type,
transformation_logic.
Use each source file's basename without its extension as source_table.
For Bronze, map every source column and include _load_timestamp and _source_file
metadata columns for each source table.
For Silver, put a surrogate-key rule first, followed by data-cleansing rules.
Use only columns approved by the Bronze STTM.
For Gold, put a surrogate-key rule first, followed by joins and aggregations that
support the supplied business intent.
""".strip()

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

FilePath = Union[str, Path]


def _read_profile(profile_path: FilePath) -> Dict[str, Any]:
	with Path(profile_path).open("r", encoding="utf-8") as profile_file:
		return json.load(profile_file)


def _read_sttm(sttm_path: FilePath) -> List[Dict[str, str]]:
	with Path(sttm_path).open("r", encoding="utf-8", newline="") as sttm_file:
		return list(csv.DictReader(sttm_file))


def _extract_sttm_rows(messages: Iterable[Any]) -> List[Dict[str, Any]]:
	decoder = json.JSONDecoder()
	for message in reversed(list(messages)):
		content = message if isinstance(message, str) else getattr(message, "content", "")
		if isinstance(content, list):
			content = "\n".join(
				str(block.get("text", "")) if isinstance(block, dict) else str(block)
				for block in content
			)
		if not isinstance(content, str):
			continue

		candidate_rows = []
		for position, character in enumerate(content):
			if character != "[":
				continue
			try:
				value, _ = decoder.raw_decode(content[position:])
			except json.JSONDecodeError:
				continue
			if isinstance(value, list) and all(isinstance(row, dict) for row in value):
				candidate_rows = value
		if candidate_rows or "[]" in content:
			return candidate_rows
	return []


def _read_parquet_schema(file_path: FilePath) -> Dict[str, Any]:
	import pyarrow.parquet as parquet

	schema = parquet.read_schema(file_path)
	return {
		"path": str(file_path),
		"columns": [
			{"name": field.name, "dtype": str(field.type)} for field in schema
		],
	}


def _profile_sources(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
	inspection_files = profile.get("inspection", {}).get("files", [])
	statistics_files = profile.get("statistics", {}).get("files", [])
	statistics_by_path = {
		item.get("path"): item for item in statistics_files if item.get("path")
	}
	sources = []
	for item in inspection_files or profile.get("files", []):
		path = item.get("path", "")
		columns = item.get("columns", [])
		if isinstance(columns, dict):
			columns = list(columns)
		if not columns:
			stats = statistics_by_path.get(path, {})
			stat_columns = stats.get("columns", {})
			columns = list(stat_columns) if isinstance(stat_columns, dict) else []
		sources.append(
			{
				"path": path,
				"table": Path(path).stem if path else "source_table",
				"columns": [str(column) for column in columns],
			}
		)
	return sources


def _prepare_bronze_context(profile_path: FilePath) -> Dict[str, Any]:
	profile = _read_profile(profile_path)
	return {
		"layer": "bronze",
		"profile_path": str(profile_path),
		"source_files": _profile_sources(profile),
		"profile": profile,
	}


def _prepare_silver_context(
	bronze_paths: Sequence[FilePath], bronze_sttm_path: FilePath
) -> Dict[str, Any]:
	bronze_rules = _read_sttm(bronze_sttm_path)
	approved_columns = {
		row["target_column"]
		for row in bronze_rules
		if row.get("target_column")
	}
	parquet_files = []
	for file_path in bronze_paths:
		file_schema = _read_parquet_schema(file_path)
		file_schema["file_name"] = Path(file_path).name
		all_columns = file_schema["columns"]
		file_schema["columns"] = [
			column for column in all_columns if column["name"] in approved_columns
		]
		file_schema["excluded_columns"] = [
			column["name"] for column in all_columns if column["name"] not in approved_columns
		]
		parquet_files.append(file_schema)
	return {
		"layer": "silver",
		"bronze_files": parquet_files,
		"approved_columns": sorted(approved_columns),
		"bronze_sttm": bronze_rules,
	}


def _prepare_gold_context(
	silver_paths: Sequence[FilePath], silver_sttm_path: FilePath
) -> Dict[str, Any]:
	silver_rules = _read_sttm(silver_sttm_path)
	approved_columns = {
		row.get("target_column", "")
		for row in silver_rules
		if row.get("target_column")
	}
	silver_files = []
	for file_path in silver_paths:
		file_schema = _read_parquet_schema(file_path)
		file_schema["file_name"] = Path(file_path).name
		all_columns = file_schema["columns"]
		file_schema["columns"] = [
			column for column in all_columns if column["name"] in approved_columns
		]
		file_schema["excluded_columns"] = [
			column["name"] for column in all_columns if column["name"] not in approved_columns
		]
		silver_files.append(file_schema)
	return {
		"layer": "gold",
		"silver_files": silver_files,
		"approved_columns": sorted(approved_columns),
		"silver_sttm": silver_rules,
	}


def _normalize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
	if not rows:
		raise ValueError("The STTM generation tool requires at least one rule row.")
	if any(not isinstance(row, dict) for row in rows):
		raise ValueError("Every STTM rule must be an object with named fields.")
	return [
		{
			column: str(row.get(column, "") if row.get(column) is not None else "")
			for column in STTM_COLUMNS
		}
		for row in rows
	]


def _ensure_bronze_rules(
	rows: List[Dict[str, str]], context: Dict[str, Any]
) -> List[Dict[str, str]]:
	for source in context.get("source_files", []):
		table = source.get("table", "source_table")
		for column in source.get("columns", []):
			if not any(
				row["source_table"] == table and row["source_column"] == column
				for row in rows
			):
				rows.append(
					{
						"source_schema": "landing",
						"source_table": table,
						"source_column": column,
						"target_schema": "bronze",
						"target_table": table,
						"target_column": column,
						"transformation_type": "copy",
						"transformation_logic": "Copy source column without changes.",
					}
				)
		for target_column, logic in (
			("_load_timestamp", "UTC timestamp when the row is loaded."),
			("_source_file", "Name of the source file for this row."),
		):
			if not any(
				row["source_table"] == table
				and row["target_column"] == target_column
				for row in rows
			):
				rows.append(
					{
						"source_schema": "system",
						"source_table": "runtime_metadata",
						"source_column": "",
						"target_schema": "bronze",
						"target_table": table,
						"target_column": target_column,
						"transformation_type": "metadata",
						"transformation_logic": logic,
					}
				)
	return rows


def _ensure_surrogate_key(
	rows: List[Dict[str, str]], layer: str
) -> List[Dict[str, str]]:
	target_column = f"{layer}_sk"
	key_rule = next(
		(
			row
			for row in rows
			if row["transformation_type"].lower() == "surrogate_key"
			or row["target_column"] == target_column
		),
		None,
	)
	if key_rule is None:
		key_rule = {
			"source_schema": "",
			"source_table": "",
			"source_column": "",
			"target_schema": layer,
			"target_table": f"{layer}_dataset",
			"target_column": target_column,
			"transformation_type": "surrogate_key",
			"transformation_logic": "Generate a surrogate key for each output row.",
		}
	return [key_rule] + [row for row in rows if row is not key_rule]


def _write_sttm(
	layer: str,
	rows: Sequence[Dict[str, Any]],
	context: Dict[str, Any],
	run_id: str,
	audit_logger: AuditLogger,
) -> str:
	normalized_rows = _normalize_rows(rows)
	if layer == "bronze":
		normalized_rows = _ensure_bronze_rules(normalized_rows, context)
	elif layer in {"silver", "gold"}:
		normalized_rows = _ensure_surrogate_key(normalized_rows, layer)

	sttm_dir = user_directory("sttm", STTM_DIR)
	sttm_dir.mkdir(parents=True, exist_ok=True)
	safe_run_id = re.sub(r"[^A-Za-z0-9._-]", "_", run_id)
	output_path = sttm_dir / f"sttm_{layer}_{safe_run_id}.csv"
	with output_path.open("w", encoding="utf-8", newline="") as sttm_file:
		writer = csv.DictWriter(sttm_file, fieldnames=STTM_COLUMNS)
		writer.writeheader()
		writer.writerows(normalized_rows)
	record_sttm_version(
			layer,
			normalized_rows,
			run_id,
			"generated",
		audit_logger,
	)
	audit_logger.log(
		"sttm_generator_agent",
		f"generate_{layer}_sttm",
		path=str(output_path),
		row_count=len(normalized_rows),
	)
	return str(output_path)


def _make_sttm_tools(
	layer: str,
	context: Dict[str, Any],
	run_id: str,
	business_intent: str = "",
) -> List[Any]:
	from langchain_core.tools import tool

	if layer not in {"bronze", "silver", "gold"}:
		raise ValueError("Layer must be bronze, silver, or gold.")
	tool_context = {**context, "business_intent": business_intent}
	audit_logger = AuditLogger(run_id)

	@tool("inspect_context_tool")
	def inspect_context_tool() -> str:
		"""Inspect the current layer's source files and approved mapping context."""
		audit_logger.log("sttm_generator_agent", "inspect_context", layer=layer)
		return json.dumps(tool_context, default=str, ensure_ascii=False)

	@tool("generate_bronze_sttm_tool")
	def generate_bronze_sttm_tool(rows: List[Dict[str, Any]]) -> str:
		"""Save Bronze rules mapping all source columns and load metadata."""
		return _write_sttm("bronze", rows, tool_context, run_id, audit_logger)

	@tool("generate_silver_sttm_tool")
	def generate_silver_sttm_tool(rows: List[Dict[str, Any]]) -> str:
		"""Save Silver rules with a surrogate key followed by cleansing rules."""
		return _write_sttm("silver", rows, tool_context, run_id, audit_logger)

	@tool("generate_gold_sttm_tool")
	def generate_gold_sttm_tool(rows: List[Dict[str, Any]]) -> str:
		"""Save Gold rules with a surrogate key, joins, and business aggregations."""
		return _write_sttm("gold", rows, tool_context, run_id, audit_logger)

	return [
		inspect_context_tool,
		generate_bronze_sttm_tool,
		generate_silver_sttm_tool,
		generate_gold_sttm_tool,
	]


def _stable_cache_context(value: Any) -> Any:
	if isinstance(value, dict):
		return {
			key: _stable_cache_context(item)
			for key, item in value.items()
			if key not in {"run_id", "created_at", "profile_path"}
		}
	if isinstance(value, list):
		return [_stable_cache_context(item) for item in value]
	return value


def _run_sttm_agent(
	layer: str,
	context: Dict[str, Any],
	run_id: str,
	task_description: str,
	business_intent: str = "",
) -> str:
	from langchain_core.messages import ToolMessage
	from langgraph.prebuilt import create_react_agent

	generation_tool = f"generate_{layer}_sttm_tool"
	trace = AgentTrace("sttm_generator", run_id)
	trace.set_input(
		layer=layer,
		task_description=task_description,
		business_intent=business_intent,
	)
	trace.set_plan(
		f"Inspect {layer} context, generate the layer's STTM rules once, and verify the saved CSV."
	)

	try:
		agent_prompt = (
			f"{STTM_AGENT_PROMPT}\n\nCurrent layer: {layer}. "
			f"Use {generation_tool} exactly once after inspect_context_tool."
		)
		if business_intent:
			agent_prompt += f"\nBusiness intent: {business_intent}"
		cache_key = build_cache_key(
			f"sttm_{layer}",
			{
				"layer": layer,
				"context": _stable_cache_context(context),
				"task_description": task_description,
				"business_intent": business_intent,
				"prompt": agent_prompt,
				"model": configured_model_signature(),
			},
		)
		cached_rows = get_cached_json(f"sttm_{layer}", cache_key)
		record_metric(
			run_id,
			"sttm_generator",
			"cache_lookup",
			1,
			unit="lookup",
			details={
				"namespace": f"sttm_{layer}",
				"status": "hit" if cached_rows is not None else "miss",
			},
		)
		if cached_rows is not None:
			logger = AuditLogger(run_id)
			sttm_path = _write_sttm(layer, cached_rows, context, run_id, logger)
			trace.set_output(sttm_path=sttm_path, cache_hit=True)
			trace.complete()
			return sttm_path

		agent = create_react_agent(
			_make_llm(),
			_make_sttm_tools(layer, context, run_id, business_intent),
			prompt=agent_prompt,
		)
		response = agent.invoke(
			{
				"messages": [
					{
						"role": "user",
						"content": task_description
						+ "\nGenerate source-to-target mapping rules for the "
						+ layer
						+ " layer.",
					}
				]
			}
		)
		messages = response.get("messages", [])
		trace.extract_from_messages(messages)
		tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
		generation_tools = {
			"generate_bronze_sttm_tool",
			"generate_silver_sttm_tool",
			"generate_gold_sttm_tool",
		}
		generation_results = [
			message for message in tool_messages if message.name in generation_tools
		]
		if not tool_messages or tool_messages[0].name != "inspect_context_tool":
			raise RuntimeError("The agent must inspect context before generating STTM rules.")
		if len(generation_results) != 1 or generation_results[0].name != generation_tool:
			raise RuntimeError(
				f"Expected exactly one call to {generation_tool}; got "
				f"{[message.name for message in generation_results]}."
			)

		sttm_path = generation_results[0].content
		if not isinstance(sttm_path, str) or not Path(sttm_path).is_file():
			raise RuntimeError(f"{generation_tool} did not return a valid STTM file path.")
		set_cached_json(f"sttm_{layer}", cache_key, _read_sttm(sttm_path))
		trace.set_output(sttm_path=sttm_path, cache_hit=False)
		trace.complete()
		return sttm_path
	except Exception as error:
		trace.fail(error)
		raise


def _make_llm() -> Any:
	from core.config import (
		GEMINI_MODEL,
		GOOGLE_API_KEY,
		GITHUB_BASE_URL,
		GITHUB_MODEL,
		GITHUB_TOKEN,
		GROQ_API_KEY,
		GROQ_MODEL,
		LLM_PROVIDER,
		OPENAI_API_KEY,
		OPENAI_MODEL,
	)

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


def generate_bronze_sttm(
	profile_path: FilePath, run_id: str, task_description: str
) -> str:
	context = _prepare_bronze_context(profile_path)
	return _run_sttm_agent("bronze", context, run_id, task_description)


def generate_silver_sttm(
	bronze_output_paths: Sequence[FilePath],
	bronze_sttm_path: FilePath,
	run_id: str,
	task_description: str,
) -> str:
	context = _prepare_silver_context(bronze_output_paths, bronze_sttm_path)
	return _run_sttm_agent("silver", context, run_id, task_description)


def generate_gold_sttm(
	silver_output_paths: Sequence[FilePath],
	silver_sttm_path: FilePath,
	business_intent: str,
	run_id: str,
	task_description: str,
) -> str:
	context = _prepare_gold_context(silver_output_paths, silver_sttm_path)
	return _run_sttm_agent(
		"gold", context, run_id, task_description, business_intent
	)