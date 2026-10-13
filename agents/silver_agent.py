import csv
import json
import re
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
	GROQ_API_KEY,
	GROQ_MODEL,
	OPENAI_API_KEY,
	OPENAI_MODEL,
	SILVER_DIR,
)
from core.custom_transformations import (
	CUSTOM_TRANSFORMATION_TYPES,
	apply_custom_transformation,
)
from core.observability import AgentTrace
from core.user_workspace import user_directory

SILVER_AGENT_PROMPT = """
You are a careful Silver-layer data cleansing agent.
You MUST call both tools in this order: inspect_task_tool, then silver_ingestion_tool.
Do not stop after inspecting or planning; planning without executing is FAILURE.
Use the STTM rules to rename columns, handle nulls, standardize types and text,
remove duplicates, and keep only approved target columns. Verify that the
silver_ingestion_tool completed and report its output paths.
""".strip()

STTM_FIELDS = {
	"source_schema",
	"source_table",
	"source_column",
	"target_schema",
	"target_table",
	"target_column",
	"transformation_type",
	"transformation_logic",
}
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
			normalized_row = {}
			for raw_key, value in raw_row.items():
				key = re.sub(r"[^a-z0-9]+", "_", (raw_key or "").strip().lower()).strip("_")
				key = aliases.get(key.replace("_", ""), key)
				if key:
					normalized_row[key] = "" if value is None else str(value).strip()
			for field in STTM_FIELDS:
				normalized_row.setdefault(field, "")
			rows.append(normalized_row)
	return rows


def _inspect_task(
	input_files: Sequence[FilePath], sttm_path: FilePath
) -> Dict[str, Any]:
	import pyarrow.parquet as parquet

	file_previews = []
	for file_path in input_files:
		path = Path(file_path)
		try:
			schema = parquet.read_schema(path)
			file_previews.append(
				{
					"path": str(path),
					"columns": [
						{"name": field.name, "dtype": str(field.type)} for field in schema
					],
				}
			)
		except Exception as error:
			file_previews.append({"path": str(path), "error": str(error)})

	try:
		sttm_rules = _normalize_sttm_rows(sttm_path)
		sttm_context: Dict[str, Any] = {"path": str(sttm_path), "rules": sttm_rules}
	except Exception as error:
		sttm_context = {"path": str(sttm_path), "error": str(error), "rules": []}
	return {"files": file_previews, "sttm": sttm_context}


def _cast_column(series: Any, rule: Dict[str, str]) -> Any:
	import pandas as pd

	cast_description = " ".join(
		(rule.get(field, "") for field in ("transformation_type", "transformation_logic"))
	).lower()
	if re.search(r"\b(integer|int|int32|int64)\b", cast_description):
		return pd.to_numeric(series, errors="coerce").round().astype("Int64")
	if re.search(r"\b(float|double|numeric|decimal)\b", cast_description):
		return pd.to_numeric(series, errors="coerce")
	if re.search(r"\b(date|datetime|timestamp)\b", cast_description):
		return _standardize_dates(series)
	return series


def _standardize_dates(series: Any) -> Any:
	import pandas as pd

	parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
	for date_format in (
		"%Y-%m-%d",
		"%m/%d/%Y",
		"%m-%d-%Y",
		"%Y/%m/%d",
		"%d-%m-%Y",
		"%d/%m/%Y",
	):
		candidate = pd.to_datetime(series, format=date_format, errors="coerce", utc=True)
		parsed = parsed.fillna(candidate)
	fallback = pd.to_datetime(series, format="mixed", errors="coerce", utc=True)
	parsed = parsed.fillna(fallback)
	return parsed.dt.strftime("%Y-%m-%d")


def _null_strategy(rule: Dict[str, str]) -> str:
	text = " ".join(
		(rule.get(field, "") for field in ("transformation_type", "transformation_logic"))
	).lower().replace("_", " ")
	if "dropna" in text or "drop null" in text or "drop missing" in text:
		return "dropna"
	for strategy in ("mean", "median", "mode"):
		if re.search(rf"\b{strategy}\b", text):
			return strategy
	if "fillna" in text or "fill null" in text or "constant" in text:
		return "constant"
	return ""


def _constant_fill_value(rule: Dict[str, str]) -> Any:
	text = " ".join(
		(rule.get(field, "") for field in ("transformation_type", "transformation_logic"))
	).replace("_", " ")
	patterns = (
		r"\bconstant(?:\s+value)?\s*(?:with|is|[:=])\s*['\"]?(.+?)['\"]?$",
		r"\bfillna\s*(?:with\s*)?(?:value\s*)?[:=]\s*['\"]?(.+?)['\"]?$",
	)
	for pattern in patterns:
		match = re.search(pattern, text, flags=re.IGNORECASE)
		if match:
			return match.group(1).strip()
	return "unknown"


def _apply_null_rule(dataframe: Any, column: str, rule: Dict[str, str]) -> Any:
	import pandas as pd

	if column not in dataframe.columns:
		return dataframe
	strategy = _null_strategy(rule)
	if strategy == "dropna":
		return dataframe.dropna(subset=[column])
	if strategy in {"mean", "median"}:
		values = pd.to_numeric(dataframe[column], errors="coerce")
		fill_value = values.mean() if strategy == "mean" else values.median()
		if pd.notna(fill_value):
			dataframe[column] = values.fillna(fill_value)
	elif strategy == "mode":
		modes = dataframe[column].mode(dropna=True)
		if not modes.empty:
			dataframe[column] = dataframe[column].fillna(modes.iloc[0])
	elif strategy == "constant":
		dataframe[column] = dataframe[column].fillna(_constant_fill_value(rule))
	return dataframe


def _normalize_text(series: Any, rule: Dict[str, str]) -> Any:
	text = " ".join(
		(rule.get(field, "") for field in ("transformation_type", "transformation_logic"))
	).lower().replace("_", " ")
	values = series.astype("string")
	if "lowercase" in text or "lower case" in text or "lower()" in text:
		return values.str.lower()
	if "uppercase" in text or "upper case" in text or "upper()" in text:
		return values.str.upper()
	if "title case" in text or "titlecase" in text or "title()" in text:
		return values.str.title()
	if "strip" in text or "trim" in text:
		return values.str.strip()
	return series


def _apply_silver_rules(
	input_files: Sequence[FilePath], sttm_path: FilePath, run_id: str
) -> List[str]:
	import pandas as pd

	sttm_rules = _normalize_sttm_rows(sttm_path)
	silver_dir = user_directory("silver_layer", SILVER_DIR)
	silver_dir.mkdir(parents=True, exist_ok=True)
	output_paths = []

	for file_index, file_path in enumerate(input_files):
		source_path = Path(file_path)
		dataframe = pd.read_parquet(source_path)
		table_rules = [
			rule
			for rule in sttm_rules
			if rule.get("source_table") in {source_path.stem, ""}
		]
		table_name = source_path.stem
		if table_name.endswith("_bronze"):
			table_name = table_name[: -len("_bronze")]
		approved_columns = {
			rule["target_column"] for rule in table_rules if rule.get("target_column")
		}
		rename_map = {}
		for rule in table_rules:
			source_column = rule.get("source_column", "")
			target_column = rule.get("target_column", "")
			if source_column in dataframe.columns and target_column:
				rename_map[source_column] = target_column
		if rename_map:
			dataframe = dataframe.rename(columns=rename_map)

		for rule in table_rules:
			target_column = rule.get("target_column", "")
			source_column = rule.get("source_column", "")
			column = target_column if target_column in dataframe.columns else source_column
			transformation_type = rule.get("transformation_type", "").strip().lower()
			if transformation_type in CUSTOM_TRANSFORMATION_TYPES:
				if target_column:
					dataframe[target_column] = apply_custom_transformation(dataframe, rule)
				continue
			if not column:
				continue
			dataframe = _apply_null_rule(dataframe, column, rule)
			dataframe[column] = _cast_column(dataframe[column], rule)
			dataframe[column] = _normalize_text(dataframe[column], rule)

		dataframe = dataframe.drop_duplicates(ignore_index=True)
		key_column = f"pk_{table_name}_silver_id"
		dataframe[key_column] = range(1, len(dataframe) + 1)
		approved_columns.discard(key_column)
		columns_to_keep = [key_column] + [
			column for column in dataframe.columns if column in approved_columns
		]
		dataframe = dataframe.loc[:, columns_to_keep]

		output_path = silver_dir / f"{table_name}_silver.parquet"
		if output_path in [Path(path) for path in output_paths]:
			output_path = silver_dir / f"{table_name}_{file_index + 1}_silver.parquet"
		dataframe.to_parquet(output_path, index=False)
		output_paths.append(str(output_path))

	return output_paths


def _make_silver_tools(
	input_files: Sequence[FilePath], sttm_path: FilePath, run_id: str
) -> List[Any]:
	from langchain_core.tools import tool

	audit_logger = AuditLogger(run_id)
	paths = [str(path) for path in input_files]

	@tool("inspect_task_tool")
	def inspect_task_tool() -> str:
		"""Preview Bronze Parquet schemas and normalized Silver STTM rules."""
		context = _inspect_task(input_files, sttm_path)
		audit_logger.log(
			"silver_agent",
			"inspect_task",
			input_files=paths,
			sttm_path=str(sttm_path),
		)
		return json.dumps(context, default=str, ensure_ascii=False)

	@tool("silver_ingestion_tool")
	def silver_ingestion_tool() -> str:
		"""Clean Bronze Parquet files and write Silver Parquet outputs."""
		output_paths = _apply_silver_rules(input_files, sttm_path, run_id)
		audit_logger.log(
			"silver_agent",
			"silver_ingestion",
			input_files=paths,
			output_paths=output_paths,
		)
		return json.dumps(output_paths)

	return [inspect_task_tool, silver_ingestion_tool]


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


def execute_silver(
	input_files: Sequence[FilePath],
	sttm_path: FilePath,
	run_id: str,
	task_description: str,
) -> List[str]:
	from langchain_core.messages import ToolMessage
	from langgraph.prebuilt import create_react_agent

	paths = [Path(file_path) for file_path in input_files]
	if not paths:
		raise ValueError("At least one Bronze Parquet file is required.")

	trace = AgentTrace("silver_agent", run_id)
	trace.set_input(
		input_files=[str(path) for path in paths],
		sttm_path=str(sttm_path),
		task_description=task_description,
	)
	trace.set_plan("Inspect Bronze schemas and Silver rules, then execute Silver cleansing.")

	try:
		agent = create_react_agent(
			_make_llm(),
			_make_silver_tools(paths, sttm_path, run_id),
			prompt=SILVER_AGENT_PROMPT,
		)
		response = agent.invoke(
			{
				"messages": [
					{
						"role": "user",
						"content": task_description
						+ "\nBronze files: "
						+ ", ".join(str(path) for path in paths)
						+ f"\nSilver STTM: {sttm_path}",
					}
				]
			}
		)
		messages = response.get("messages", [])
		trace.extract_from_messages(messages)
		tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
		if len(tool_messages) < 2 or [message.name for message in tool_messages[:2]] != [
			"inspect_task_tool",
			"silver_ingestion_tool",
		]:
			raise RuntimeError(
				"The agent must call inspect_task_tool, then silver_ingestion_tool."
			)

		try:
			output_paths = json.loads(tool_messages[1].content)
		except (json.JSONDecodeError, TypeError) as error:
			raise RuntimeError("silver_ingestion_tool returned invalid output paths.") from error
		if not isinstance(output_paths, list) or not output_paths:
			raise RuntimeError("Silver ingestion did not return any output paths.")
		if any(not Path(path).is_file() for path in output_paths):
			raise RuntimeError("One or more Silver output files were not created.")

		trace.set_output(output_paths=output_paths)
		trace.complete()
		return [str(path) for path in output_paths]
	except Exception as error:
		trace.fail(error)
		raise