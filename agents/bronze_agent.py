import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from core.audit import AuditLogger
from core.config import (
	BRONZE_DIR,
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
from core.custom_transformations import (
	CUSTOM_TRANSFORMATION_TYPES,
	apply_custom_transformation,
)
from core.data_io import read_tabular_datasets
from core.observability import AgentTrace
from core.user_workspace import user_directory

BRONZE_AGENT_PROMPT = """
You are a careful Bronze-layer ingestion agent. Bronze preserves raw source data
with only the transformations explicitly described in the STTM, plus load
metadata.
You MUST call both tools in this order: inspect_task_tool, then
bronze_ingestion_tool. Do not stop after inspecting or planning.
Planning without actually calling bronze_ingestion_tool is FAILURE.
Use the STTM to rename columns and apply requested numeric or datetime casts.
Verify that bronze_ingestion_tool completed and report its output paths.
""".strip()

FilePath = Union[str, Path]


def _read_sttm(sttm_path: FilePath) -> List[Dict[str, str]]:
	with Path(sttm_path).open("r", encoding="utf-8", newline="") as sttm_file:
		return list(csv.DictReader(sttm_file))


def _inspect_task(
	input_files: Sequence[FilePath], sttm_path: FilePath
) -> Dict[str, Any]:
	file_previews = []
	for file_path in input_files:
		path = Path(file_path)
		try:
			for dataset in read_tabular_datasets(path):
				dataframe = dataset["dataframe"]
				file_previews.append(
					{
						"path": str(path),
						"table_name": dataset["table_name"],
						"sheet_name": dataset["sheet_name"],
						"shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
						"columns": [str(column) for column in dataframe.columns],
						"sample_rows": dataframe.head(3).to_dict(orient="records"),
					}
				)
		except Exception as error:
			file_previews.append({"path": str(path), "error": str(error)})

	try:
		sttm_rules = _read_sttm(sttm_path)
		sttm_context: Dict[str, Any] = {"path": str(sttm_path), "rules": sttm_rules}
	except Exception as error:
		sttm_context = {"path": str(sttm_path), "error": str(error), "rules": []}
	return {"files": file_previews, "sttm": sttm_context}


def _cast_column(series: Any, rule: Dict[str, str]) -> Any:
	import pandas as pd

	cast_description = " ".join(
		(rule.get(field, "") for field in ("transformation_type", "transformation_logic"))
	).lower()
	if re.search(r"\b(to_numeric|numeric|integer|int|float|double|decimal)\b", cast_description):
		return pd.to_numeric(series, errors="coerce")
	if re.search(r"\b(to_datetime|datetime|timestamp|date)\b", cast_description):
		return pd.to_datetime(series, errors="coerce", utc=True)
	return series


def _apply_bronze_rules(
	input_files: Sequence[FilePath], sttm_path: FilePath, run_id: str
) -> List[str]:
	sttm_rules = _read_sttm(sttm_path)
	bronze_dir = user_directory("bronze_layer", BRONZE_DIR)
	bronze_dir.mkdir(parents=True, exist_ok=True)
	output_paths = []
	load_timestamp = datetime.now(timezone.utc).isoformat()

	for file_path in input_files:
		source_path = Path(file_path)
		for dataset_index, dataset in enumerate(read_tabular_datasets(source_path)):
			dataframe = dataset["dataframe"]
			source_table = dataset["table_name"]
			table_rules = [
				rule
				for rule in sttm_rules
				if rule.get("source_table") in {source_table, ""}
			]
			rename_map = {}
			for rule in table_rules:
				source_column = rule.get("source_column", "")
				target_column = rule.get("target_column", "")
				transformation_type = rule.get("transformation_type", "").strip().lower()
				if transformation_type in CUSTOM_TRANSFORMATION_TYPES:
					if not target_column:
						continue
					dataframe[target_column] = apply_custom_transformation(dataframe, rule)
					if source_column and source_column != target_column and source_column in dataframe.columns:
						dataframe = dataframe.drop(columns=[source_column])
					continue
				if transformation_type == "metadata" or not source_column:
					continue
				if source_column not in dataframe.columns or not target_column:
					continue
				dataframe[source_column] = _cast_column(dataframe[source_column], rule)
				if source_column != target_column:
					rename_map[source_column] = target_column

			if rename_map:
				dataframe = dataframe.rename(columns=rename_map)
			dataframe["_load_timestamp"] = load_timestamp
			source_file = str(source_path)
			if dataset["sheet_name"] is not None:
				source_file = f"{source_file}#{dataset['sheet_name']}"
			dataframe["_source_file"] = source_file

			output_path = bronze_dir / f"{source_table}_bronze.parquet"
			name_counter = 2
			while str(output_path) in output_paths:
				output_path = bronze_dir / f"{source_table}_{name_counter}_bronze.parquet"
				name_counter += 1
			dataframe.to_parquet(output_path, index=False)
			output_paths.append(str(output_path))

	return output_paths


def _make_bronze_tools(
	input_files: Sequence[FilePath], sttm_path: FilePath, run_id: str
) -> List[Any]:
	from langchain_core.tools import tool

	audit_logger = AuditLogger(run_id)
	paths = [str(path) for path in input_files]

	@tool("inspect_task_tool")
	def inspect_task_tool() -> str:
		"""Preview input CSV files and the supplied Bronze STTM rules."""
		context = _inspect_task(input_files, sttm_path)
		audit_logger.log(
			"bronze_agent",
			"inspect_task",
			input_files=paths,
			sttm_path=str(sttm_path),
		)
		return json.dumps(context, default=str, ensure_ascii=False)

	@tool("bronze_ingestion_tool")
	def bronze_ingestion_tool() -> str:
		"""Apply Bronze STTM transformations and write Parquet outputs."""
		output_paths = _apply_bronze_rules(input_files, sttm_path, run_id)
		audit_logger.log(
			"bronze_agent",
			"bronze_ingestion",
			input_files=paths,
			output_paths=output_paths,
		)
		return json.dumps(output_paths)

	return [inspect_task_tool, bronze_ingestion_tool]


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


def execute_bronze(
	input_files: Sequence[FilePath],
	sttm_path: FilePath,
	run_id: str,
	task_description: str,
) -> List[str]:
	from langchain_core.messages import ToolMessage
	from langgraph.prebuilt import create_react_agent

	paths = [Path(file_path) for file_path in input_files]
	if not paths:
		raise ValueError("At least one input CSV file is required.")

	trace = AgentTrace("bronze_agent", run_id)
	trace.set_input(
		input_files=[str(path) for path in paths],
		sttm_path=str(sttm_path),
		task_description=task_description,
	)
	trace.set_plan("Inspect CSV files and STTM rules, then execute Bronze ingestion.")

	try:
		agent = create_react_agent(
			_make_llm(),
			_make_bronze_tools(paths, sttm_path, run_id),
			prompt=BRONZE_AGENT_PROMPT,
		)
		response = agent.invoke(
			{
				"messages": [
					{
						"role": "user",
						"content": task_description
						+ "\nInput files: "
						+ ", ".join(str(path) for path in paths)
						+ f"\nBronze STTM: {sttm_path}",
					}
				]
			}
		)
		messages = response.get("messages", [])
		trace.extract_from_messages(messages)
		tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
		if len(tool_messages) < 2 or [message.name for message in tool_messages[:2]] != [
			"inspect_task_tool",
			"bronze_ingestion_tool",
		]:
			raise RuntimeError(
				"The agent must call inspect_task_tool, then bronze_ingestion_tool."
			)

		try:
			output_paths = json.loads(tool_messages[1].content)
		except (json.JSONDecodeError, TypeError) as error:
			raise RuntimeError("bronze_ingestion_tool returned invalid output paths.") from error
		if not isinstance(output_paths, list) or not output_paths:
			raise RuntimeError("Bronze ingestion did not return any output paths.")
		if any(not Path(path).is_file() for path in output_paths):
			raise RuntimeError("One or more Bronze output files were not created.")

		trace.set_output(output_paths=output_paths)
		trace.complete()
		return [str(path) for path in output_paths]
	except Exception as error:
		trace.fail(error)
		raise