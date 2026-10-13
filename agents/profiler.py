import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from core.audit import AuditLogger
from core.cache import (
	build_cache_key,
	configured_model_signature,
	fingerprint_file,
	get_cached_json,
	set_cached_json,
)
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
	PROFILES_DIR,
)
from core.data_io import read_tabular_datasets
from core.observability import AgentTrace
from core.performance import record_metric
from core.user_workspace import user_directory

PROFILER_AGENT_PROMPT = """
You are a Data Analyst specializing in data profiling.
Follow this sequence: THINK → INSPECT → PLAN → ACT → VERIFY.
First call inspect_files_tool to preview every input file.
Then call profiler_tool to get full column statistics.
Return only valid JSON with these keys:
{
  "semantic_meanings": [],
  "join_keys": [],
  "quality_notes": []
}
Summarize conclusions concisely. Do not expose private chain-of-thought.
""".strip()


FilePath = Union[str, Path]


def _json_default(value: Any) -> Any:
	if hasattr(value, "item"):
		return value.item()
	if hasattr(value, "isoformat"):
		return value.isoformat()
	return str(value)


def _inspect_files(file_paths: Sequence[FilePath]) -> Dict[str, Any]:
	inspections = []
	for file_path in file_paths:
		path = Path(file_path)
		try:
			for dataset in read_tabular_datasets(path):
				dataframe = dataset["dataframe"]
				inspections.append(
					{
						"path": str(path),
						"table_name": dataset["table_name"],
						"sheet_name": dataset["sheet_name"],
						"shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
						"columns": [str(column) for column in dataframe.columns],
						"dtypes": {
							str(column): str(dtype)
							for column, dtype in dataframe.dtypes.items()
						},
						"sample_values": {
							str(column): [
								_json_default(value)
								for value in dataframe[column].dropna().head(3).tolist()
							]
							for column in dataframe.columns
						},
					}
				)
		except Exception as error:
			inspections.append({"path": str(path), "error": str(error)})

	return {"files": inspections}


def _compute_stats(file_paths: Sequence[FilePath]) -> Dict[str, Any]:
	import pandas as pd

	profiles = []
	for file_path in file_paths:
		path = Path(file_path)
		try:
			for dataset in read_tabular_datasets(path):
				dataframe = dataset["dataframe"]
				column_stats = {}
				for column in dataframe.columns:
					series = dataframe[column]
					null_count = int(series.isna().sum())
					stats = {
						"dtype": str(series.dtype),
						"null_count": null_count,
						"null_pct": round(null_count / len(dataframe) * 100, 2)
						if len(dataframe)
						else 0.0,
						"unique_count": int(series.nunique(dropna=True)),
					}
					if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
						series.dtype
					):
						numeric = pd.to_numeric(series, errors="coerce").dropna()
						stats.update(
							{
								"min": _json_default(numeric.min()) if not numeric.empty else None,
								"max": _json_default(numeric.max()) if not numeric.empty else None,
								"mean": _json_default(numeric.mean()) if not numeric.empty else None,
							}
						)
					else:
						stats["sample_values"] = [
							_json_default(value) for value in series.dropna().head(5).tolist()
						]
					column_stats[str(column)] = stats

				profiles.append(
					{
						"path": str(path),
						"table_name": dataset["table_name"],
						"sheet_name": dataset["sheet_name"],
						"shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
						"columns": column_stats,
					}
				)
		except Exception as error:
			profiles.append({"path": str(path), "error": str(error)})

	return {"files": profiles}


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
				"table": item.get("table_name")
					or (Path(path).stem if path else "source_table"),
				"sheet_name": item.get("sheet_name"),
				"columns": [str(column) for column in columns],
			}
		)
	return sources

def _make_profiler_tools(
	file_paths: Sequence[FilePath], run_id: str
) -> List[Any]:
	from langchain_core.tools import tool

	audit_logger = AuditLogger(run_id)
	paths = [str(path) for path in file_paths]

	@tool("inspect_files_tool")
	def inspect_files_tool() -> str:
		"""Preview CSV files and every worksheet in XLSX workbooks."""
		result = _inspect_files(file_paths)
		audit_logger.log("profiler_agent", "inspect_files", file_paths=paths)
		return json.dumps(result, default=_json_default, ensure_ascii=False)

	@tool("profiler_tool")
	def profiler_tool() -> str:
		"""Calculate null, uniqueness, numeric, and text statistics for each dataset."""
		result = _compute_stats(file_paths)
		audit_logger.log("profiler_agent", "compute_profile_stats", file_paths=paths)
		return json.dumps(result, default=_json_default, ensure_ascii=False)

	return [inspect_files_tool, profiler_tool]


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


def _message_content(message: Any) -> str:
	content = getattr(message, "content", "")
	if isinstance(content, str):
		return content
	return json.dumps(content, default=_json_default, ensure_ascii=False)


def _parse_analysis(response_text: str) -> Dict[str, Any]:
	text = response_text.strip()
	if text.startswith("```"):
		text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
	try:
		parsed = json.loads(text)
	except (json.JSONDecodeError, TypeError):
		parsed = {}
	if not isinstance(parsed, dict):
		parsed = {}
	return {
		"semantic_meanings": parsed.get("semantic_meanings", []),
		"join_keys": parsed.get("join_keys", []),
		"quality_notes": parsed.get("quality_notes", []),
		"raw_response": response_text if not parsed else None,
	}


def profile_dataset(file_path: FilePath, run_id: str, task_description: str) -> str:
	return profile_multiple_datasets([file_path], run_id, task_description)


def profile_multiple_datasets(
	file_paths: Sequence[FilePath], run_id: str, task_description: str
) -> str:
	from langchain_core.messages import ToolMessage
	from langgraph.prebuilt import create_react_agent

	paths = [Path(file_path) for file_path in file_paths]
	if not paths:
		raise ValueError("At least one CSV or XLSX file path is required.")

	timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
	profiles_dir = user_directory("profiles", PROFILES_DIR)
	profile_path = profiles_dir / f"profile_combined_{timestamp}.json"
	trace = AgentTrace("profiler", run_id)
	trace.set_input(
		file_paths=[str(path) for path in paths],
		task_description=task_description,
	)
	trace.set_plan("Inspect each dataset, calculate statistics, and summarize its meaning and quality.")

	try:
		cache_key = build_cache_key(
			"profile",
			{
				"files": [
					{"path": str(path.resolve()), **fingerprint_file(path)} for path in paths
				],
				"task_description": task_description,
				"prompt": PROFILER_AGENT_PROMPT,
				"model": configured_model_signature(),
			},
		)
		cached_profile = get_cached_json("profile", cache_key)
		record_metric(
			run_id,
			"profiler",
			"cache_lookup",
			1,
			unit="lookup",
			details={"namespace": "profile", "status": "hit" if cached_profile else "miss"},
		)
		if cached_profile:
			cached_profile = dict(cached_profile)
			cached_profile.update(
				{
					"run_id": run_id,
					"created_at": datetime.now(timezone.utc).isoformat(),
					"task_description": task_description,
					"files": [str(path) for path in paths],
				}
			)
			profiles_dir.mkdir(parents=True, exist_ok=True)
			with profile_path.open("w", encoding="utf-8") as profile_file:
				json.dump(cached_profile, profile_file, indent=2, default=_json_default)
				profile_file.write("\n")
			trace.set_output(profile_path=str(profile_path), cache_hit=bool(cached_profile))
			trace.complete()
			return str(profile_path)

		tools = _make_profiler_tools(paths, run_id)
		agent = create_react_agent(_make_llm(), tools, prompt=PROFILER_AGENT_PROMPT)
		response = agent.invoke(
			{
				"messages": [
					{
						"role": "user",
						"content": (
							task_description
							+ "\nProfile these CSV/XLSX datasets: "
							+ ", ".join(str(path) for path in paths)
						),
					}
				]
			}
		)
		messages = response.get("messages", [])
		trace.extract_from_messages(messages)

		tool_results = {}
		final_response = ""
		for message in messages:
			if isinstance(message, ToolMessage):
				try:
					tool_results[message.name] = json.loads(_message_content(message))
				except (json.JSONDecodeError, TypeError):
					continue
			elif getattr(message, "type", None) == "ai" and message.content:
				final_response = _message_content(message)

		analysis = _parse_analysis(final_response)
		combined_profile = {
			"run_id": run_id,
			"created_at": datetime.now(timezone.utc).isoformat(),
			"task_description": task_description,
			"files": [str(path) for path in paths],
			"inspection": tool_results.get("inspect_files_tool") or _inspect_files(paths),
			"statistics": tool_results.get("profiler_tool") or _compute_stats(paths),
			**analysis,
		}
		profiles_dir.mkdir(parents=True, exist_ok=True)
		with profile_path.open("w", encoding="utf-8") as profile_file:
			json.dump(combined_profile, profile_file, indent=2, default=_json_default)
			profile_file.write("\n")
		set_cached_json("profile", cache_key, combined_profile)

		trace.set_output(profile_path=str(profile_path), cache_hit=False)
		trace.complete()
		return str(profile_path)
	except Exception as error:
		trace.fail(error)
		raise