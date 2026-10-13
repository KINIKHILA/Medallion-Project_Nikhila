"""Coordinate the retail analytics pipeline in four phases:

Phase 1: Profile data and generate Bronze STTM rules.
Phase 2: Execute Bronze ingestion and generate Silver STTM rules.
Phase 3: Execute Silver cleansing and generate Gold STTM rules.
Phase 4: Execute Gold aggregation and generate the business report.
"""

import json
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from agents.bronze_agent import execute_bronze
from agents.gold_agent import execute_gold
from agents.profiler import profile_multiple_datasets
from agents.reporter import generate_report
from agents.silver_agent import execute_silver
from agents.sttm_generator import (
	generate_bronze_sttm,
	generate_gold_sttm,
	generate_silver_sttm,
)
from core.audit import AuditLogger
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
from core.data_quality import assess_parquet_files
from core.email_notifications import send_report_email
from core.exceptions import DataQualityGateError, PipelineError, PipelinePhaseError
from core.observability import AgentTrace
from core.retry import is_transient_error, retry_transient
from core.run_store import save_run_state


class PipelineState(TypedDict, total=False):
	"""State values exposed to the pipeline UI and specialist agents."""

	run_id: str
	status: str
	uploaded_files: List[str]
	business_intent: str
	profile_path: str
	sttm_bronze_path: str
	sttm_silver_path: str
	sttm_gold_path: str
	bronze_sttm_approved: bool
	silver_sttm_approved: bool
	gold_sttm_approved: bool
	bronze_output_paths: List[str]
	silver_output_paths: List[str]
	gold_output_paths: List[str]
	report_path: str
	error: Optional[str]
	quality_report: Dict[str, Any]
	quality_issues: List[Dict[str, Any]]
	failed_phase: Optional[str]
	last_completed_phase: str
	error_details: Dict[str, Any]


SUPERVISOR_PROMPT = """
You are the Supervisor for the IDAMP retail analytics pipeline. Coordinate the
specialist AI agents by calling them as tools; do not perform their work yourself.
You will receive TWO STEPS to execute. Call the corresponding specialist tool for
each step EXACTLY ONCE, in the order provided. Each tool is an autonomous agent
that handles its own execution, so do not duplicate its work.
Verify each tool's output and confirm that it completed successfully before
proceeding to the next step. If a tool fails or returns an invalid result, stop,
report the failure, and do not claim that later steps completed.
""".strip()


def _make_llm():
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


def _run_supervisor(
	tools: List[Any], phase_goal: str, phase_name: str, run_id: str
) -> Dict[str, Any]:
	from langgraph.prebuilt import create_react_agent

	trace = AgentTrace(f"supervisor_{phase_name}", run_id)
	trace.set_input(phase=phase_name, phase_goal=phase_goal)
	trace.set_plan("Dispatch the two requested specialist tools in order and verify their outputs.")
	try:
		agent = create_react_agent(_make_llm(), tools, prompt=SUPERVISOR_PROMPT)
		audit_logger = AuditLogger(run_id)

		def log_retry(error: BaseException, attempt: int) -> None:
			audit_logger.log(
				"supervisor",
				"transient_retry",
				phase=phase_name,
				attempt=attempt,
				error_type=type(error).__name__,
				error=str(error),
			)

		result = retry_transient(
			lambda: agent.invoke(
				{"messages": [{"role": "user", "content": phase_goal}]}
			),
			max_attempts=3,
			on_retry=log_retry,
		)
		if not isinstance(result, dict):
			raise RuntimeError("Supervisor returned an invalid result.")
		messages = result.get("messages", [])
		trace.extract_from_messages(messages)
		trace.set_output(message_count=len(messages))
		trace.complete()
		return result
	except Exception as error:
		trace.fail(error)
		raise


def _begin_phase(state: PipelineState, phase_name: str) -> tuple[str, AuditLogger]:
	run_id = state.get("run_id") or str(uuid.uuid4())
	state["run_id"] = run_id
	state["status"] = "running"
	state["error"] = None
	state["quality_report"] = {}
	state["quality_issues"] = []
	state.pop("failed_phase", None)
	state.pop("error_details", None)
	audit_logger = AuditLogger(run_id)
	audit_logger.log("orchestrator", "phase_start", phase=phase_name)
	save_run_state(state)
	return run_id, audit_logger


def _record_phase_failure(
	state: PipelineState,
	audit_logger: AuditLogger,
	phase_name: str,
	error: Exception,
) -> PipelineState:
	state["status"] = "failed"
	state["failed_phase"] = phase_name
	state["error"] = str(error)
	error_context = error.context if isinstance(error, PipelineError) else {}
	state["error_details"] = {
		"phase": phase_name,
		"code": getattr(error, "code", "unexpected_error"),
		"exception_type": type(error).__name__,
		"message": str(error),
		"retryable": is_transient_error(error),
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"context": error_context,
		"traceback": "".join(
			traceback.format_exception(type(error), error, error.__traceback__)
		),
	}
	audit_logger.log(
		"orchestrator",
		"phase_fail",
		phase=phase_name,
		run_id=state.get("run_id"),
		error=state["error_details"],
	)
	save_run_state(state)
	return state


def _required_path(scratchpad: Dict[str, Any], key: str) -> str:
	path = scratchpad.get(key)
	if not isinstance(path, str) or not path:
		raise RuntimeError(f"Phase did not produce required state value '{key}'.")
	return path


def _required_paths(scratchpad: Dict[str, Any], key: str) -> List[str]:
	paths = scratchpad.get(key)
	if not isinstance(paths, list) or not paths:
		raise RuntimeError(f"Phase did not produce required state value '{key}'.")
	return [str(path) for path in paths]


def _check_layer_quality(
	state: PipelineState,
	output_paths: List[str],
	layer: str,
	phase_name: str,
	audit_logger: AuditLogger,
) -> bool:
	quality_report = assess_parquet_files(output_paths, layer)
	state["quality_report"] = quality_report
	state["quality_issues"] = quality_report["issues"]
	audit_logger.log(
		"orchestrator",
		"data_quality_check",
		phase=phase_name,
		layer=layer,
		passed=quality_report["passed"],
		score=quality_report["score"],
		issue_count=len(quality_report["issues"]),
	)
	if not quality_report["passed"]:
		error = DataQualityGateError(phase_name, quality_report)
		_record_phase_failure(state, audit_logger, phase_name, error)
		return False
	return True


def run_until_bronze_sttm(
	uploaded_files: List[str],
	business_intent: str,
	run_id: Optional[str] = None,
) -> PipelineState:
	run_id = run_id or str(uuid.uuid4())
	state: PipelineState = {
		"run_id": run_id,
		"status": "initialized",
		"uploaded_files": [str(path) for path in uploaded_files],
		"business_intent": business_intent,
		"profile_path": "",
		"sttm_bronze_path": "",
		"sttm_silver_path": "",
		"sttm_gold_path": "",
		"bronze_sttm_approved": False,
		"silver_sttm_approved": False,
		"gold_sttm_approved": False,
		"bronze_output_paths": [],
		"silver_output_paths": [],
		"gold_output_paths": [],
		"report_path": "",
		"error": None,
		"quality_report": {},
		"quality_issues": [],
		"failed_phase": None,
		"last_completed_phase": "",
		"error_details": {},
	}
	phase_name = "phase1_profile_and_bronze_sttm"
	_, audit_logger = _begin_phase(state, phase_name)
	try:
		tools, scratchpad = _make_phase1_tools(state["uploaded_files"], run_id)
		phase_goal = (
			f"Business intent: {business_intent}\n"
			"Step 1: Call profiler_agent_tool exactly once to profile the uploaded files. "
			"Verify profile_path was returned. Step 2: Call sttm_agent_tool exactly "
			"once to generate Bronze STTM rules, then verify sttm_bronze_path."
		)
		_run_supervisor(tools, phase_goal, phase_name, run_id)
		state["profile_path"] = _required_path(scratchpad, "profile_path")
		state["sttm_bronze_path"] = _required_path(scratchpad, "sttm_bronze_path")
		state["status"] = "awaiting_bronze_sttm_approval"
		state["last_completed_phase"] = phase_name
		audit_logger.log(
			"orchestrator",
			"phase_complete",
			phase=phase_name,
			profile_path=state["profile_path"],
			sttm_bronze_path=state["sttm_bronze_path"],
		)
		save_run_state(state)
	except Exception as error:
		return _record_phase_failure(state, audit_logger, phase_name, error)
	return state


def run_bronze_to_silver_sttm(state: PipelineState) -> PipelineState:
	phase_name = "phase2_bronze_and_silver_sttm"
	run_id, audit_logger = _begin_phase(state, phase_name)
	state["bronze_sttm_approved"] = True
	try:
		bronze_sttm_path = _required_path(state, "sttm_bronze_path")
		tools, scratchpad = _make_phase2_tools(
			state.get("uploaded_files", []), bronze_sttm_path, run_id
		)
		phase_goal = (
			"Step 1: Call bronze_agent_tool exactly once to ingest the approved Bronze "
			"rules and verify bronze_output_paths. Step 2: Call sttm_agent_tool exactly "
			"once to generate Silver STTM rules and verify sttm_silver_path."
		)
		_run_supervisor(tools, phase_goal, phase_name, run_id)
		state["bronze_output_paths"] = _required_paths(
			scratchpad, "bronze_output_paths"
		)
		state["sttm_silver_path"] = _required_path(scratchpad, "sttm_silver_path")
		if not _check_layer_quality(
			state,
			state["bronze_output_paths"],
			"bronze",
			phase_name,
			audit_logger,
		):
			return state
		state["status"] = "awaiting_silver_sttm_approval"
		state["last_completed_phase"] = phase_name
		audit_logger.log(
			"orchestrator",
			"phase_complete",
			phase=phase_name,
			bronze_output_paths=state["bronze_output_paths"],
			sttm_silver_path=state["sttm_silver_path"],
		)
		save_run_state(state)
	except Exception as error:
		return _record_phase_failure(state, audit_logger, phase_name, error)
	return state


def run_silver_to_gold_sttm(state: PipelineState) -> PipelineState:
	phase_name = "phase3_silver_and_gold_sttm"
	run_id, audit_logger = _begin_phase(state, phase_name)
	state["silver_sttm_approved"] = True
	try:
		sttm_silver_path = _required_path(state, "sttm_silver_path")
		tools, scratchpad = _make_phase3_tools(
			state.get("bronze_output_paths", []),
			sttm_silver_path,
			state.get("business_intent", ""),
			run_id,
		)
		phase_goal = (
			"Step 1: Call silver_agent_tool exactly once to cleanse Bronze data and "
			"verify silver_output_paths. Step 2: Call sttm_agent_tool exactly once to "
			"generate Gold STTM rules and verify sttm_gold_path."
		)
		_run_supervisor(tools, phase_goal, phase_name, run_id)
		state["silver_output_paths"] = _required_paths(
			scratchpad, "silver_output_paths"
		)
		state["sttm_gold_path"] = _required_path(scratchpad, "sttm_gold_path")
		if not _check_layer_quality(
			state,
			state["silver_output_paths"],
			"silver",
			phase_name,
			audit_logger,
		):
			return state
		state["status"] = "awaiting_gold_sttm_approval"
		state["last_completed_phase"] = phase_name
		audit_logger.log(
			"orchestrator",
			"phase_complete",
			phase=phase_name,
			silver_output_paths=state["silver_output_paths"],
			sttm_gold_path=state["sttm_gold_path"],
		)
		save_run_state(state)
	except Exception as error:
		return _record_phase_failure(state, audit_logger, phase_name, error)
	return state


def run_gold_and_report(state: PipelineState) -> PipelineState:
	phase_name = "phase4_gold_and_report"
	run_id, audit_logger = _begin_phase(state, phase_name)
	state["gold_sttm_approved"] = True
	scratchpad: Dict[str, Any] = {}
	try:
		sttm_gold_path = _required_path(state, "sttm_gold_path")
		tools, scratchpad = _make_phase4_tools(
			state.get("silver_output_paths", []),
			sttm_gold_path,
			state.get("business_intent", ""),
			run_id,
		)
		phase_goal = (
			"Step 1: Call gold_agent_tool exactly once to build Gold tables and verify "
			"gold_output_paths. Step 2: Call reporter_agent_tool exactly once to query "
			"the Gold data and create the final report, then verify report_path."
		)
		_run_supervisor(tools, phase_goal, phase_name, run_id)
		quality_report = scratchpad.get("quality_report")
		if quality_report:
			state["quality_report"] = quality_report
			state["quality_issues"] = quality_report["issues"]
			if not quality_report["passed"]:
				return _record_phase_failure(
					state, audit_logger, phase_name, DataQualityGateError(phase_name, quality_report)
				)
		else:
			raise RuntimeError("Gold data quality report was not produced.")
		state["gold_output_paths"] = _required_paths(scratchpad, "gold_output_paths")
		state["report_path"] = _required_path(scratchpad, "report_path")
		state["status"] = "completed"
		state["last_completed_phase"] = phase_name
		try:
			email_sent = send_report_email(
				state["report_path"],
				state.get("business_intent", ""),
				run_id,
			)
			audit_logger.log(
				"orchestrator",
				"report_email_sent" if email_sent else "report_email_skipped",
				phase=phase_name,
				report_path=state["report_path"],
			)
		except Exception as email_error:
			audit_logger.log(
				"orchestrator",
				"report_email_failed",
				phase=phase_name,
				error=str(email_error),
				report_path=state["report_path"],
			)
		audit_logger.log(
			"orchestrator",
			"phase_complete",
			phase=phase_name,
			gold_output_paths=state["gold_output_paths"],
			report_path=state["report_path"],
		)
		save_run_state(state)
	except Exception as error:
		if scratchpad.get("quality_report"):
			state["quality_report"] = scratchpad["quality_report"]
			state["quality_issues"] = scratchpad["quality_report"]["issues"]
		return _record_phase_failure(state, audit_logger, phase_name, error)
	return state


def _make_phase1_tools(
	uploaded_files: List[str], run_id: str
) -> tuple[List[Any], Dict[str, Any]]:
	from langchain_core.tools import tool

	scratchpad: Dict[str, Any] = {}

	@tool("profiler_agent_tool")
	def profiler_agent_tool(goal: str) -> str:
		"""Profile the uploaded CSV files and save the combined profile."""
		print(f"Dispatching profiler agent for run {run_id}: {goal}")
		task_description = (
			f"Phase 1 profiling goal: {goal}\n"
			"Inspect every uploaded CSV, calculate data quality statistics, identify "
			"semantic meanings and possible join keys, and save the combined profile."
		)
		profile_path = profile_multiple_datasets(uploaded_files, run_id, task_description)
		if not profile_path:
			raise RuntimeError("Profiler agent did not return a profile path.")
		scratchpad["profile_path"] = profile_path
		return json.dumps({"profile_path": profile_path})

	@tool("sttm_agent_tool")
	def sttm_agent_tool(goal: str) -> str:
		"""Generate Bronze STTM rules after profiling has completed."""
		profile_path = scratchpad.get("profile_path")
		if not profile_path:
			raise RuntimeError("Run profiler_agent_tool before sttm_agent_tool.")
		print(f"Dispatching Bronze STTM agent for run {run_id}: {goal}")
		task_description = (
			f"Phase 1 Bronze STTM goal: {goal}\n"
			"Create Bronze mappings for all profiled source columns, including load "
			"timestamp and source-file metadata."
		)
		sttm_path = generate_bronze_sttm(profile_path, run_id, task_description)
		if not sttm_path:
			raise RuntimeError("Bronze STTM agent did not return a rules path.")
		scratchpad["sttm_bronze_path"] = sttm_path
		return json.dumps({"sttm_bronze_path": sttm_path})

	return [profiler_agent_tool, sttm_agent_tool], scratchpad


def _make_phase2_tools(
	uploaded_files: List[str], sttm_bronze_path: str, run_id: str
) -> tuple[List[Any], Dict[str, Any]]:
	from langchain_core.tools import tool

	scratchpad: Dict[str, Any] = {}

	@tool("bronze_agent_tool")
	def bronze_agent_tool(goal: str) -> str:
		"""Ingest uploaded CSV files into Bronze Parquet files."""
		print(f"Dispatching Bronze ingestion agent for run {run_id}: {goal}")
		task_description = (
			f"Phase 2 Bronze ingestion goal: {goal}\n"
			"Apply the approved Bronze STTM to every uploaded CSV, preserve raw "
			"values otherwise, add load metadata, and write Parquet outputs."
		)
		output_paths = execute_bronze(
			uploaded_files, sttm_bronze_path, run_id, task_description
		)
		if not output_paths:
			raise RuntimeError("Bronze agent did not return output paths.")
		scratchpad["bronze_output_paths"] = output_paths
		return json.dumps({"bronze_output_paths": output_paths})

	@tool("sttm_agent_tool")
	def sttm_agent_tool(goal: str) -> str:
		"""Generate Silver STTM rules after Bronze ingestion has completed."""
		bronze_output_paths = scratchpad.get("bronze_output_paths")
		if not bronze_output_paths:
			raise RuntimeError("Run bronze_agent_tool before sttm_agent_tool.")
		print(f"Dispatching Silver STTM agent for run {run_id}: {goal}")
		task_description = (
			f"Phase 2 Silver STTM goal: {goal}\n"
			"Inspect Bronze schemas and define approved Silver columns, cleansing, "
			"type standardization, deduplication, and surrogate-key rules."
		)
		sttm_path = generate_silver_sttm(
			bronze_output_paths,
			sttm_bronze_path,
			run_id,
			task_description,
		)
		if not sttm_path:
			raise RuntimeError("Silver STTM agent did not return a rules path.")
		scratchpad["sttm_silver_path"] = sttm_path
		return json.dumps({"sttm_silver_path": sttm_path})

	return [bronze_agent_tool, sttm_agent_tool], scratchpad


def _make_phase3_tools(
	bronze_output_paths: List[str],
	sttm_silver_path: str,
	business_intent: str,
	run_id: str,
) -> tuple[List[Any], Dict[str, Any]]:
	from langchain_core.tools import tool

	scratchpad: Dict[str, Any] = {}

	@tool("silver_agent_tool")
	def silver_agent_tool(goal: str) -> str:
		"""Clean Bronze Parquet files into Silver Parquet files."""
		print(f"Dispatching Silver cleansing agent for run {run_id}: {goal}")
		task_description = (
			f"Phase 3 Silver cleansing goal: {goal}\n"
			f"Business intent: {business_intent}\n"
			"Apply approved Silver transformations, handle nulls, standardize types "
			"and text, remove duplicate rows, and write Parquet outputs."
		)
		output_paths = execute_silver(
			bronze_output_paths, sttm_silver_path, run_id, task_description
		)
		if not output_paths:
			raise RuntimeError("Silver agent did not return output paths.")
		scratchpad["silver_output_paths"] = output_paths
		return json.dumps({"silver_output_paths": output_paths})

	@tool("sttm_agent_tool")
	def sttm_agent_tool(goal: str) -> str:
		"""Generate Gold STTM rules after Silver cleansing has completed."""
		silver_output_paths = scratchpad.get("silver_output_paths")
		if not silver_output_paths:
			raise RuntimeError("Run silver_agent_tool before sttm_agent_tool.")
		print(f"Dispatching Gold STTM agent for run {run_id}: {goal}")
		task_description = (
			f"Phase 3 Gold STTM goal: {goal}\n"
			f"Business intent: {business_intent}\n"
			"Inspect Silver schemas and define Gold joins, business aggregations, "
			"approved output columns, and a surrogate key."
		)
		sttm_path = generate_gold_sttm(
			silver_output_paths,
			sttm_silver_path,
			business_intent,
			run_id,
			task_description,
		)
		if not sttm_path:
			raise RuntimeError("Gold STTM agent did not return a rules path.")
		scratchpad["sttm_gold_path"] = sttm_path
		return json.dumps({"sttm_gold_path": sttm_path})

	return [silver_agent_tool, sttm_agent_tool], scratchpad


def _make_phase4_tools(
	silver_output_paths: List[str],
	sttm_gold_path: str,
	business_intent: str,
	run_id: str,
) -> tuple[List[Any], Dict[str, Any]]:
	from langchain_core.tools import tool

	scratchpad: Dict[str, Any] = {}
	audit_logger = AuditLogger(run_id)

	@tool("gold_agent_tool")
	def gold_agent_tool(goal: str) -> str:
		"""Build analytics-ready Gold tables from Silver data."""
		print(f"Dispatching Gold aggregation agent for run {run_id}: {goal}")
		task_description = (
			f"Phase 4 Gold aggregation goal: {goal}\n"
			f"Business intent: {business_intent}\n"
			"Apply the Gold STTM, join and aggregate Silver data, and save the "
			"analytics-ready Parquet outputs."
		)
		output_paths = execute_gold(
			silver_output_paths, sttm_gold_path, run_id, task_description
		)
		if not output_paths:
			raise RuntimeError("Gold agent did not return output paths.")
		quality_report = assess_parquet_files(output_paths, "gold")
		scratchpad["quality_report"] = quality_report
		audit_logger.log(
			"orchestrator",
			"data_quality_check",
			layer="gold",
			passed=quality_report["passed"],
			score=quality_report["score"],
			issue_count=len(quality_report["issues"]),
		)
		if not quality_report["passed"]:
			raise DataQualityGateError("phase4_gold_and_report", quality_report)
		scratchpad["gold_output_paths"] = output_paths
		return json.dumps({"gold_output_paths": output_paths})

	@tool("reporter_agent_tool")
	def reporter_agent_tool(goal: str) -> str:
		"""Query Gold tables and generate the final business report."""
		gold_output_paths = scratchpad.get("gold_output_paths")
		if not gold_output_paths:
			raise RuntimeError("Run gold_agent_tool before reporter_agent_tool.")
		if not scratchpad.get("quality_report", {}).get("passed"):
			raise RuntimeError("Gold quality checks must pass before reporting.")
		print(f"Dispatching reporter agent for run {run_id}: {goal}")
		task_description = (
			f"Phase 4 reporting goal: {goal}\n"
			f"Business intent: {business_intent}\n"
			"Inspect and query the Gold tables, answer the business question, and "
			"save an HTML and JSON report with relevant charts."
		)
		report_path = generate_report(
			gold_output_paths,
			business_intent,
			run_id,
			task_description,
		)
		if not report_path:
			raise RuntimeError("Reporter agent did not return a report path.")
		scratchpad["report_path"] = report_path
		return json.dumps({"report_path": report_path})

	return [gold_agent_tool, reporter_agent_tool], scratchpad


def retry_failed_phase(state: PipelineState) -> PipelineState:
	"""Retry only the failed phase, preserving the run ID and completed outputs."""
	failed_phase = state.get("failed_phase")
	run_id = state.get("run_id")
	if not failed_phase or not run_id:
		raise PipelinePhaseError(
			str(failed_phase or "unknown"),
			"There is no failed phase to retry.",
			code="no_failed_phase",
		)

	if failed_phase == "phase1_profile_and_bronze_sttm":
		return run_until_bronze_sttm(
			state.get("uploaded_files", []),
			state.get("business_intent", ""),
			run_id=run_id,
		)
	if failed_phase == "phase2_bronze_and_silver_sttm":
		return run_bronze_to_silver_sttm(state)
	if failed_phase == "phase3_silver_and_gold_sttm":
		return run_silver_to_gold_sttm(state)
	if failed_phase == "phase4_gold_and_report":
		return run_gold_and_report(state)
	raise PipelinePhaseError(
		failed_phase,
		f"Phase '{failed_phase}' cannot be retried by this orchestrator.",
		code="unknown_failed_phase",
	)