"""Human-facing entry point for the IDAMP Streamlit pipeline interface."""

import html
import json
import sys
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from agents.orchestrator import (
	run_bronze_to_silver_sttm,
	run_gold_and_report,
	retry_failed_phase,
	run_silver_to_gold_sttm,
	run_until_bronze_sttm,
)
from core.audit import AuditLogger
from core.cache import clear_cache
from core.config import AUDIT_DIR, LANDING_DIR, STTM_DIR
from core.custom_transformations import validate_custom_rules
from core.data_lineage import build_lineage_graph, filter_lineage_graph, to_graphviz
from core.performance import get_run_metrics, summarize_run_metrics
from core.natural_language_query import (
	answer_gold_question,
	load_gold_tables,
	suggest_chart,
)
from core.sttm_versions import (
	STTM_COLUMNS,
	compare_sttm_rows,
	get_sttm_version,
	list_sttm_versions,
	record_sttm_version,
)
from core.user_workspace import (
	clear_current_user,
	current_username,
	set_current_user,
	user_directory,
)

st.set_page_config(
	page_title="IDAMP - Intent-Driven Agentic Medallion Pipeline",
	page_icon="🏗️",
	layout="wide",
)

PROGRESS_STEPS = [
	("upload", "Upload & Intent"),
	("bronze_sttm", "Bronze STTM Review"),
	("bronze_load", "Bronze Layer Load"),
	("silver_sttm", "Silver STTM Review"),
	("silver_load", "Silver Layer Load"),
	("gold_sttm", "Gold STTM Review"),
	("gold_load", "Gold Layer Load"),
	("report", "Executive Report"),
]

SELECTION_COL = "_selected_for_approval"


def render_progress_banner(
	current_phase: str, state: dict, report_complete: bool
) -> None:
	"""Render the current pipeline phase as a sticky, accessible stepper."""
	status = state.get("status", "") if isinstance(state, dict) else ""
	complete = report_complete or status == "completed"
	step_ids = [step_id for step_id, _ in PROGRESS_STEPS]
	if current_phase not in step_ids:
		current_phase = "report" if complete else "upload"
	current_index = step_ids.index(current_phase)

	step_markup = []
	for index, (step_id, label) in enumerate(PROGRESS_STEPS):
		if complete or index < current_index:
			step_class = "is-complete"
			indicator = "✓"
		elif index == current_index:
			step_class = "is-current"
			indicator = str(index + 1)
		else:
			step_class = "is-upcoming"
			indicator = str(index + 1)
		current_attribute = ' aria-current="step"' if step_class == "is-current" else ""
		step_markup.append(
			f'<div class="progress-step {step_class}"{current_attribute}>'
			f'<span class="step-indicator">{indicator}</span>'
			f'<span class="step-label">{html.escape(label)}</span>'
			"</div>"
		)

	st.markdown(
		dedent(
			f"""
			<style>
			.idamp-progress-banner {{
				position: sticky;
				top: 0;
				z-index: 1000;
				margin: -0.5rem 0 1.25rem;
				padding: 1rem 1.25rem 1.1rem;
				border: 1px solid rgba(255, 255, 255, 0.35);
				border-radius: 0 0 12px 12px;
				background: linear-gradient(112deg, #123c48 0%, #176b72 52%, #329b80 100%);
				box-shadow: 0 10px 28px rgba(14, 55, 63, 0.18);
				color: #f4fffc;
			}}
			.progress-heading {{
				margin: 0 0 0.8rem;
				color: #eafff8;
				font-size: 0.78rem;
				font-weight: 700;
				text-transform: uppercase;
			}}
			.progress-track {{
				display: grid;
				grid-template-columns: repeat(8, minmax(6.7rem, 1fr));
				gap: 0.35rem;
				overflow-x: auto;
				padding: 0.15rem 0 0.35rem;
				scrollbar-color: rgba(232, 255, 249, 0.5) transparent;
			}}
			.progress-step {{
				position: relative;
				display: flex;
				min-width: 6.7rem;
				flex-direction: column;
				align-items: center;
				gap: 0.5rem;
				text-align: center;
			}}
			.progress-step::before {{
				position: absolute;
				top: 1rem;
				left: -55%;
				width: 110%;
				height: 2px;
				background: rgba(226, 248, 241, 0.28);
				content: "";
			}}
			.progress-step:first-child::before {{ display: none; }}
			.step-indicator {{
				position: relative;
				z-index: 1;
				display: grid;
				width: 2rem;
				height: 2rem;
				place-items: center;
				border: 2px solid rgba(239, 255, 250, 0.52);
				border-radius: 50%;
				background: rgba(12, 52, 60, 0.72);
				color: #e6f8f2;
				font-size: 0.8rem;
				font-weight: 700;
			}}
			.is-complete .step-indicator {{
				border-color: #8fe0c0;
				background: #d7fae9;
				color: #155b4c;
			}}
			.is-current .step-indicator {{
				border-color: #fff3c4;
				background: #ffd166;
				box-shadow: 0 0 0 4px rgba(255, 209, 102, 0.2);
				color: #26343a;
			}}
			.step-label {{
				max-width: 8rem;
				color: rgba(239, 255, 250, 0.76);
				font-size: 0.69rem;
				font-weight: 600;
				line-height: 1.25;
			}}
			.is-current .step-label, .is-complete .step-label {{ color: #fff; }}
			@media (max-width: 700px) {{
				.idamp-progress-banner {{ padding-right: 0.8rem; padding-left: 0.8rem; }}
				.progress-track {{ grid-template-columns: repeat(8, 6.7rem); }}
			}}
			</style>
			<div class="idamp-progress-banner" role="region" aria-label="Pipeline progress">
				<div class="progress-heading">Pipeline progress</div>
				<div class="progress-track">{"".join(step_markup)}</div>
			</div>
			"""
		),
		unsafe_allow_html=True,
	)


def _reset_analysis_session() -> None:
	st.session_state.phase = "upload"
	st.session_state.pipeline_state = None
	st.session_state.current_run_id = ""
	st.session_state.report_complete = False


def _prepare_sttm_editor_df(df: pd.DataFrame) -> pd.DataFrame:
	edited_df = df.copy()
	for column in edited_df.columns:
		edited_df[column] = edited_df[column].astype(str)
	edited_df[SELECTION_COL] = True
	return edited_df


def _extract_selected_rows(edited_df: pd.DataFrame) -> pd.DataFrame:
	if SELECTION_COL not in edited_df.columns:
		return edited_df.iloc[0:0].copy()
	selected = edited_df[edited_df[SELECTION_COL].fillna(False).astype(bool)].copy()
	return selected.drop(columns=[SELECTION_COL])


def _current_audit_logs() -> list:
	run_id = st.session_state.get("current_run_id")
	if not run_id:
		pipeline_state = st.session_state.get("pipeline_state") or {}
		run_id = pipeline_state.get("run_id") if isinstance(pipeline_state, dict) else None
	if not run_id:
		return []
	try:
		return AuditLogger(str(run_id)).get_logs()
	except Exception:
		return []


def _current_user_run_history(limit: int = 20) -> list:
	audit_dir = user_directory("audit_logs", AUDIT_DIR)
	if not audit_dir.exists():
		return []
	entries = []
	for log_path in sorted(
		audit_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True
	):
		try:
			with log_path.open("r", encoding="utf-8") as log_file:
				events = [json.loads(line) for line in log_file if line.strip()]
		except (OSError, json.JSONDecodeError):
			continue
		if not events:
			continue
		run_id = events[-1].get("run_id", log_path.stem)
		failed = any(event.get("action") == "phase_fail" for event in events)
		completed = any(
			event.get("action") == "phase_complete"
			and event.get("phase") == "phase4_gold_and_report"
			for event in events
		)
		report_path = next(
			(event.get("report_path") for event in reversed(events) if event.get("report_path")),
			"",
		)
		entries.append(
			{
				"run_id": run_id,
				"started_at": events[0].get("timestamp", ""),
				"last_event": events[-1].get("action", ""),
				"status": "failed" if failed else "completed" if completed else "in progress",
				"report_path": report_path,
			}
		)
	return entries[:limit]


def _render_user_run_history() -> None:
	with st.sidebar:
		st.caption(f"Workspace: {st.session_state.user_id}")
		if st.button("Log out", key="logout_user"):
			clear_current_user()
			_reset_analysis_session()
			st.session_state.user_id = ""
			st.rerun()
		with st.expander("My Run History", expanded=False):
			run_history = _current_user_run_history()
			if run_history:
				st.dataframe(pd.DataFrame(run_history), hide_index=True, use_container_width=True)
			else:
				st.caption("No previous runs in this workspace.")


def _render_performance_panel(run_id: str) -> None:
	metrics = get_run_metrics(run_id)
	if not metrics:
		st.info("No performance metrics were captured for this run.")
		return

	summary = summarize_run_metrics(run_id)
	st.subheader("Run Performance")
	metric_columns = st.columns(4)
	with metric_columns[0]:
		st.metric("LLM API calls", summary["api_calls"])
	with metric_columns[1]:
		peak_memory = summary["peak_memory_mb"]
		st.metric("Peak process memory", f"{peak_memory:.1f} MB" if peak_memory is not None else "Unavailable")
	with metric_columns[2]:
		cost = summary["estimated_cost_usd"]
		st.metric("Estimated LLM cost", f"${cost:.4f}" if cost is not None else "Rates not configured")
	with metric_columns[3]:
		cache = summary["cache"]
		st.metric(
			"Cache hits / misses",
			f"{cache['hits']} / {cache['misses']}",
			delta=f"{cache['lookups']} lookups",
			delta_color="off",
		)
	st.caption(
		f"Tokens: {summary['input_tokens']:,} input / {summary['output_tokens']:,} output. "
		"Memory is process RSS sampled during agents and may include concurrent work in the same Streamlit process."
	)
	if summary["agent_durations_seconds"]:
		duration_frame = pd.DataFrame(
			[
				{"Agent": agent, "Duration (seconds)": round(duration, 3)}
				for agent, duration in summary["agent_durations_seconds"].items()
			]
		)
		st.dataframe(duration_frame, hide_index=True, use_container_width=True)
	st.download_button(
		"Download performance metrics (JSON)",
		data=json.dumps(summary, indent=2, ensure_ascii=False, default=str),
		file_name=f"performance_{run_id}.json",
		mime="application/json",
		key=f"download_metrics_{run_id}",
	)
	with st.expander("Cache controls"):
		st.caption("Cached profiles and STTM rules are scoped to the active user.")
		if st.button("Clear my cache", key=f"clear_cache_{run_id}"):
			removed = clear_cache()
			st.success(f"Cleared {removed} cached result(s).")


def _render_natural_language_chat(pipeline_state: dict) -> None:
	run_id = str(pipeline_state.get("run_id") or "")
	gold_paths = [str(path) for path in pipeline_state.get("gold_output_paths", [])]
	if not run_id or not gold_paths:
		st.info("No Gold tables are available for follow-up questions in this run.")
		return

	file_signature = []
	for gold_path in gold_paths:
		path = Path(gold_path)
		try:
			stat = path.stat()
			file_signature.append((str(path), stat.st_mtime_ns, stat.st_size))
		except OSError:
			file_signature.append((str(path), None, None))
	context_key = (current_username(), run_id, tuple(file_signature))
	if st.session_state.get("nlq_context_key") != context_key:
		try:
			tables, catalog = load_gold_tables(gold_paths)
		except Exception as error:
			st.error(f"Unable to load this run's Gold tables: {error}")
			return
		st.session_state.nlq_context_key = context_key
		st.session_state.nlq_tables = tables
		st.session_state.nlq_catalog = catalog
		st.session_state.nlq_messages = []

	st.subheader("Ask a follow-up question")
	st.caption(
		"Questions use only the Gold tables from this run. Results are limited to 500 rows."
	)
	clear_col, context_col = st.columns([1, 5])
	with clear_col:
		if st.button("Clear chat", key=f"nlq_clear_{run_id}"):
			st.session_state.nlq_messages = []
			st.rerun()
	with context_col:
		st.caption("Available tables: " + ", ".join(st.session_state.nlq_tables))

	for message in st.session_state.nlq_messages:
		with st.chat_message(message["role"]):
			st.write(message["content"])
			if message.get("dataframe") is not None:
				dataframe = message["dataframe"]
				st.dataframe(dataframe, hide_index=True, use_container_width=True)
				chart = message.get("chart")
				if chart:
					chart_function = px.line if chart["type"] == "line" else px.bar
					figure = chart_function(
						dataframe,
						x=chart["x"],
						y=chart["y"],
						title="Query result trend" if chart["type"] == "line" else "Query result comparison",
					)
					st.plotly_chart(figure, use_container_width=True)
				if message.get("truncated"):
					st.caption("Showing the first 500 rows of the query result.")
				with st.expander("Generated SQL"):
					st.code(message["sql"], language="sql")

	question = st.chat_input("Ask about products, trends, regions, or other Gold data")
	if question:
		messages = st.session_state.nlq_messages
		messages.append({"role": "user", "content": question})
		try:
			result = answer_gold_question(
				question,
				st.session_state.nlq_tables,
				st.session_state.nlq_catalog,
				run_id,
				history=messages[:-1],
			)
			dataframe = result["dataframe"]
			row_count = result["row_count"]
			answer = f"Found {row_count} result row(s)." if row_count else "No rows matched that question."
			if result["truncated"]:
				answer += " The displayed results are limited to 500 rows."
			messages.append(
				{
					"role": "assistant",
					"content": answer,
					"sql": result["sql"],
					"dataframe": dataframe,
					"chart": suggest_chart(question, dataframe),
					"truncated": result["truncated"],
					"sample_rows": dataframe.head(3).to_dict(orient="records"),
				}
			)
		except Exception as error:
			messages.append(
				{
					"role": "assistant",
					"content": f"I couldn't answer that query: {error}",
				}
			)
		st.rerun()


def _render_lineage_diagram(pipeline_state: dict) -> None:
	st.subheader("Data Lineage")
	if pipeline_state.get("lineage_error"):
		st.warning(f"The latest lineage snapshot could not be saved: {pipeline_state['lineage_error']}")
	lineage_path = pipeline_state.get("lineage_path")
	graph = None
	if lineage_path and Path(lineage_path).is_file():
		try:
			graph = json.loads(Path(lineage_path).read_text(encoding="utf-8"))
		except (OSError, json.JSONDecodeError):
			graph = None
	if graph is None:
		graph = build_lineage_graph(pipeline_state)

	nodes = graph.get("nodes", [])
	if not nodes:
		st.info("Lineage will appear as pipeline inputs and outputs become available.")
		return
	filter_col, inspect_col = st.columns([1, 2])
	with filter_col:
		layer_filter = st.selectbox(
			"Focus lineage stage",
			["All", "Source", "Bronze", "Silver", "Gold", "Report"],
			key=f"lineage_filter_{pipeline_state.get('run_id', 'run')}",
		)
	filtered_graph = filter_lineage_graph(graph, layer_filter)
	filtered_nodes = filtered_graph.get("nodes", [])
	if not filtered_nodes:
		st.info("No lineage nodes are available for this stage yet.")
		return
	labels = {node["id"]: node["label"] for node in filtered_nodes}
	with inspect_col:
		selected_node = st.selectbox(
			"Inspect element",
			list(labels),
			format_func=lambda node_id: labels[node_id],
			key=f"lineage_node_{pipeline_state.get('run_id', 'run')}_{layer_filter}",
		)
	st.graphviz_chart(
		to_graphviz(filtered_graph, selected_node),
		use_container_width=True,
	)
	selected_details = next(
		(node for node in filtered_nodes if node["id"] == selected_node), None
	)
	if selected_details:
		st.caption(f"Selected: {selected_details['kind'].replace('_', ' ').title()}")
		st.code(selected_details.get("details", selected_details["label"]))
	if graph.get("warnings"):
		for warning in graph["warnings"]:
			st.warning(warning)
	download_col, _ = st.columns([1, 5])
	with download_col:
		st.download_button(
			"Download Graphviz DOT",
			data=to_graphviz(graph),
			file_name=f"lineage_{pipeline_state.get('run_id', 'run')}.dot",
			mime="text/vnd.graphviz",
			key=f"download_lineage_{pipeline_state.get('run_id', 'run')}",
		)


def _format_audit_entry(entry: dict) -> str:
	timestamp = entry.get("timestamp") or "Time unknown"
	agent = entry.get("agent") or "System"
	action = entry.get("action") or "Event"
	rationale = entry.get("rationale") or entry.get("details") or ""
	phase = entry.get("phase") or ""
	status = entry.get("status") or "neutral"
	parts = [f"{timestamp} | {agent} | {action}"]
	if phase:
		parts.append(f"Phase: {phase}")
	if status:
		parts.append(f"Status: {status}")
	if rationale:
		parts.append(f"Rationale: {rationale}")
	return " · ".join(parts)


def _status_class(entry: dict) -> str:
	status = str(entry.get("status", "")).strip().lower()
	if status in {"failed", "failure", "error"}:
		return "is-failed"
	if status in {"success", "completed", "complete"}:
		return "is-success"
	if status in {"running", "in_progress", "in progress", "started", "progress"}:
		return "is-progress"
	return "is-neutral"


def _render_audit_card(entry: dict, latest: bool = False) -> str:
	timestamp = html.escape(str(entry.get("timestamp") or "Time unknown"))
	agent = html.escape(str(entry.get("agent") or "System"))
	action = html.escape(str(entry.get("action") or "Event"))
	rationale = html.escape(
		str(entry.get("rationale") or entry.get("details") or "No additional details")
	)
	phase = html.escape(str(entry.get("phase") or "Unspecified phase"))
	status = html.escape(str(entry.get("status") or "neutral"))
	latest_class = " is-latest" if latest else ""
	status_class = _status_class(entry)
	return dedent(
		f"""
		<div class="audit-card {status_class}{latest_class}">
			<div class="audit-card-top">
				<span class="audit-action">{action}</span>
				<span class="audit-status {status_class}">{status}</span>
			</div>
			<div class="audit-meta"><span>{agent}</span><span>{phase}</span><time>{timestamp}</time></div>
			<div class="audit-rationale">{rationale}</div>
		</div>
		"""
	).strip()


def render_audit_trail_panel() -> None:
	st.markdown(
		dedent(
			"""
			<style>
			.audit-trail-title { margin: 0 0 .75rem; color: #173d47; font-size: 1.15rem; font-weight: 750; }
			.audit-card {
				margin: .55rem 0; padding: .85rem 1rem; border: 1px solid #dce7e8;
				border-left: 4px solid #8ea5a8; border-radius: 8px; background: #fff;
				box-shadow: 0 4px 14px rgba(25, 55, 65, .05);
			}
			.audit-card.is-latest { padding: 1.1rem 1.15rem; border-left-width: 6px; background: linear-gradient(110deg, #f0fbf8, #fff); }
			.audit-card.is-failed { border-left-color: #cf4b4b; }
			.audit-card.is-success { border-left-color: #24916f; }
			.audit-card.is-progress { border-left-color: #d99422; }
			.audit-card.is-neutral { border-left-color: #82999d; }
			.audit-card-top, .audit-meta { display: flex; align-items: center; justify-content: space-between; gap: .6rem; }
			.audit-action { color: #193f48; font-size: .96rem; font-weight: 700; }
			.audit-status { padding: .15rem .5rem; border-radius: 999px; background: #edf2f2; color: #52666a; font-size: .7rem; font-weight: 700; text-transform: uppercase; }
			.audit-status.is-failed { background: #fdecec; color: #9b3030; }
			.audit-status.is-success { background: #e6f6ef; color: #176b50; }
			.audit-status.is-progress { background: #fff4db; color: #8a5a0a; }
			.audit-meta { justify-content: flex-start; flex-wrap: wrap; margin-top: .45rem; color: #607479; font-size: .74rem; }
			.audit-meta span:not(:last-child)::after { margin-left: .6rem; color: #a2b2b4; content: "•"; }
			.audit-meta time { margin-left: auto; }
			.audit-rationale { margin-top: .55rem; color: #33494f; font-size: .84rem; line-height: 1.5; overflow-wrap: anywhere; }
			.audit-history { max-height: 24rem; overflow-y: auto; padding-right: .35rem; }
			@media (max-width: 620px) { .audit-meta time { width: 100%; margin-left: 0; } }
			</style>
			"""
		),
		unsafe_allow_html=True,
	)
	st.markdown('<div class="audit-trail-title">Audit Trail</div>', unsafe_allow_html=True)
	logs = _current_audit_logs()
	if not logs:
		st.info("No audit events are available for this run yet.")
		return

	st.markdown(_render_audit_card(logs[-1], latest=True), unsafe_allow_html=True)
	previous_logs = list(reversed(logs[:-1]))
	if previous_logs:
		previous_markup = "".join(_render_audit_card(entry) for entry in previous_logs)
		st.markdown(
			f'<div class="audit-history">{previous_markup}</div>',
			unsafe_allow_html=True,
		)
	run_id = st.session_state.get("current_run_id") or "pipeline-run"
	st.download_button(
		label="Download audit log (JSON)",
		data=json.dumps(logs, indent=2, ensure_ascii=False, default=str),
		file_name=f"audit_{run_id}.json",
		mime="application/json",
		key=f"download_audit_{run_id}",
	)


def _render_quality_report(state: dict) -> None:
	quality_report = state.get("quality_report") if isinstance(state, dict) else None
	if not quality_report:
		return

	score = quality_report.get("score", 0.0)
	minimum_score = quality_report.get("minimum_score", 0.0)
	if quality_report.get("passed"):
		st.success(
			f"{quality_report.get('layer', 'Layer').title()} data quality passed "
			f"with score {score:.0%} (minimum {minimum_score:.0%})."
		)
	else:
		st.error(
			f"{quality_report.get('layer', 'Layer').title()} data quality failed "
			f"with score {score:.0%} (minimum {minimum_score:.0%})."
		)
	for issue in quality_report.get("issues", []):
		message = f"{issue.get('dataset', 'Dataset')}: {issue.get('message', 'Quality issue')}"
		if issue.get("severity") == "error":
			st.error(message)
		else:
			st.warning(message)


def _render_failure_panel(state: dict) -> None:
	failed_phase = state.get("failed_phase") or "unknown phase"
	st.error(f"Pipeline failed during {failed_phase}: {state.get('error', 'Unknown error')}")
	error_details = state.get("error_details")
	if error_details:
		with st.expander("Technical error details", expanded=False):
			st.json(error_details)
	_render_quality_report(state)

	retry_col, restart_col = st.columns(2)
	with retry_col:
		if st.button("🔁 Retry Failed Step", type="primary", key="retry_failed_step"):
			try:
				with st.spinner(f"Retrying {failed_phase}..."):
					updated_state = retry_failed_phase(state)
				st.session_state.pipeline_state = updated_state
				st.session_state.current_run_id = updated_state.get("run_id", "")
				if updated_state.get("status") == "failed":
					st.rerun()
				phase_by_status = {
					"awaiting_bronze_sttm_approval": "bronze_sttm",
					"awaiting_silver_sttm_approval": "silver_sttm",
					"awaiting_gold_sttm_approval": "gold_sttm",
					"completed": "report",
				}
				st.session_state.phase = phase_by_status.get(
					updated_state.get("status"), "upload"
				)
				st.session_state.report_complete = updated_state.get("status") == "completed"
				st.rerun()
			except Exception as error:
				st.error(f"Retry failed: {error}")
	with restart_col:
		if st.button("↩️ Start Over", key="restart_after_failure"):
			_reset_analysis_session()
			st.rerun()


def _render_sttm_review(
	heading: str,
	description: str,
	path_key: str,
	approve_label: str,
	button_key: str,
	layer: str,
	next_phase: str,
	transition,
) -> None:
	st.header(heading)
	st.info(description)
	pipeline_state = st.session_state.get("pipeline_state")
	if isinstance(pipeline_state, dict):
		if pipeline_state.get("status") == "failed":
			_render_failure_panel(pipeline_state)
			return
		_render_quality_report(pipeline_state)
	sttm_path = pipeline_state.get(path_key) if isinstance(pipeline_state, dict) else None
	if not sttm_path:
		st.error("The STTM path is missing from the pipeline state.")
		if st.button("↩️ Restart Workflow", key=f"{button_key}_restart"):
			_reset_analysis_session()
			st.rerun()
		return

	path = Path(sttm_path)
	try:
		sttm_df = pd.read_csv(path)
		if sttm_df.empty:
			st.error("This STTM file contains no transformations. Restart the workflow to try again.")
			if st.button("↩️ Restart Workflow", key=f"{button_key}_empty_restart"):
				_reset_analysis_session()
				st.rerun()
			return
		run_id = str(
			pipeline_state.get("run_id")
			or st.session_state.get("current_run_id")
			or "unknown"
		)
		current_rows = sttm_df.fillna("").astype(str).to_dict(orient="records")
		version_logger = AuditLogger(run_id)
		if not list_sttm_versions(layer):
			record_sttm_version(layer, current_rows, run_id, "baseline", version_logger)
		versions = list_sttm_versions(layer)
		with st.expander("Rule version history", expanded=False):
			st.dataframe(
				pd.DataFrame(
					[
						{
							"Version": version["version_id"],
							"Saved at": version["created_at"],
							"Change": version["reason"],
							"Run": version["run_id"],
							"Previous": version.get("previous_version_id") or "Initial",
							"Restored from": version.get("restored_from") or "",
							"Rules": version["row_count"],
						}
						for version in versions
					]
				),
				hide_index=True,
				use_container_width=True,
			)
			version_ids = [version["version_id"] for version in versions]
			selected_version_id = st.selectbox(
				"Compare current rules with",
				version_ids,
				format_func=lambda version_id: next(
					f"{version_id} ({version['reason']})"
					for version in versions
					if version["version_id"] == version_id
				),
				key=f"{button_key}_history_version",
			)
			selected_version = get_sttm_version(layer, selected_version_id)
			version_diff = compare_sttm_rows(
				current_rows,
				selected_version["rows"],
				"Current STTM",
				selected_version_id,
			)
			if version_diff:
				st.code("\n".join(version_diff), language="diff")
			else:
				st.info("The selected version matches the current STTM rules.")
			if st.button(
				"Restore selected version",
				key=f"{button_key}_restore_version",
				disabled=not version_diff,
			):
				restored_rows = selected_version["rows"]
				record_sttm_version(
					layer,
					restored_rows,
					run_id,
					"restored",
					version_logger,
					restored_from=selected_version_id,
				)
				pd.DataFrame(restored_rows, columns=STTM_COLUMNS).to_csv(path, index=False)
				st.session_state.pop(f"{button_key}_editor", None)
				st.success(f"Restored rules from version {selected_version_id}.")
				st.rerun()

		st.caption(f"Total transformations: {len(sttm_df)}")
		st.caption(
			"Custom rules: expression (safe arithmetic/functions), regex_replace "
			"(JSON pattern/replacement), or lookup (JSON mapping/default). Arbitrary Python is not executed."
		)
		transformation_options = [
			"copy",
			"rename",
			"cast",
			"dropna",
			"fillna",
			"expression",
			"regex_replace",
			"lookup",
		]
		if "transformation_type" in sttm_df.columns:
			transformation_options.extend(
				value
				for value in sttm_df["transformation_type"].dropna().astype(str).unique()
				if value not in transformation_options
			)
		edited_df = st.data_editor(
			_prepare_sttm_editor_df(sttm_df),
			key=f"{button_key}_editor",
			hide_index=True,
			use_container_width=True,
			disabled=[
				column
				for column in sttm_df.columns
				if column not in {"transformation_type", "transformation_logic"}
			],
			column_config={
				SELECTION_COL: st.column_config.CheckboxColumn(
					"Approve",
					help="Select the transformation rules to approve for this layer.",
					default=True,
				),
				"transformation_type": st.column_config.SelectboxColumn(
					"Transformation type",
					options=transformation_options,
				),
				"transformation_logic": st.column_config.TextColumn(
					"Transformation logic",
					help="For custom rules, use a safe expression, regex JSON, or lookup JSON.",
				),
			},
		)
		if st.button(approve_label, type="primary", key=f"{button_key}_approve"):
			selected_rows = _extract_selected_rows(edited_df)
			if selected_rows.empty:
				st.error("Select at least one transformation before approving.")
				return
			available_columns = set(sttm_df.get("source_column", pd.Series(dtype=str)).dropna())
			available_columns.update(sttm_df.get("target_column", pd.Series(dtype=str)).dropna())
			validation_errors = validate_custom_rules(
				selected_rows.to_dict(orient="records"),
				available_columns,
			)
			if validation_errors:
				st.error("Custom transformation validation failed:")
				for validation_error in validation_errors:
					st.warning(validation_error)
				return
			try:
				record_sttm_version(
					layer,
					selected_rows.to_dict(orient="records"),
					run_id,
					"approved",
					version_logger,
				)
				selected_rows.to_csv(path, index=False)
				with st.spinner("Applying approved transformations..."):
					updated_state = transition(pipeline_state)
				st.session_state.pipeline_state = updated_state
				st.session_state.current_run_id = updated_state.get("run_id", "")
				if updated_state.get("status") == "failed":
					_render_failure_panel(updated_state)
				else:
					st.session_state.phase = next_phase
					st.session_state.report_complete = next_phase == "report"
					st.rerun()
			except Exception as error:
				st.error(f"Unable to apply the approved STTM: {error}")
	except Exception as error:
		st.error(f"Unable to read the STTM file: {error}")
		if st.button("↩️ Restart Workflow", key=f"{button_key}_corrupt_restart"):
			_reset_analysis_session()
			st.rerun()


st.title("Intent-Driven Agentic Medallion Workflow")
st.markdown(
	"Upload retail data and describe the business question; specialist agents "
	"will profile, transform, and report on it with approval gates between layers."
)

for session_key, default_value in (
	("user_id", ""),
	("pipeline_state", None),
	("phase", "upload"),
	("current_run_id", ""),
	("report_complete", False),
):
	if session_key not in st.session_state:
		st.session_state[session_key] = default_value

if not st.session_state.user_id:
	st.subheader("Sign in to your pipeline workspace")
	st.caption("A username selects an isolated workspace; it does not verify identity.")
	with st.form("workspace_login_form"):
		entered_username = st.text_input("Username")
		submitted = st.form_submit_button("Continue")
	if submitted:
		try:
			st.session_state.user_id = set_current_user(entered_username)
			st.rerun()
		except ValueError as error:
			st.error(str(error))
	st.stop()

try:
	st.session_state.user_id = set_current_user(st.session_state.user_id)
except ValueError as error:
	clear_current_user()
	st.session_state.user_id = ""
	st.error(str(error))
	st.stop()

_render_user_run_history()

pipeline_state = st.session_state.get("pipeline_state")
if isinstance(pipeline_state, dict) and pipeline_state.get("run_id"):
	st.session_state.current_run_id = pipeline_state["run_id"]
	if pipeline_state.get("status") == "completed":
		st.session_state.report_complete = True

render_progress_banner(
	st.session_state.phase,
	pipeline_state or {},
	st.session_state.report_complete,
)

main_col, audit_col = st.columns([3.4, 1.4], gap="large")

with audit_col:
	render_audit_trail_panel()

with main_col:
	current_pipeline_state = st.session_state.get("pipeline_state")
	if isinstance(current_pipeline_state, dict) and current_pipeline_state.get("status") == "failed":
		_render_failure_panel(current_pipeline_state)
	elif st.session_state.phase == "upload":
		st.header("📤 Phase 1: Upload Data & Define Intent")
		uploaded_files = st.file_uploader(
			"Upload CSV or Excel files",
			type=["csv", "xlsx"],
			accept_multiple_files=True,
			help="Each worksheet in an XLSX workbook is profiled and ingested as a separate dataset.",
		)
		business_intent = st.text_area(
			"Business question or intent",
			placeholder="Example: Which product categories generated the most revenue by region?",
			height=120,
		)
		start_workflow = st.button(
			"🚀 Start Workflow",
			type="primary",
			disabled=not uploaded_files or not business_intent.strip(),
			use_container_width=True,
		)

		if start_workflow:
			landing_dir = user_directory("landing", LANDING_DIR)
			landing_dir.mkdir(parents=True, exist_ok=True)
			saved_paths = []
			for file_index, uploaded_file in enumerate(uploaded_files, start=1):
				original_name = Path(uploaded_file.name).name or f"upload_{file_index}.csv"
				file_path = landing_dir / original_name
				counter = 1
				while file_path.exists() or str(file_path) in saved_paths:
					file_path = landing_dir / (
						f"{Path(original_name).stem}_{counter}{Path(original_name).suffix}"
					)
					counter += 1
				file_path.write_bytes(uploaded_file.getvalue())
				saved_paths.append(str(file_path))

			try:
				with st.spinner("🔍 Profiling data..."):
					new_state = run_until_bronze_sttm(
						saved_paths,
						business_intent.strip(),
					)
				st.session_state.pipeline_state = new_state
				st.session_state.current_run_id = new_state.get("run_id", "")
				if new_state.get("status") == "failed":
					st.rerun()
				else:
					st.session_state.phase = "bronze_sttm"
					st.rerun()
			except Exception as error:
				st.error(f"Unable to start the workflow: {error}")
	elif st.session_state.phase == "bronze_sttm":
		_render_sttm_review(
			"🥉 Phase 2: Review Bronze Layer STTM",
			"Bronze is the raw ingestion layer. It preserves source values with "
			"minimal changes and adds load metadata for traceability.",
			"sttm_bronze_path",
			"✅ Approve & Continue",
			"bronze_sttm",
			"bronze",
			"silver_sttm",
			run_bronze_to_silver_sttm,
		)
	elif st.session_state.phase == "silver_sttm":
		_render_sttm_review(
			"🥈 Phase 3: Review Silver Layer STTM",
			"Silver cleanses Bronze data by handling nulls, removing duplicates, "
			"standardizing types, and normalizing values.",
			"sttm_silver_path",
			"✅ Approve & Continue",
			"silver_sttm",
			"silver",
			"gold_sttm",
			run_silver_to_gold_sttm,
		)
	elif st.session_state.phase == "gold_sttm":
		_render_sttm_review(
			"🥇 Phase 4: Review Gold Layer STTM",
			"Gold joins and aggregates Silver data into analytics-ready tables "
			"for business reporting.",
			"sttm_gold_path",
			"✅ Approve & Execute",
			"gold_sttm",
			"gold",
			"report",
			run_gold_and_report,
		)
	elif st.session_state.phase == "report":
		st.header("📊 Phase 5: Executive Report")
		st.session_state.report_complete = True
		pipeline_state = st.session_state.get("pipeline_state") or {}
		_render_quality_report(pipeline_state)
		run_id = pipeline_state.get("run_id") or st.session_state.get("current_run_id")
		if run_id:
			_render_performance_panel(run_id)
		report_path = pipeline_state.get("report_path")
		if report_path and Path(report_path).is_file():
			try:
				report_html = Path(report_path).read_text(encoding="utf-8")
				st.components.v1.html(report_html, height=2000, scrolling=True)
				download_col, restart_col = st.columns(2)
				with download_col:
					st.download_button(
						"📥 Download Report",
						data=report_html,
						file_name=Path(report_path).name,
						mime="text/html",
						key="download_executive_report",
						use_container_width=True,
					)
				with restart_col:
					if st.button(
						"🔄 Start New Analysis",
						key="start_new_analysis",
						use_container_width=True,
					):
						_reset_analysis_session()
						st.rerun()
			except Exception as error:
				st.error(f"Unable to open the report file: {error}")
		else:
			st.error(
				"Report file not found. Please check the pipeline execution."
			)
		_render_lineage_diagram(pipeline_state)
		_render_natural_language_chat(pipeline_state)