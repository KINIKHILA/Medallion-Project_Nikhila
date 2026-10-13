"""FastAPI endpoints for uploads, asynchronous pipeline runs, status, and reports."""

import logging
import re
import traceback
import uuid
from pathlib import Path
from typing import List, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agents.orchestrator import (
	run_bronze_to_silver_sttm,
	run_gold_and_report,
	run_silver_to_gold_sttm,
	run_until_bronze_sttm,
)
from core.config import LANDING_DIR, REPORTS_DIR
from core.run_store import get_run_state, save_run_state
from core.user_workspace import normalize_username, use_user_workspace, user_directory

logger = logging.getLogger("idamp.api")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(
	title="Agentic Medallion Pipeline API",
	description=(
		"Upload CSV/XLSX data, launch pipeline runs, approve generated STTM phases, "
		"check run status, and download reports."
	),
	version="1.0.0",
)


class StartRunRequest(BaseModel):
	input_files: List[str] = Field(min_length=1)
	business_intent: str = Field(min_length=1)
	auto_approve: bool = False


class ApproveRunRequest(BaseModel):
	phase: Literal["bronze", "silver", "gold"]


def _request_username(x_username: str = Header(..., alias="X-Username")) -> str:
	try:
		return normalize_username(x_username)
	except ValueError as error:
		raise HTTPException(status_code=400, detail=str(error)) from error


def _user_input_paths(paths: List[str]) -> List[str]:
	landing_root = user_directory("landing", LANDING_DIR).resolve()
	validated = []
	for raw_path in paths:
		path = Path(raw_path).expanduser().resolve()
		try:
			path.relative_to(landing_root)
		except ValueError as error:
			raise HTTPException(
				status_code=400,
				detail="Input files must be uploaded to this user's landing folder.",
			) from error
		if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx"}:
			raise HTTPException(
				status_code=400,
				detail=f"Input must be an existing CSV or XLSX file: {path.name}",
			)
		validated.append(str(path))
	return validated


def _save_background_failure(run_id: str, phase: str, error: Exception) -> None:
	state = get_run_state(run_id) or {"run_id": run_id}
	state.update(
		{
			"status": "failed",
			"failed_phase": phase,
			"error": str(error),
			"error_details": {
				"phase": phase,
				"exception_type": type(error).__name__,
				"message": str(error),
				"traceback": "".join(
					traceback.format_exception(type(error), error, error.__traceback__)
				),
			},
		}
	)
	save_run_state(state)
	logger.exception("Background pipeline run %s failed during %s", run_id, phase)


def _run_pipeline_task(
	username: str,
	run_id: str,
	input_files: List[str],
	business_intent: str,
	auto_approve: bool,
) -> None:
	with use_user_workspace(username):
		try:
			state = run_until_bronze_sttm(input_files, business_intent, run_id=run_id)
			if state.get("status") == "failed" or not auto_approve:
				return
			for phase_name, phase_runner in (
				("phase2_bronze_and_silver_sttm", run_bronze_to_silver_sttm),
				("phase3_silver_and_gold_sttm", run_silver_to_gold_sttm),
				("phase4_gold_and_report", run_gold_and_report),
			):
				state = phase_runner(state)
				if state.get("status") == "failed":
					return
		except Exception as error:
			_save_background_failure(run_id, "api_background_run", error)


def _approve_phase_task(username: str, run_id: str, phase: str) -> None:
	with use_user_workspace(username):
		state = get_run_state(run_id)
		if state is None:
			return
		phase_runner = {
			"bronze": run_bronze_to_silver_sttm,
			"silver": run_silver_to_gold_sttm,
			"gold": run_gold_and_report,
		}[phase]
		try:
			phase_runner(state)
		except Exception as error:
			_save_background_failure(run_id, f"api_approval_{phase}", error)


@app.get("/api/health")
def health_check() -> dict:
	return {"status": "ok"}


@app.post("/api/uploads", status_code=201)
async def upload_files(
	files: List[UploadFile] = File(...),
	username: str = Depends(_request_username),
) -> dict:
	if not files:
		raise HTTPException(status_code=400, detail="Upload at least one file.")

	saved_paths = []
	with use_user_workspace(username):
		landing_dir = user_directory("landing", LANDING_DIR)
		landing_dir.mkdir(parents=True, exist_ok=True)
		for upload in files:
			original_name = Path(re.split(r"[\\/]", upload.filename or "")[ -1]).name
			if Path(original_name).suffix.lower() not in {".csv", ".xlsx"}:
				raise HTTPException(status_code=415, detail="Only CSV and XLSX files are supported.")
			safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", original_name)
			if not safe_name or safe_name in {".", ".."}:
				raise HTTPException(status_code=400, detail="Invalid uploaded filename.")
			destination = landing_dir / safe_name
			base = destination.stem
			suffix = destination.suffix
			index = 2
			while destination.exists():
				destination = landing_dir / f"{base}_{index}{suffix}"
				index += 1

			total_bytes = 0
			try:
				with destination.open("wb") as output_file:
					while chunk := await upload.read(1024 * 1024):
						total_bytes += len(chunk)
						if total_bytes > MAX_UPLOAD_BYTES:
							raise HTTPException(
								status_code=413,
								detail=f"{safe_name} exceeds the 50 MB upload limit.",
							)
						output_file.write(chunk)
			except Exception:
				destination.unlink(missing_ok=True)
				raise
			finally:
				await upload.close()
			saved_paths.append(str(destination))
	return {"username": username, "files": saved_paths}


@app.post("/api/runs", status_code=202)
def start_run(
	request: StartRunRequest,
	background_tasks: BackgroundTasks,
	username: str = Depends(_request_username),
) -> dict:
	with use_user_workspace(username):
		input_files = _user_input_paths(request.input_files)
		run_id = str(uuid.uuid4())
		initial_state = {
			"run_id": run_id,
			"status": "queued",
			"uploaded_files": input_files,
			"business_intent": request.business_intent,
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
		save_run_state(initial_state)
	background_tasks.add_task(
		_run_pipeline_task,
		username,
		run_id,
		input_files,
		request.business_intent,
		request.auto_approve,
	)
	return {
		"run_id": run_id,
		"status": "queued",
		"status_url": f"/api/runs/{run_id}",
		"auto_approve": request.auto_approve,
	}


@app.post("/api/runs/{run_id}/approve", status_code=202)
def approve_phase(
	run_id: str,
	request: ApproveRunRequest,
	background_tasks: BackgroundTasks,
	username: str = Depends(_request_username),
) -> dict:
	with use_user_workspace(username):
		state = get_run_state(run_id)
	if state is None:
		raise HTTPException(status_code=404, detail="Run not found for this user.")
	expected_status = {
		"bronze": "awaiting_bronze_sttm_approval",
		"silver": "awaiting_silver_sttm_approval",
		"gold": "awaiting_gold_sttm_approval",
	}[request.phase]
	if state.get("status") != expected_status:
		raise HTTPException(
			status_code=409,
			detail=f"Run is not waiting for {request.phase} STTM approval.",
		)
	background_tasks.add_task(_approve_phase_task, username, run_id, request.phase)
	return {"run_id": run_id, "status": "approval_accepted", "phase": request.phase}


@app.get("/api/runs/{run_id}")
def get_run_status(
	run_id: str,
	username: str = Depends(_request_username),
) -> dict:
	with use_user_workspace(username):
		state = get_run_state(run_id)
	if state is None:
		raise HTTPException(status_code=404, detail="Run not found for this user.")
	return state


@app.get("/api/runs/{run_id}/report")
def download_report(
	run_id: str,
	username: str = Depends(_request_username),
) -> FileResponse:
	with use_user_workspace(username):
		state = get_run_state(run_id)
		reports_dir = user_directory("reports", REPORTS_DIR).resolve()
	if state is None:
		raise HTTPException(status_code=404, detail="Run not found for this user.")
	report_path = state.get("report_path")
	if not report_path:
		raise HTTPException(status_code=409, detail="This run does not have a report yet.")
	path = Path(report_path).resolve()
	try:
		path.relative_to(reports_dir)
	except ValueError as error:
		raise HTTPException(status_code=404, detail="Report not found for this user.") from error
	if not path.is_file():
		raise HTTPException(status_code=404, detail="Report file is missing.")
	return FileResponse(path, media_type="text/html", filename=path.name)