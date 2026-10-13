"""Run the Agentic Medallion Pipeline without the Streamlit interface."""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run the Agentic Medallion Pipeline on CSV or Excel inputs."
	)
	parser.add_argument(
		"--input",
		dest="input_files",
		nargs="+",
		type=Path,
		required=True,
		help="One or more CSV or XLSX input paths.",
	)
	parser.add_argument(
		"--business-intent",
		required=True,
		help="Business question for the reporting agent.",
	)
	parser.add_argument(
		"--auto-approve",
		action="store_true",
		help="Automatically approve generated STTM rules and run all phases unattended.",
	)
	parser.add_argument(
		"--log-file",
		type=Path,
		help="Log destination (defaults to audit_logs/cli_runs with a timestamped name).",
	)
	parser.add_argument(
		"--log-level",
		choices=("DEBUG", "INFO", "WARNING", "ERROR"),
		default="INFO",
		help="Console and file logging threshold.",
	)
	return parser.parse_args(argv)


def _configure_logging(log_file: Optional[Path], log_level: str) -> tuple[logging.Logger, Path]:
	if log_file is None:
		timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
		log_file = PROJECT_ROOT / "audit_logs" / "cli_runs" / f"pipeline_{timestamp}.log"
	elif not log_file.is_absolute():
		log_file = PROJECT_ROOT / log_file
	log_file.parent.mkdir(parents=True, exist_ok=True)

	logger = logging.getLogger("idamp.cli")
	logger.setLevel(getattr(logging, log_level))
	logger.handlers.clear()
	logger.propagate = False
	formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(formatter)
	file_handler = logging.FileHandler(log_file, encoding="utf-8")
	file_handler.setFormatter(formatter)
	logger.addHandler(console_handler)
	logger.addHandler(file_handler)
	return logger, log_file


def _log_state_failure(logger: logging.Logger, phase: str, state: dict) -> bool:
	if state.get("status") != "failed":
		return False
	logger.error("Pipeline failed during %s: %s", phase, state.get("error", "Unknown error"))
	return True


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = _parse_args(argv)
	logger, log_file = _configure_logging(args.log_file, args.log_level)
	logger.info("Starting pipeline; log file: %s", log_file)

	try:
		for input_file in args.input_files:
			if not input_file.is_file():
				raise FileNotFoundError(f"Input file does not exist: {input_file}")
			if input_file.suffix.lower() not in {".csv", ".xlsx"}:
				raise ValueError(f"Unsupported input format: {input_file.suffix}")

		from agents.orchestrator import (
			run_bronze_to_silver_sttm,
			run_gold_and_report,
			run_silver_to_gold_sttm,
			run_until_bronze_sttm,
		)

		state = run_until_bronze_sttm(
			[str(path.resolve()) for path in args.input_files],
			args.business_intent,
		)
		if _log_state_failure(logger, "profiling and Bronze STTM generation", state):
			return 1
		logger.info(
			"Bronze STTM ready for approval: %s",
			state.get("sttm_bronze_path", "not provided"),
		)

		if not args.auto_approve:
			logger.warning(
				"Stopping at the Bronze STTM approval gate. Re-run with --auto-approve "
				"only if unattended approval is acceptable."
			)
			logger.info("Pipeline state: %s", json.dumps(state, default=str))
			return 2

		for phase_name, phase_runner in (
			("Bronze ingestion and Silver STTM", run_bronze_to_silver_sttm),
			("Silver cleansing and Gold STTM", run_silver_to_gold_sttm),
			("Gold aggregation and report", run_gold_and_report),
		):
			logger.info("Starting %s", phase_name)
			state = phase_runner(state)
			if _log_state_failure(logger, phase_name, state):
				return 1

		summary = {
			"run_id": state.get("run_id"),
			"status": state.get("status"),
			"report_path": state.get("report_path"),
			"gold_output_paths": state.get("gold_output_paths", []),
		}
		logger.info("Pipeline completed: %s", json.dumps(summary, ensure_ascii=False))
		return 0
	except KeyboardInterrupt:
		logger.warning("Pipeline interrupted by user or scheduler.")
		return 130
	except Exception:
		logger.exception("Pipeline terminated with an unexpected error.")
		return 1
	finally:
		for handler in logger.handlers:
			handler.flush()
			handler.close()
		logger.handlers.clear()


if __name__ == "__main__":
	sys.exit(main())