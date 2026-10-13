import pytest

from core.exceptions import PipelinePhaseError, TransientPipelineError
from core.retry import is_transient_error, retry_transient


def test_retries_transient_failure_with_backoff_callback():
	calls = []
	retries = []
	delays = []

	def operation():
		calls.append(True)
		if len(calls) == 1:
			raise TimeoutError("model request timed out")
		return "complete"

	result = retry_transient(
		operation,
		max_attempts=3,
		base_delay_seconds=0.1,
		on_retry=lambda error, attempt: retries.append((str(error), attempt)),
		sleep=delays.append,
	)

	assert result == "complete"
	assert len(calls) == 2
	assert retries == [("model request timed out", 1)]
	assert delays == [0.1]


def test_does_not_retry_non_transient_error():
	calls = []

	def operation():
		calls.append(True)
		raise ValueError("invalid STTM")

	with pytest.raises(ValueError, match="invalid STTM"):
		retry_transient(operation, sleep=lambda delay: None)
	assert len(calls) == 1


def test_custom_exception_carries_phase_context():
	error = PipelinePhaseError(
		"phase2_bronze_and_silver_sttm",
		"Bronze ingestion failed",
		code="bronze_failed",
		context={"input_count": 2},
	)

	assert error.phase == "phase2_bronze_and_silver_sttm"
	assert error.context["input_count"] == 2
	assert error.code == "bronze_failed"
	assert is_transient_error(TransientPipelineError("provider unavailable"))