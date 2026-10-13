import time
from typing import Any, Callable, Optional, TypeVar

from core.exceptions import TransientPipelineError

T = TypeVar("T")


def is_transient_error(error: BaseException) -> bool:
	"""Recognize explicit transient errors and common provider/network failures."""
	if isinstance(error, TransientPipelineError):
		return True
	if getattr(error, "retryable", False):
		return True
	status_code = getattr(error, "status_code", None)
	if status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600):
		return True
	name = type(error).__name__.casefold()
	message = str(error).casefold()
	transient_markers = (
		"timeout",
		"connection",
		"rate limit",
		"ratelimit",
		"temporarily unavailable",
		"service unavailable",
		"overloaded",
	)
	return any(marker in name or marker in message for marker in transient_markers)


def retry_transient(
	operation: Callable[[], T],
	*,
	max_attempts: int = 3,
	base_delay_seconds: float = 0.5,
	max_delay_seconds: float = 4.0,
	on_retry: Optional[Callable[[BaseException, int], None]] = None,
	sleep: Callable[[float], None] = time.sleep,
) -> T:
	"""Retry an operation only for transient errors, with exponential backoff."""
	if max_attempts < 1:
		raise ValueError("max_attempts must be at least 1")
	for attempt in range(1, max_attempts + 1):
		try:
			return operation()
		except Exception as error:
			if not is_transient_error(error) or attempt == max_attempts:
				raise
			if on_retry is not None:
				on_retry(error, attempt)
			delay = min(base_delay_seconds * (2 ** (attempt - 1)), max_delay_seconds)
			sleep(delay)
	raise RuntimeError("Retry loop exited unexpectedly")