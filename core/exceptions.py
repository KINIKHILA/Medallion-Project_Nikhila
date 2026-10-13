from typing import Any, Dict, Optional


class PipelineError(Exception):
	"""Base class for errors with user-safe messages and diagnostic context."""

	def __init__(
		self,
		message: str,
		*,
		code: str = "pipeline_error",
		context: Optional[Dict[str, Any]] = None,
		retryable: bool = False,
	) -> None:
		super().__init__(message)
		self.code = code
		self.context = context or {}
		self.retryable = retryable


class PipelinePhaseError(PipelineError):
	"""An error tied to a named pipeline phase."""

	def __init__(self, phase: str, message: str, **kwargs: Any) -> None:
		context = {**kwargs.pop("context", {}), "phase": phase}
		super().__init__(message, context=context, **kwargs)
		self.phase = phase


class DataQualityGateError(PipelinePhaseError):
	"""Raised when a layer does not meet its configured quality threshold."""

	def __init__(self, phase: str, report: Dict[str, Any]) -> None:
		super().__init__(
			phase,
			f"{report.get('layer', 'Data')} quality score "
			f"{report.get('score', 0):.2f} is below required threshold "
			f"{report.get('minimum_score', 0):.2f}.",
			code="data_quality_gate_failed",
			context={"quality_report": report},
		)
		self.quality_report = report


class TransientPipelineError(PipelineError):
	"""An explicitly retryable external-service or network failure."""

	def __init__(self, message: str, **kwargs: Any) -> None:
		kwargs["retryable"] = True
		kwargs.setdefault("code", "transient_pipeline_error")
		super().__init__(message, **kwargs)