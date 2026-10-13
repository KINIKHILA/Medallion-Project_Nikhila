import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from core.user_workspace import user_directory
from core.performance import AgentMemorySampler, record_api_call, record_metric

TRACES_DIR = Path("data/traces")
TRACES_DIR.mkdir(parents=True, exist_ok=True)


class AgentTrace:
	def __init__(self, agent_name: str, run_id: str) -> None:
		self.agent_name = agent_name
		self.run_id = run_id
		self.start_time = time.time()
		self.memory_sampler = AgentMemorySampler(run_id, agent_name)
		self.trace = {
			"agent": agent_name,
			"run_id": run_id,
			"started_at": datetime.now(timezone.utc).isoformat(),
			"input": {},
			"plan": "",
			"tool_calls": [],
			"reasoning_steps": [],
			"output": {},
			"duration_seconds": None,
			"status": "running",
			"metrics": {
				"api_calls": 0,
				"input_tokens": 0,
				"output_tokens": 0,
				"estimated_cost_usd": None,
			},
		}

	def set_input(self, **kwargs: Any) -> "AgentTrace":
		self.trace["input"] = kwargs
		return self

	def set_plan(self, plan_str: str) -> "AgentTrace":
		self.trace["plan"] = plan_str
		return self

	def set_output(self, **kwargs: Any) -> "AgentTrace":
		self.trace["output"] = kwargs
		return self

	def extract_from_messages(self, messages: Iterable[Any]) -> "AgentTrace":
		for message in messages:
			if isinstance(message, HumanMessage):
				self.trace["reasoning_steps"].append(
					{"type": "task_input", "content": message.content}
				)
			elif isinstance(message, AIMessage):
				self.trace["tool_calls"].extend(message.tool_calls or [])
				usage = record_api_call(self.run_id, self.agent_name, message)
				metrics = self.trace["metrics"]
				metrics["api_calls"] += 1
				metrics["input_tokens"] += usage["input_tokens"]
				metrics["output_tokens"] += usage["output_tokens"]
				if usage["estimated_cost_usd"] is not None:
					metrics["estimated_cost_usd"] = (
						metrics["estimated_cost_usd"] or 0.0
					) + usage["estimated_cost_usd"]
				content = message.content
				substantive = bool(content.strip()) if isinstance(content, str) else bool(content)
				if substantive:
					self.trace["reasoning_steps"].append(
						{"type": "ai_reasoning", "content": content}
					)
					if not self.trace["plan"]:
						self.trace["plan"] = content
			elif isinstance(message, ToolMessage):
				tool_result = {"type": "tool_result", "content": message.content}
				if message.name:
					tool_result["name"] = message.name
				if message.tool_call_id:
					tool_result["tool_call_id"] = message.tool_call_id
				self.trace["reasoning_steps"].append(tool_result)
		return self

	def complete(self, status: str = "success") -> None:
		self.trace["duration_seconds"] = time.time() - self.start_time
		self.trace["status"] = status
		record_metric(
			self.run_id,
			self.agent_name,
			"agent_duration_seconds",
			self.trace["duration_seconds"],
			unit="seconds",
		)
		self.trace["metrics"]["peak_memory_mb"] = self.memory_sampler.stop()
		trace_dir = user_directory("traces", TRACES_DIR)
		trace_path = trace_dir / f"trace_{self.agent_name}_{self.run_id[:8]}.json"
		trace_path.parent.mkdir(parents=True, exist_ok=True)
		with trace_path.open("w", encoding="utf-8") as trace_file:
			json.dump(self.trace, trace_file, indent=2, default=str)
			trace_file.write("\n")
		print(
			f"Trace {self.agent_name} ({self.run_id[:8]}): "
			f"{self.trace['status']} in {self.trace['duration_seconds']:.2f}s"
		)

	def fail(self, error: Any) -> None:
		self.trace["error"] = str(error)
		self.complete(status="failed")