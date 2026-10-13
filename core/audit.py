import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import AUDIT_DIR
from core.user_workspace import user_directory


class AuditLogger:
	def __init__(self, run_id: Optional[str] = None) -> None:
		self.run_id = str(run_id) if run_id is not None else str(uuid.uuid4())
		self.log_path = user_directory("audit_logs", AUDIT_DIR) / f"{self.run_id}.jsonl"

	def log(self, agent: str, action: str, **kwargs: Any) -> None:
		entry = {
			**kwargs,
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"run_id": self.run_id,
			"agent": agent,
			"action": action,
		}
		self.log_path.parent.mkdir(parents=True, exist_ok=True)
		with self.log_path.open("a", encoding="utf-8") as log_file:
			log_file.write(json.dumps(entry, default=str) + "\n")

	def get_logs(self) -> List[Dict[str, Any]]:
		if not self.log_path.exists():
			return []

		with self.log_path.open("r", encoding="utf-8") as log_file:
			return [json.loads(line) for line in log_file if line.strip()]