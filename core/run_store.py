import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import RUNS_DIR
from core.user_workspace import user_directory

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


def _run_path(run_id: str) -> Path:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("Invalid run ID.")
    return user_directory("runs", RUNS_DIR) / f"{run_id}.json"


def save_run_state(state: Dict[str, Any]) -> Path:
    run_id = str(state.get("run_id", ""))
    path = _run_path(run_id)
    try:
        from core.data_lineage import persist_lineage

        state["lineage_path"] = str(persist_lineage(state))
        state.pop("lineage_error", None)
    except Exception as error:
        state["lineage_error"] = str(error)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{run_id}_", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2, ensure_ascii=False, default=str)
            state_file.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def get_run_state(run_id: str) -> Optional[Dict[str, Any]]:
    path = _run_path(run_id)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as state_file:
        return json.load(state_file)


def list_run_states(limit: int = 50) -> List[Dict[str, Any]]:
    runs_dir = user_directory("runs", RUNS_DIR)
    if not runs_dir.exists():
        return []
    states = []
    for path in sorted(runs_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            with path.open("r", encoding="utf-8") as state_file:
                states.append(json.load(state_file))
        except (OSError, json.JSONDecodeError):
            continue
        if len(states) >= limit:
            break
    return states
