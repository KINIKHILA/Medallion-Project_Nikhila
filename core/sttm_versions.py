import difflib
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.config import DATA_DIR
from core.user_workspace import user_directory

STTM_COLUMNS = (
    "source_schema",
    "source_table",
    "source_column",
    "target_schema",
    "target_table",
    "target_column",
    "transformation_type",
    "transformation_logic",
)
_VALID_LAYERS = {"bronze", "silver", "gold"}


def _layer_directory(layer: str) -> Path:
    if layer not in _VALID_LAYERS:
        raise ValueError("Layer must be bronze, silver, or gold.")
    return user_directory("sttm_versions", DATA_DIR / "sttm_versions") / layer


def _normalize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            column: str(row.get(column, "") if row.get(column) is not None else "")
            for column in STTM_COLUMNS
        }
        for row in rows
    ]


def _content_hash(rows: Sequence[Dict[str, str]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def list_sttm_versions(layer: str) -> List[Dict[str, Any]]:
    directory = _layer_directory(layer)
    if not directory.exists():
        return []
    versions = []
    for path in directory.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as version_file:
                versions.append(json.load(version_file))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(versions, key=lambda version: version.get("created_at", ""), reverse=True)


def get_sttm_version(layer: str, version_id: str) -> Dict[str, Any]:
    if not re.fullmatch(r"\d{8}T\d{12}Z-[a-f0-9]{12}-[a-f0-9]{8}", version_id):
        raise ValueError("Invalid STTM version ID.")
    path = _layer_directory(layer) / f"{version_id}.json"
    with path.open("r", encoding="utf-8") as version_file:
        return json.load(version_file)


def record_sttm_version(
    layer: str,
    rows: Sequence[Dict[str, Any]],
    run_id: str,
    reason: str,
    audit_logger: Any,
    restored_from: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_rows = _normalize_rows(rows)
    digest = _content_hash(normalized_rows)
    latest = list_sttm_versions(layer)
    if latest and latest[0].get("content_hash") == digest:
        version = latest[0]
        audit_logger.log(
            "sttm_version_control",
            "version_unchanged",
            layer=layer,
            version_id=version["version_id"],
            reason=reason,
            content_hash=digest,
        )
        return version

    created_at = datetime.now(timezone.utc).isoformat()
    version_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-{digest[:12]}-{uuid.uuid4().hex[:8]}"
    )
    version = {
        "version_id": version_id,
        "layer": layer,
        "created_at": created_at,
        "run_id": str(run_id),
        "reason": reason,
        "previous_version_id": latest[0]["version_id"] if latest else None,
        "restored_from": restored_from,
        "content_hash": digest,
        "row_count": len(normalized_rows),
        "rows": normalized_rows,
    }
    directory = _layer_directory(layer)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{version_id}.json"
    temporary_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as version_file:
            json.dump(version, version_file, indent=2, ensure_ascii=False)
            version_file.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)

    audit_logger.log(
        "sttm_version_control",
        "version_created",
        layer=layer,
        version_id=version_id,
        reason=reason,
        previous_version_id=version["previous_version_id"],
        restored_from=restored_from,
        content_hash=digest,
        row_count=len(normalized_rows),
    )
    return version


def compare_sttm_versions(layer: str, left_id: str, right_id: str) -> List[str]:
    left = get_sttm_version(layer, left_id)
    right = get_sttm_version(layer, right_id)
    return compare_sttm_rows(left["rows"], right["rows"], left_id, right_id)


def compare_sttm_rows(
    left_rows: Sequence[Dict[str, Any]],
    right_rows: Sequence[Dict[str, Any]],
    left_label: str = "Current",
    right_label: str = "Selected version",
) -> List[str]:
    left_lines = [
        json.dumps(row, sort_keys=True, ensure_ascii=False)
        for row in sorted(left_rows, key=lambda item: json.dumps(item, sort_keys=True))
    ]
    right_lines = [
        json.dumps(row, sort_keys=True, ensure_ascii=False)
        for row in sorted(right_rows, key=lambda item: json.dumps(item, sort_keys=True))
    ]
    return list(
        difflib.unified_diff(
            left_lines,
            right_lines,
            fromfile=left_label,
            tofile=right_label,
            lineterm="",
        )
    )
