import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from core.config import CACHE_DIR, GEMINI_MODEL, GITHUB_MODEL, GROQ_MODEL, LLM_PROVIDER, OPENAI_MODEL
from core.user_workspace import user_directory

CACHE_FORMAT_VERSION = 1
FilePath = Union[str, Path]


def build_cache_key(namespace: str, payload: Any) -> str:
    encoded = json.dumps(
        {"version": CACHE_FORMAT_VERSION, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    safe_namespace = re.sub(r"[^A-Za-z0-9._-]", "_", namespace)
    return f"{safe_namespace}_{digest}"


def fingerprint_file(file_path: FilePath) -> Dict[str, Any]:
    path = Path(file_path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def configured_model_signature() -> Dict[str, str]:
    models = {
        "groq": GROQ_MODEL,
        "gemini": GEMINI_MODEL,
        "openai": OPENAI_MODEL,
        "github": GITHUB_MODEL,
    }
    return {"provider": LLM_PROVIDER or "none", "model": models.get(LLM_PROVIDER, "unknown")}


def _cache_path(namespace: str, key: str) -> Path:
    cache_dir = user_directory("cache", CACHE_DIR)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", f"{namespace}_{key}")
    return cache_dir / f"{safe_name}.json"


def get_cached_json(namespace: str, key: str) -> Optional[Any]:
    path = _cache_path(namespace, key)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        return None


def set_cached_json(namespace: str, key: str, value: Any) -> None:
    path = _cache_path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as cache_file:
            json.dump(value, cache_file, indent=2, ensure_ascii=False, default=str)
            cache_file.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def clear_cache(namespace: Optional[str] = None) -> int:
    """Clear the active user's cache, optionally restricting to a namespace."""
    cache_dir = user_directory("cache", CACHE_DIR)
    if not cache_dir.exists():
        return 0
    prefix = re.sub(r"[^A-Za-z0-9._-]", "_", namespace) + "_" if namespace else None
    removed = 0
    for path in cache_dir.glob("*.json"):
        if prefix is None or path.name.startswith(prefix):
            path.unlink(missing_ok=True)
            removed += 1
    return removed
