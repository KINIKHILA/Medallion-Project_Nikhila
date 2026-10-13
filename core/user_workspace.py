import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Optional

from core.config import DATA_DIR

_ACTIVE_USERNAME: ContextVar[Optional[str]] = ContextVar(
    "idamp_active_username", default=None
)
_ALLOWED_DIRECTORIES = {
    "landing",
    "profiles",
    "sttm",
	"sttm_versions",
    "bronze_layer",
    "silver_layer",
    "gold_layer",
    "reports",
    "audit_logs",
    "traces",
    "metrics",
    "cache",
	"runs",
    "chroma",
}


def normalize_username(username: str) -> str:
    normalized = username.strip().casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,39}", normalized):
        raise ValueError(
            "Username must be 1-40 characters, start with a letter or number, "
            "and contain only letters, numbers, dots, underscores, or hyphens."
        )
    return normalized


def current_username() -> Optional[str]:
    return _ACTIVE_USERNAME.get()


def set_current_user(username: str) -> str:
    """Set a context-local username and initialize that user's data folders."""
    normalized = normalize_username(username)
    _ACTIVE_USERNAME.set(normalized)
    for directory in _ALLOWED_DIRECTORIES:
        user_directory(directory, DATA_DIR / directory).mkdir(parents=True, exist_ok=True)
    return normalized


def clear_current_user() -> None:
    _ACTIVE_USERNAME.set(None)


@contextmanager
def use_user_workspace(username: str) -> Iterator[str]:
    """Temporarily activate a user workspace and restore the previous context."""
    normalized = normalize_username(username)
    token = _ACTIVE_USERNAME.set(normalized)
    try:
        for directory in _ALLOWED_DIRECTORIES:
            user_directory(directory, DATA_DIR / directory).mkdir(parents=True, exist_ok=True)
        yield normalized
    finally:
        _ACTIVE_USERNAME.reset(token)


def user_directory(name: str, default_path: Path) -> Path:
    """Return the active user's directory, or the supplied legacy default."""
    if name not in _ALLOWED_DIRECTORIES:
        raise ValueError(f"Unsupported user workspace directory: {name}")
    username = current_username()
    if username is None:
        return Path(default_path)
    return DATA_DIR / "users" / username / name


def user_workspace_root(username: Optional[str] = None) -> Path:
    """Return a user's root directory for displaying workspace details."""
    selected_user = normalize_username(username) if username else current_username()
    if selected_user is None:
        return DATA_DIR
    return DATA_DIR / "users" / selected_user
