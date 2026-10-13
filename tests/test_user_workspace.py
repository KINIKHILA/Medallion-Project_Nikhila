"""Tests for user-specific workspace path isolation."""

from pathlib import Path

import pytest

from core import user_workspace


def test_users_receive_separate_workspace_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(user_workspace, "DATA_DIR", tmp_path / "data")

    alice = user_workspace.set_current_user("Alice")
    alice_landing = user_workspace.user_directory("landing", tmp_path / "landing")
    alice_reports = user_workspace.user_directory("reports", tmp_path / "reports")
    user_workspace.clear_current_user()

    bob = user_workspace.set_current_user("Bob")
    bob_landing = user_workspace.user_directory("landing", tmp_path / "landing")
    bob_reports = user_workspace.user_directory("reports", tmp_path / "reports")
    user_workspace.clear_current_user()

    assert alice == "alice"
    assert bob == "bob"
    assert alice_landing == tmp_path / "data" / "users" / "alice" / "landing"
    assert bob_landing == tmp_path / "data" / "users" / "bob" / "landing"
    assert alice_reports != bob_reports
    assert alice_landing.is_dir() and bob_landing.is_dir()


def test_default_directory_is_used_without_active_user(tmp_path):
    user_workspace.clear_current_user()

    assert user_workspace.user_directory("reports", tmp_path / "reports") == tmp_path / "reports"


def test_username_cannot_escape_user_data_root():
    with pytest.raises(ValueError):
        user_workspace.set_current_user("../other-user")
    user_workspace.clear_current_user()
