from pathlib import Path

from core import cache, user_workspace


def test_cache_round_trip_clear_and_user_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(user_workspace, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    key = cache.build_cache_key("profile", {"input": "same"})

    with user_workspace.use_user_workspace("alice"):
        cache.set_cached_json("profile", key, {"columns": ["id"]})
        assert cache.get_cached_json("profile", key) == {"columns": ["id"]}
        assert cache.clear_cache("sttm_silver") == 0
        assert cache.clear_cache("profile") == 1
        assert cache.get_cached_json("profile", key) is None

    with user_workspace.use_user_workspace("bob"):
        assert cache.get_cached_json("profile", key) is None


def test_file_fingerprint_changes_with_content(tmp_path):
    source = tmp_path / "sales.csv"
    source.write_text("id,total\n1,10\n", encoding="utf-8")
    first = cache.fingerprint_file(source)
    source.write_text("id,total\n1,11\n", encoding="utf-8")

    assert cache.fingerprint_file(source)["sha256"] != first["sha256"]


def test_cache_key_is_order_independent_for_json_objects():
    assert cache.build_cache_key("profile", {"a": 1, "b": 2}) == cache.build_cache_key(
        "profile", {"b": 2, "a": 1}
    )