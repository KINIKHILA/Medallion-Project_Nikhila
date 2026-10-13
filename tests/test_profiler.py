"""Tests for profiler profile-file generation with a mocked LLM and agent runner."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import profiler
from core import cache


class _FakeTrace:
    def __init__(self, *args):
        pass

    def set_input(self, **kwargs):
        return self

    def set_plan(self, *args):
        return self

    def extract_from_messages(self, *args):
        return self

    def set_output(self, **kwargs):
        return self

    def complete(self):
        return None

    def fail(self, error):
        raise error


class _FakeAuditLogger:
    def __init__(self, *args):
        pass

    def log(self, *args, **kwargs):
        return None


def _write_csv(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _mock_profiler_llm(monkeypatch, profiles_dir: Path) -> list[dict]:
    llm_calls = []
    provider_module = types.ModuleType("langchain_openai")

    def fake_chat_openai(**kwargs):
        llm_calls.append(kwargs)
        return SimpleNamespace(name="mock-chat-openai")

    provider_module.ChatOpenAI = fake_chat_openai
    monkeypatch.setitem(sys.modules, "langchain_openai", provider_module)
    monkeypatch.setattr(profiler, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(profiler, "OPENAI_API_KEY", "test-api-key")
    monkeypatch.setattr(profiler, "PROFILES_DIR", profiles_dir)
    monkeypatch.setattr(profiler, "AgentTrace", _FakeTrace)
    monkeypatch.setattr(profiler, "AuditLogger", _FakeAuditLogger)

    prebuilt_module = types.ModuleType("langgraph.prebuilt")

    def fake_create_react_agent(llm, tools, prompt):
        assert llm.name == "mock-chat-openai"

        class FakeAgent:
            def invoke(self, request):
                from langchain_core.messages import ToolMessage

                messages = []
                for index, tool in enumerate(tools):
                    result = tool.invoke({})
                    messages.append(
                        ToolMessage(
                            content=str(result),
                            name=tool.name,
                            tool_call_id=f"mock-tool-call-{index}",
                        )
                    )
                messages.append(
                    SimpleNamespace(
                        type="ai",
                        content=json.dumps(
                            {
                                "semantic_meanings": ["Retail transaction data"],
                                "join_keys": ["product_id"],
                                "quality_notes": ["Check missing store identifiers."],
                            }
                        ),
                    )
                )
                return {"messages": messages}

        return FakeAgent()

    prebuilt_module.create_react_agent = fake_create_react_agent
    langgraph_module = types.ModuleType("langgraph")
    langgraph_module.prebuilt = prebuilt_module
    monkeypatch.setitem(sys.modules, "langgraph", langgraph_module)
    monkeypatch.setitem(sys.modules, "langgraph.prebuilt", prebuilt_module)
    return llm_calls


class TestProfiler:
    def test_profile_dataset_creates_json(self, tmp_path, monkeypatch):
        csv_path = _write_csv(
            tmp_path / "sales.csv",
            "transaction_id,product_id,revenue\nT1,P001,12.5\nT2,P002,20.0\n",
        )
        profiles_dir = tmp_path / "profiles"
        llm_calls = _mock_profiler_llm(monkeypatch, profiles_dir)

        profile_path = profiler.profile_dataset(
            csv_path,
            "test-run-single",
            "Describe retail sales quality.",
        )

        result_path = Path(profile_path)
        assert result_path.is_file()
        profile = json.loads(result_path.read_text(encoding="utf-8"))
        inspection = profile["inspection"]["files"][0]
        assert inspection["shape"] == [2, 3]
        assert inspection["columns"] == ["transaction_id", "product_id", "revenue"]
        assert set(profile["statistics"]["files"][0]["columns"]) == {
            "transaction_id",
            "product_id",
            "revenue",
        }
        assert profile["semantic_meanings"] == ["Retail transaction data"]
        assert len(llm_calls) == 1

    def test_repeated_profile_uses_cached_result(self, tmp_path, monkeypatch):
        csv_path = _write_csv(
            tmp_path / "sales.csv",
            "transaction_id,revenue\nT1,12.5\nT2,20.0\n",
        )
        profiles_dir = tmp_path / "profiles"
        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
        llm_calls = _mock_profiler_llm(monkeypatch, profiles_dir)

        first_path = profiler.profile_dataset(csv_path, "first-run", "Profile sales.")
        second_path = profiler.profile_dataset(csv_path, "second-run", "Profile sales.")

        assert first_path != second_path
        assert len(llm_calls) == 1
        second_profile = json.loads(Path(second_path).read_text(encoding="utf-8"))
        assert second_profile["run_id"] == "second-run"

    def test_profile_multiple_datasets(self, tmp_path, monkeypatch):
        sales_path = _write_csv(
            tmp_path / "sales.csv",
            "transaction_id,revenue\nT1,12.5\nT2,20.0\n",
        )
        products_path = _write_csv(
            tmp_path / "products.csv",
            "product_id,category\nP001,Books\nP002,Games\n",
        )
        profiles_dir = tmp_path / "profiles"
        llm_calls = _mock_profiler_llm(monkeypatch, profiles_dir)

        profile_path = profiler.profile_multiple_datasets(
            [sales_path, products_path],
            "test-run-multiple",
            "Profile sales and product catalog data.",
        )

        profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        assert len(profile["inspection"]["files"]) == 2
        assert len(profile["statistics"]["files"]) == 2
        assert {Path(item["path"]).name for item in profile["statistics"]["files"]} == {
            "sales.csv",
            "products.csv",
        }
        assert len(llm_calls) == 1
