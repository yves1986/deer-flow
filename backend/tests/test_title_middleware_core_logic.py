"""Core behavior tests for TitleMiddleware."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.middlewares import title_middleware as title_middleware_module
from deerflow.agents.middlewares.title_middleware import TitleMiddleware
from deerflow.config.app_config import AppConfig
from deerflow.config.deer_flow_context import DeerFlowContext
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.title_config import TitleConfig


def _make_title_config(**overrides) -> TitleConfig:
    return TitleConfig(**overrides)


def _make_runtime(**title_overrides) -> SimpleNamespace:
    """Build a runtime whose context carries a DeerFlowContext with the given TitleConfig."""
    app_config = AppConfig(sandbox=SandboxConfig(use="test"), title=TitleConfig(**title_overrides))
    ctx = DeerFlowContext(app_config=app_config, thread_id="t1")
    return SimpleNamespace(context=ctx)


class TestTitleMiddlewareCoreLogic:
    def test_should_generate_title_for_first_complete_exchange(self):
        middleware = TitleMiddleware()
        state = {
            "messages": [
                HumanMessage(content="帮我总结这段代码"),
                AIMessage(content="好的，我先看结构"),
            ]
        }

        assert middleware._should_generate_title(state, _make_title_config(enabled=True)) is True

    def test_should_not_generate_title_when_disabled_or_already_set(self):
        middleware = TitleMiddleware()

        disabled_state = {
            "messages": [HumanMessage(content="Q"), AIMessage(content="A")],
            "title": None,
        }
        assert middleware._should_generate_title(disabled_state, _make_title_config(enabled=False)) is False

        titled_state = {
            "messages": [HumanMessage(content="Q"), AIMessage(content="A")],
            "title": "Existing Title",
        }
        assert middleware._should_generate_title(titled_state, _make_title_config(enabled=True)) is False

    def test_should_not_generate_title_after_second_user_turn(self):
        middleware = TitleMiddleware()
        state = {
            "messages": [
                HumanMessage(content="第一问"),
                AIMessage(content="第一答"),
                HumanMessage(content="第二问"),
                AIMessage(content="第二答"),
            ]
        }

        assert middleware._should_generate_title(state, _make_title_config(enabled=True)) is False

    def test_generate_title_uses_async_model_and_respects_max_chars(self, monkeypatch):
        middleware = TitleMiddleware()
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="短标题"))
        monkeypatch.setattr(title_middleware_module, "create_chat_model", MagicMock(return_value=model))

        state = {
            "messages": [
                HumanMessage(content="请帮我写一个很长很长的脚本标题"),
                AIMessage(content="好的，先确认需求"),
            ]
        }
        result = asyncio.run(middleware._agenerate_title_result(state, _make_title_config(max_chars=12)))
        title = result["title"]

        assert title == "短标题"
        title_middleware_module.create_chat_model.assert_called_once_with(thinking_enabled=False)
        model.ainvoke.assert_awaited_once()

    def test_generate_title_normalizes_structured_message_content(self, monkeypatch):
        middleware = TitleMiddleware()
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="请帮我总结这段代码"))
        monkeypatch.setattr(title_middleware_module, "create_chat_model", MagicMock(return_value=model))

        state = {
            "messages": [
                HumanMessage(content=[{"type": "text", "text": "请帮我总结这段代码"}]),
                AIMessage(content=[{"type": "text", "text": "好的，先看结构"}]),
            ]
        }

        result = asyncio.run(middleware._agenerate_title_result(state, _make_title_config(max_chars=20)))
        title = result["title"]

        assert title == "请帮我总结这段代码"

    def test_generate_title_fallback_for_long_message(self, monkeypatch):
        middleware = TitleMiddleware()
        model = MagicMock()
        model.ainvoke = AsyncMock(side_effect=RuntimeError("model unavailable"))
        monkeypatch.setattr(title_middleware_module, "create_chat_model", MagicMock(return_value=model))

        state = {
            "messages": [
                HumanMessage(content="这是一个非常长的问题描述，需要被截断以形成fallback标题"),
                AIMessage(content="收到"),
            ]
        }
        result = asyncio.run(middleware._agenerate_title_result(state, _make_title_config(max_chars=20)))
        title = result["title"]

        # Assert behavior (truncated fallback + ellipsis) without overfitting exact text.
        assert title.endswith("...")
        assert title.startswith("这是一个非常长的问题描述")

    def test_aafter_model_delegates_to_async_helper(self, monkeypatch):
        middleware = TitleMiddleware()

        monkeypatch.setattr(middleware, "_agenerate_title_result", AsyncMock(return_value={"title": "异步标题"}))
        result = asyncio.run(middleware.aafter_model({"messages": []}, runtime=_make_runtime()))
        assert result == {"title": "异步标题"}

        monkeypatch.setattr(middleware, "_agenerate_title_result", AsyncMock(return_value=None))
        assert asyncio.run(middleware.aafter_model({"messages": []}, runtime=_make_runtime())) is None

    def test_after_model_sync_delegates_to_sync_helper(self, monkeypatch):
        middleware = TitleMiddleware()

        monkeypatch.setattr(middleware, "_generate_title_result", MagicMock(return_value={"title": "同步标题"}))
        result = middleware.after_model({"messages": []}, runtime=_make_runtime())
        assert result == {"title": "同步标题"}

        monkeypatch.setattr(middleware, "_generate_title_result", MagicMock(return_value=None))
        assert middleware.after_model({"messages": []}, runtime=_make_runtime()) is None

    def test_sync_generate_title_uses_fallback_without_model(self):
        """Sync path avoids LLM calls and derives a local fallback title."""
        middleware = TitleMiddleware()

        state = {
            "messages": [
                HumanMessage(content="请帮我写测试"),
                AIMessage(content="好的"),
            ]
        }
        result = middleware._generate_title_result(state, _make_title_config(max_chars=20))
        assert result == {"title": "请帮我写测试"}

    def test_sync_generate_title_respects_fallback_truncation(self):
        """Sync fallback path still respects max_chars truncation rules."""
        middleware = TitleMiddleware()

        state = {
            "messages": [
                HumanMessage(content="这是一个非常长的问题描述，需要被截断以形成fallback标题，而且这里继续补充更多上下文，确保超过本地fallback截断阈值"),
                AIMessage(content="回复"),
            ]
        }
        result = middleware._generate_title_result(state, _make_title_config(max_chars=50))
        assert result["title"].endswith("...")
        assert result["title"].startswith("这是一个非常长的问题描述")
