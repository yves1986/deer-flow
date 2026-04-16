"""Tests for lead agent runtime model resolution behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deerflow.agents.lead_agent import agent as lead_agent_module
from deerflow.config.app_config import AppConfig
from deerflow.config.model_config import ModelConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.summarization_config import SummarizationConfig


def _make_app_config(models: list[ModelConfig]) -> AppConfig:
    return AppConfig(
        models=models,
        sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
    )


def _make_model(name: str, *, supports_thinking: bool) -> ModelConfig:
    return ModelConfig(
        name=name,
        display_name=name,
        description=None,
        use="langchain_openai:ChatOpenAI",
        model=name,
        supports_thinking=supports_thinking,
        supports_vision=False,
    )


def test_resolve_model_name_falls_back_to_default(caplog):
    app_config = _make_app_config(
        [
            _make_model("default-model", supports_thinking=False),
            _make_model("other-model", supports_thinking=True),
        ]
    )

    with caplog.at_level("WARNING"):
        resolved = lead_agent_module._resolve_model_name(app_config, "missing-model")

    assert resolved == "default-model"
    assert "fallback to default model 'default-model'" in caplog.text


def test_resolve_model_name_uses_default_when_none():
    app_config = _make_app_config(
        [
            _make_model("default-model", supports_thinking=False),
            _make_model("other-model", supports_thinking=True),
        ]
    )

    resolved = lead_agent_module._resolve_model_name(app_config, None)

    assert resolved == "default-model"


def test_resolve_model_name_raises_when_no_models_configured():
    app_config = _make_app_config([])

    with pytest.raises(
        ValueError,
        match="No chat models are configured",
    ):
        lead_agent_module._resolve_model_name(app_config, "missing-model")


def test_make_lead_agent_disables_thinking_when_model_does_not_support_it(monkeypatch):
    app_config = _make_app_config([_make_model("safe-model", supports_thinking=False)])

    import deerflow.tools as tools_module

    monkeypatch.setattr(AppConfig, "current", staticmethod(lambda: app_config))
    monkeypatch.setattr(tools_module, "get_available_tools", lambda **kwargs: [])
    monkeypatch.setattr(lead_agent_module, "_build_middlewares", lambda app_config, config, model_name, agent_name=None: [])

    captured: dict[str, object] = {}

    def _fake_create_chat_model(*, name, thinking_enabled, reasoning_effort=None):
        captured["name"] = name
        captured["thinking_enabled"] = thinking_enabled
        captured["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(lead_agent_module, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(lead_agent_module, "create_agent", lambda **kwargs: kwargs)

    result = lead_agent_module.make_lead_agent(
        {
            "configurable": {
                "model_name": "safe-model",
                "thinking_enabled": True,
                "is_plan_mode": False,
                "subagent_enabled": False,
            }
        }
    )

    assert captured["name"] == "safe-model"
    assert captured["thinking_enabled"] is False
    assert result["model"] is not None


def test_build_middlewares_uses_resolved_model_name_for_vision(monkeypatch):
    app_config = _make_app_config(
        [
            _make_model("stale-model", supports_thinking=False),
            ModelConfig(
                name="vision-model",
                display_name="vision-model",
                description=None,
                use="langchain_openai:ChatOpenAI",
                model="vision-model",
                supports_thinking=False,
                supports_vision=True,
            ),
        ]
    )

    AppConfig.init(app_config)
    monkeypatch.setattr(AppConfig, "current", staticmethod(lambda: app_config))
    monkeypatch.setattr(lead_agent_module, "_create_summarization_middleware", lambda _ac: None)
    monkeypatch.setattr(lead_agent_module, "_create_todo_list_middleware", lambda is_plan_mode: None)

    middlewares = lead_agent_module._build_middlewares(app_config, {"configurable": {"model_name": "stale-model", "is_plan_mode": False, "subagent_enabled": False}}, model_name="vision-model", custom_middlewares=[MagicMock()])

    assert any(isinstance(m, lead_agent_module.ViewImageMiddleware) for m in middlewares)
    # verify the custom middleware is injected correctly
    assert len(middlewares) > 0 and isinstance(middlewares[-2], MagicMock)


def test_create_summarization_middleware_uses_configured_model_alias(monkeypatch):
    app_config = _make_app_config([_make_model("default", supports_thinking=False)])
    patched = app_config.model_copy(update={"summarization": SummarizationConfig(enabled=True, model_name="model-masswork")})
    AppConfig.init(patched)
    monkeypatch.setattr(AppConfig, "current", staticmethod(lambda: patched))

    from unittest.mock import MagicMock

    captured: dict[str, object] = {}
    fake_model = MagicMock()
    fake_model.with_config.return_value = fake_model

    def _fake_create_chat_model(*, name=None, thinking_enabled, reasoning_effort=None):
        captured["name"] = name
        captured["thinking_enabled"] = thinking_enabled
        captured["reasoning_effort"] = reasoning_effort
        return fake_model

    monkeypatch.setattr(lead_agent_module, "create_chat_model", _fake_create_chat_model)
    monkeypatch.setattr(lead_agent_module, "SummarizationMiddleware", lambda **kwargs: kwargs)

    middleware = lead_agent_module._create_summarization_middleware(patched)

    assert captured["name"] == "model-masswork"
    assert captured["thinking_enabled"] is False
    assert middleware["model"] is fake_model
    fake_model.with_config.assert_called_once_with(tags=["middleware:summarize"])
