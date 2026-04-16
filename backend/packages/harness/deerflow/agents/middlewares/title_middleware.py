"""Middleware for automatic thread title generation."""

import logging
from typing import Any, NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config
from langgraph.runtime import Runtime

from deerflow.config.deer_flow_context import DeerFlowContext
from deerflow.config.title_config import TitleConfig
from deerflow.models import create_chat_model

logger = logging.getLogger(__name__)


class TitleMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    title: NotRequired[str | None]


class TitleMiddleware(AgentMiddleware[TitleMiddlewareState]):
    """Automatically generate a title for the thread after the first user message."""

    state_schema = TitleMiddlewareState

    def _normalize_content(self, content: object) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = [self._normalize_content(item) for item in content]
            return "\n".join(part for part in parts if part)

        if isinstance(content, dict):
            text_value = content.get("text")
            if isinstance(text_value, str):
                return text_value

            nested_content = content.get("content")
            if nested_content is not None:
                return self._normalize_content(nested_content)

        return ""

    def _should_generate_title(self, state: TitleMiddlewareState, title_config: TitleConfig) -> bool:
        """Check if we should generate a title for this thread."""
        if not title_config.enabled:
            return False

        # Check if thread already has a title in state
        if state.get("title"):
            return False

        # Check if this is the first turn (has at least one user message and one assistant response)
        messages = state.get("messages", [])
        if len(messages) < 2:
            return False

        # Count user and assistant messages
        user_messages = [m for m in messages if m.type == "human"]
        assistant_messages = [m for m in messages if m.type == "ai"]

        # Generate title after first complete exchange
        return len(user_messages) == 1 and len(assistant_messages) >= 1

    def _build_title_prompt(self, state: TitleMiddlewareState, title_config: TitleConfig) -> tuple[str, str]:
        """Extract user/assistant messages and build the title prompt.

        Returns (prompt_string, user_msg) so callers can use user_msg as fallback.
        """
        messages = state.get("messages", [])

        user_msg_content = next((m.content for m in messages if m.type == "human"), "")
        assistant_msg_content = next((m.content for m in messages if m.type == "ai"), "")

        user_msg = self._normalize_content(user_msg_content)
        assistant_msg = self._normalize_content(assistant_msg_content)

        prompt = title_config.prompt_template.format(
            max_words=title_config.max_words,
            user_msg=user_msg[:500],
            assistant_msg=assistant_msg[:500],
        )
        return prompt, user_msg

    def _parse_title(self, content: object, title_config: TitleConfig) -> str:
        """Normalize model output into a clean title string."""
        title_content = self._normalize_content(content)
        title = title_content.strip().strip('"').strip("'")
        return title[: title_config.max_chars] if len(title) > title_config.max_chars else title

    def _fallback_title(self, user_msg: str, title_config: TitleConfig) -> str:
        fallback_chars = min(title_config.max_chars, 50)
        if len(user_msg) > fallback_chars:
            return user_msg[:fallback_chars].rstrip() + "..."
        return user_msg if user_msg else "New Conversation"

    def _get_runnable_config(self) -> dict[str, Any]:
        """Inherit the parent RunnableConfig and add middleware tag.

        This ensures RunJournal identifies LLM calls from this middleware
        as ``middleware:title`` instead of ``lead_agent``.
        """
        try:
            parent = get_config()
        except Exception:
            parent = {}
        config = {**parent}
        config["tags"] = [*(config.get("tags") or []), "middleware:title"]
        return config

    def _generate_title_result(self, state: TitleMiddlewareState, title_config: TitleConfig) -> dict | None:
        """Generate a local fallback title without blocking on an LLM call."""
        if not self._should_generate_title(state, title_config):
            return None

        _, user_msg = self._build_title_prompt(state, title_config)
        return {"title": self._fallback_title(user_msg, title_config)}

    async def _agenerate_title_result(self, state: TitleMiddlewareState, title_config: TitleConfig) -> dict | None:
        """Generate a title asynchronously and fall back locally on failure."""
        if not self._should_generate_title(state, title_config):
            return None

        prompt, user_msg = self._build_title_prompt(state, title_config)

        try:
            if title_config.model_name:
                model = create_chat_model(name=title_config.model_name, thinking_enabled=False)
            else:
                model = create_chat_model(thinking_enabled=False)
            response = await model.ainvoke(prompt, config=self._get_runnable_config())
            title = self._parse_title(response.content, title_config)
            if title:
                return {"title": title}
        except Exception:
            logger.debug("Failed to generate async title; falling back to local title", exc_info=True)
        return {"title": self._fallback_title(user_msg, title_config)}

    @override
    def after_model(self, state: TitleMiddlewareState, runtime: Runtime[DeerFlowContext]) -> dict | None:
        return self._generate_title_result(state, runtime.context.app_config.title)

    @override
    async def aafter_model(self, state: TitleMiddlewareState, runtime: Runtime[DeerFlowContext]) -> dict | None:
        return await self._agenerate_title_result(state, runtime.context.app_config.title)
