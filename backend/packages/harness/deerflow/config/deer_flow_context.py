"""Per-invocation context for DeerFlow agent execution.

Injected via LangGraph Runtime. Middleware and tools access this
via Runtime[DeerFlowContext] parameters, through resolve_context().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeerFlowContext:
    """Typed, immutable, per-invocation context injected via LangGraph Runtime.

    Fields are all known at run start and never change during execution.
    Mutable runtime state (e.g. sandbox_id) flows through ThreadState, not here.
    """

    app_config: AppConfig
    thread_id: str
    agent_name: str | None = None


def resolve_context(runtime: Any) -> DeerFlowContext:
    """Extract or construct DeerFlowContext from runtime.

    Gateway/Client paths: runtime.context is already DeerFlowContext → return directly.
    LangGraph Server / legacy dict path: construct from dict context or configurable fallback.
    """
    ctx = getattr(runtime, "context", None)
    if isinstance(ctx, DeerFlowContext):
        return ctx

    from deerflow.config.app_config import AppConfig

    # Try dict context first (legacy path, tests), then configurable
    if isinstance(ctx, dict):
        thread_id = ctx.get("thread_id", "")
        if not thread_id:
            logger.warning("resolve_context: dict context has empty thread_id — may cause incorrect path resolution")
        return DeerFlowContext(
            app_config=AppConfig.current(),
            thread_id=thread_id,
            agent_name=ctx.get("agent_name"),
        )

    # No context at all — fall back to LangGraph configurable
    try:
        from langgraph.config import get_config

        cfg = get_config().get("configurable", {})
    except RuntimeError:
        # Outside runnable context (e.g. unit tests)
        cfg = {}

    thread_id = cfg.get("thread_id", "")
    if not thread_id:
        logger.warning("resolve_context: falling back to empty thread_id — no DeerFlowContext or configurable found")

    return DeerFlowContext(
        app_config=AppConfig.current(),
        thread_id=thread_id,
        agent_name=cfg.get("agent_name"),
    )
