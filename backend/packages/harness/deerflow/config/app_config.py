from __future__ import annotations

import logging
import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, ClassVar, Self

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from deerflow.config.acp_config import ACPAgentConfig
from deerflow.config.checkpointer_config import CheckpointerConfig
from deerflow.config.database_config import DatabaseConfig
from deerflow.config.extensions_config import ExtensionsConfig
from deerflow.config.guardrails_config import GuardrailsConfig
from deerflow.config.memory_config import MemoryConfig
from deerflow.config.model_config import ModelConfig
from deerflow.config.run_events_config import RunEventsConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.skill_evolution_config import SkillEvolutionConfig
from deerflow.config.skills_config import SkillsConfig
from deerflow.config.stream_bridge_config import StreamBridgeConfig
from deerflow.config.subagents_config import SubagentsAppConfig
from deerflow.config.summarization_config import SummarizationConfig
from deerflow.config.title_config import TitleConfig
from deerflow.config.token_usage_config import TokenUsageConfig
from deerflow.config.tool_config import ToolConfig, ToolGroupConfig
from deerflow.config.tool_search_config import ToolSearchConfig

load_dotenv()

logger = logging.getLogger(__name__)


def _default_config_candidates() -> tuple[Path, ...]:
    """Return deterministic config.yaml locations without relying on cwd."""
    backend_dir = Path(__file__).resolve().parents[4]
    repo_root = backend_dir.parent
    return (backend_dir / "config.yaml", repo_root / "config.yaml")


class AppConfig(BaseModel):
    """Config for the DeerFlow application"""

    log_level: str = Field(default="info", description="Logging level for deerflow modules (debug/info/warning/error)")
    token_usage: TokenUsageConfig = Field(default_factory=TokenUsageConfig, description="Token usage tracking configuration")
    models: list[ModelConfig] = Field(default_factory=list, description="Available models")
    sandbox: SandboxConfig = Field(description="Sandbox configuration")
    tools: list[ToolConfig] = Field(default_factory=list, description="Available tools")
    tool_groups: list[ToolGroupConfig] = Field(default_factory=list, description="Available tool groups")
    skills: SkillsConfig = Field(default_factory=SkillsConfig, description="Skills configuration")
    skill_evolution: SkillEvolutionConfig = Field(default_factory=SkillEvolutionConfig, description="Agent-managed skill evolution configuration")
    extensions: ExtensionsConfig = Field(default_factory=ExtensionsConfig, description="Extensions configuration (MCP servers and skills state)")
    tool_search: ToolSearchConfig = Field(default_factory=ToolSearchConfig, description="Tool search / deferred loading configuration")
    title: TitleConfig = Field(default_factory=TitleConfig, description="Automatic title generation configuration")
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig, description="Conversation summarization configuration")
    memory: MemoryConfig = Field(default_factory=MemoryConfig, description="Memory subsystem configuration")
    subagents: SubagentsAppConfig = Field(default_factory=SubagentsAppConfig, description="Subagent runtime configuration")
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig, description="Guardrail middleware configuration")
    model_config = ConfigDict(extra="allow", frozen=True)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Unified database backend configuration")
    run_events: RunEventsConfig = Field(default_factory=RunEventsConfig, description="Run event storage configuration")
    checkpointer: CheckpointerConfig | None = Field(default=None, description="Checkpointer configuration")
    stream_bridge: StreamBridgeConfig | None = Field(default=None, description="Stream bridge configuration")
    acp_agents: dict[str, ACPAgentConfig] = Field(default_factory=dict, description="ACP agent configurations keyed by agent name")

    @classmethod
    def resolve_config_path(cls, config_path: str | None = None) -> Path:
        """Resolve the config file path.

        Priority:
        1. If provided `config_path` argument, use it.
        2. If provided `DEER_FLOW_CONFIG_PATH` environment variable, use it.
        3. Otherwise, search deterministic backend/repository-root defaults from `_default_config_candidates()`.
        """
        if config_path:
            path = Path(config_path)
            if not Path.exists(path):
                raise FileNotFoundError(f"Config file specified by param `config_path` not found at {path}")
            return path
        elif os.getenv("DEER_FLOW_CONFIG_PATH"):
            path = Path(os.getenv("DEER_FLOW_CONFIG_PATH"))
            if not Path.exists(path):
                raise FileNotFoundError(f"Config file specified by environment variable `DEER_FLOW_CONFIG_PATH` not found at {path}")
            return path
        else:
            for path in _default_config_candidates():
                if path.exists():
                    return path
            raise FileNotFoundError("`config.yaml` file not found at the default backend or repository root locations")

    @classmethod
    def from_file(cls, config_path: str | None = None) -> Self:
        """Load config from YAML file.

        See `resolve_config_path` for more details.

        Args:
            config_path: Path to the config file.

        Returns:
            AppConfig: The loaded config.
        """
        resolved_path = cls.resolve_config_path(config_path)
        with open(resolved_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        # Check config version before processing
        cls._check_config_version(config_data, resolved_path)

        config_data = cls.resolve_env_variables(config_data)

        # Load extensions config separately (it's in a different file)
        extensions_config = ExtensionsConfig.from_file()
        config_data["extensions"] = extensions_config.model_dump()

        result = cls.model_validate(config_data)
        return result

    @classmethod
    def _check_config_version(cls, config_data: dict, config_path: Path) -> None:
        """Check if the user's config.yaml is outdated compared to config.example.yaml.

        Emits a warning if the user's config_version is lower than the example's.
        Missing config_version is treated as version 0 (pre-versioning).
        """
        try:
            user_version = int(config_data.get("config_version", 0))
        except (TypeError, ValueError):
            user_version = 0

        # Find config.example.yaml by searching config.yaml's directory and its parents
        example_path = None
        search_dir = config_path.parent
        for _ in range(5):  # search up to 5 levels
            candidate = search_dir / "config.example.yaml"
            if candidate.exists():
                example_path = candidate
                break
            parent = search_dir.parent
            if parent == search_dir:
                break
            search_dir = parent
        if example_path is None:
            return

        try:
            with open(example_path, encoding="utf-8") as f:
                example_data = yaml.safe_load(f)
            raw = example_data.get("config_version", 0) if example_data else 0
            try:
                example_version = int(raw)
            except (TypeError, ValueError):
                example_version = 0
        except Exception:
            return

        if user_version < example_version:
            logger.warning(
                "Your config.yaml (version %d) is outdated — the latest version is %d. Run `make config-upgrade` to merge new fields into your config.",
                user_version,
                example_version,
            )

    @classmethod
    def resolve_env_variables(cls, config: Any) -> Any:
        """Recursively resolve environment variables in the config.

        Environment variables are resolved using the `os.getenv` function. Example: $OPENAI_API_KEY

        Args:
            config: The config to resolve environment variables in.

        Returns:
            The config with environment variables resolved.
        """
        if isinstance(config, str):
            if config.startswith("$"):
                env_value = os.getenv(config[1:])
                if env_value is None:
                    raise ValueError(f"Environment variable {config[1:]} not found for config value {config}")
                return env_value
            return config
        elif isinstance(config, dict):
            return {k: cls.resolve_env_variables(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [cls.resolve_env_variables(item) for item in config]
        return config

    def get_model_config(self, name: str) -> ModelConfig | None:
        """Get the model config by name.

        Args:
            name: The name of the model to get the config for.

        Returns:
            The model config if found, otherwise None.
        """
        return next((model for model in self.models if model.name == name), None)

    def get_tool_config(self, name: str) -> ToolConfig | None:
        """Get the tool config by name.

        Args:
            name: The name of the tool to get the config for.

        Returns:
            The tool config if found, otherwise None.
        """
        return next((tool for tool in self.tools if tool.name == name), None)

    def get_tool_group_config(self, name: str) -> ToolGroupConfig | None:
        """Get the tool group config by name.

        Args:
            name: The name of the tool group to get the config for.

        Returns:
            The tool group config if found, otherwise None.
        """
        return next((group for group in self.tool_groups if group.name == name), None)

    # -- Lifecycle (process-global + per-context override) --
    #
    # _global is a plain class variable. Assignment is atomic under the GIL
    # (single pointer swap), so no lock is needed for the current read/write
    # pattern. If this ever changes to read-modify-write, add a threading.Lock.

    _global: ClassVar[AppConfig | None] = None
    _override: ClassVar[ContextVar[AppConfig]] = ContextVar("deerflow_app_config_override")

    @classmethod
    def init(cls, config: AppConfig) -> None:
        """Set the process-global AppConfig. Visible to all subsequent requests."""
        cls._global = config

    @classmethod
    def set_override(cls, config: AppConfig) -> Token[AppConfig]:
        """Set a per-context override. Returns a token for reset_override().

        Use this in DeerFlowClient or test fixtures to scope a config to the
        current async context without polluting the process-global value.
        """
        return cls._override.set(config)

    @classmethod
    def reset_override(cls, token: Token[AppConfig]) -> None:
        """Restore the override to its previous value."""
        cls._override.reset(token)

    @classmethod
    def current(cls) -> AppConfig:
        """Get the current AppConfig.

        Priority: per-context override > process-global > auto-load from file.

        The auto-load fallback exists for backward compatibility. Prefer calling
        ``AppConfig.init()`` explicitly at process startup so that config errors
        surface early rather than at an arbitrary first-use call site.
        """
        try:
            return cls._override.get()
        except LookupError:
            pass
        if cls._global is not None:
            return cls._global
        logger.warning(
            "AppConfig.current() called before init(); auto-loading from file. "
            "Call AppConfig.init() at process startup to surface config errors early."
        )
        config = cls.from_file()
        cls._global = config
        return config
