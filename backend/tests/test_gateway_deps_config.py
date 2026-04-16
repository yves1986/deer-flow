"""Tests for the FastAPI get_config dependency.

Phase 2 step 1: introduces the new explicit-config primitive that
resolves ``AppConfig`` from ``request.app.state.config``. This coexists
with the existing ``AppConfig.current()`` process-global during the
migration; it becomes the sole mechanism after Phase 2 task P2-10.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.gateway.deps import get_config
from deerflow.config.app_config import AppConfig
from deerflow.config.sandbox_config import SandboxConfig


def test_get_config_returns_app_state_config():
    """get_config returns the AppConfig stored on app.state.config."""
    app = FastAPI()
    cfg = AppConfig(sandbox=SandboxConfig(use="test"))
    app.state.config = cfg

    @app.get("/probe")
    def probe(c: AppConfig = Depends(get_config)):
        # Identity check: FastAPI must hand us the exact object from app.state
        return {"same_identity": c is cfg, "log_level": c.log_level}

    client = TestClient(app)
    response = client.get("/probe")

    assert response.status_code == 200
    body = response.json()
    assert body["same_identity"] is True
    assert body["log_level"] == "info"


def test_get_config_reads_updated_app_state():
    """When app.state.config is swapped (config reload), get_config sees the new value."""
    app = FastAPI()
    original = AppConfig(sandbox=SandboxConfig(use="test"), log_level="info")
    replacement = original.model_copy(update={"log_level": "debug"})

    app.state.config = original

    @app.get("/log-level")
    def log_level(c: AppConfig = Depends(get_config)):
        return {"level": c.log_level}

    client = TestClient(app)
    assert client.get("/log-level").json() == {"level": "info"}

    # Simulate config reload (PUT /mcp/config, etc.)
    app.state.config = replacement
    assert client.get("/log-level").json() == {"level": "debug"}
