"""Provider selection, credential isolation and legacy persona migration."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio import config
from studio.agent import models
from studio.core.store import Store


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("STUDIO_OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.delenv("STUDIO_OPENCODE_GO_BASE_URL", raising=False)
    return tmp_path


def write_credentials(home, token="fake-hermes-token", base="https://hermes.invalid/v1"):
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"opencode-go": [
        {"access_token": token, "base_url": base, "auth_type": "api_key", "priority": 0}
    ]}}), encoding="utf-8")


def test_environment_overrides_hermes_and_persona_url_never_receives_go_key(hermes_home, monkeypatch):
    write_credentials(hermes_home)
    monkeypatch.setenv("STUDIO_OPENCODE_GO_API_KEY", "fake-environment-token")
    monkeypatch.setenv("STUDIO_OPENCODE_GO_BASE_URL", "https://server-selected.invalid/v1/")
    endpoint = models.resolve_endpoint({"llm_provider": "opencode-go",
                                       "llm_base_url": "https://attacker.invalid/v1",
                                       "llm_model": "not-in-catalog"})
    assert endpoint.base_url == "https://server-selected.invalid/v1"
    assert endpoint.model == "not-in-catalog"
    assert endpoint.api_key == "fake-environment-token"
    assert "fake-environment-token" not in repr(endpoint)
    assert endpoint.headers("call-42")["x-opencode-session"] == "call-42"
    assert endpoint.headers()["x-opencode-session"] == endpoint.headers()["x-opencode-session"]


def test_hermes_pool_precedes_dotenv_and_dotenv_is_fallback(hermes_home):
    (hermes_home / ".env").write_text("OPENCODE_GO_API_KEY=fake-dotenv-token\n", encoding="utf-8")
    write_credentials(hermes_home)
    selection = {"llm_provider": "opencode-go", "llm_model": "chosen"}
    assert models.resolve_endpoint(selection).api_key == "fake-hermes-token"
    (hermes_home / "auth.json").unlink()
    endpoint = models.resolve_endpoint(selection)
    assert endpoint.api_key == "fake-dotenv-token"
    assert endpoint.base_url == models.GO_BASE_URL


def test_configured_inheritance_and_override_do_not_use_go_credentials(hermes_home, monkeypatch):
    write_credentials(hermes_home)
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://configured.invalid/v1")
    monkeypatch.setattr(config, "LLM_MODEL", "configured-model")
    monkeypatch.setattr(config, "LLM_API_KEY", "fake-configured-token")
    inherited = models.resolve_endpoint()
    assert (inherited.base_url, inherited.model, inherited.api_key) == (
        "http://configured.invalid/v1", "configured-model", "fake-configured-token")
    selected = models.resolve_endpoint({"llm_base_url": "http://local.invalid/v1/", "llm_model": "local-model"})
    assert (selected.base_url, selected.model, selected.api_key) == (
        "http://local.invalid/v1", "local-model", "")
    assert "x-opencode-session" not in selected.headers("call-42")


@pytest.mark.parametrize("override, authenticated", [
    ("https://configured.invalid/other-path", True),
    ("https://CONFIGURED.invalid:443/v2", True),
    ("https://other.invalid/v1", False),
    ("http://configured.invalid/v1", False),
    ("https://configured.invalid:8443/v1", False),
    ("https://configured.invalid.attacker.invalid/v1", False),
])
def test_configured_key_is_bound_to_trusted_origin(monkeypatch, override, authenticated):
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://configured.invalid/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "fake-origin-bound-token")
    endpoint = models.resolve_endpoint({"llm_base_url": override, "llm_model": "selected"})
    if authenticated:
        assert endpoint.headers()["Authorization"] == "Bearer fake-origin-bound-token"
    else:
        assert endpoint.api_key == ""
        assert "Authorization" not in endpoint.headers()


@pytest.mark.parametrize("override, expected_key", [
    ("", "fake-ambient-token"),
    ("https://CONFIGURED.invalid:443/other-path", "fake-ambient-token"),
    ("https://other.invalid/v1", ""),
])
def test_ambient_openai_key_fallback_remains_origin_bound(monkeypatch, override, expected_key):
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://configured.invalid/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-ambient-token")
    endpoint = models.resolve_endpoint({"llm_base_url": override})
    assert endpoint.api_key == expected_key
    if not expected_key:
        assert "Authorization" not in endpoint.headers()


def test_explicit_configured_key_precedes_ambient_openai_key(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "fake-explicit-token")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-ambient-token")
    assert models.resolve_endpoint().api_key == "fake-explicit-token"


@pytest.mark.parametrize("configured_key", ["fake-origin-bound-token", ""])
async def test_cross_origin_configured_discovery_cannot_receive_global_key(monkeypatch, configured_key):
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://configured.invalid/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", configured_key)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-ambient-token")
    requests = []

    def respond(request):
        requests.append(request)
        if "Authorization" in request.headers:
            return httpx.Response(403, text="Credentials must not reach this endpoint")
        return httpx.Response(200, json={"data": [{"id": "public-local-model"}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(models.httpx, "AsyncClient", lambda **kwargs:
                        real_client(transport=httpx.MockTransport(respond), **kwargs))
    result = await models.discover_models("configured", "https://other.invalid/v1")
    assert result["models"] == [{"id": "public-local-model", "name": "public-local-model"}]
    assert str(requests[0].url) == "https://other.invalid/v1/models"
    assert "Authorization" not in requests[0].headers


@pytest.mark.parametrize("selection, message", [
    ({"llm_provider": "unknown"}, "Unknown model provider"),
    ({"llm_provider": "opencode-go"}, "Choose an OpenCode Go model"),
    ({"llm_provider": "opencode-go", "llm_model": "chosen"}, "server credential"),
    ({"llm_model": None}, "llm_model must be a string"),
])
def test_invalid_selection_is_actionable_without_fallback(hermes_home, selection, message):
    with pytest.raises(models.ModelConfigurationError, match=message):
        models.resolve_endpoint(selection)


def test_corrupt_credentials_error_does_not_echo_file_content(hermes_home):
    (hermes_home / "auth.json").write_text('{"access_token":"fake-secret-content", broken', encoding="utf-8")
    with pytest.raises(models.ModelConfigurationError) as error:
        models.resolve_endpoint({"llm_provider": "opencode-go", "llm_model": "chosen"})
    assert "Cannot read Hermes credentials" in str(error.value)
    assert "fake-secret-content" not in str(error.value)


async def test_authenticated_discovery_deduplicates_and_ignores_persona_cloud_url(hermes_home, monkeypatch):
    write_credentials(hermes_home)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [
            {"id": "zeta"}, {"id": "alpha", "name": "Alpha"}, {"id": "alpha"},
            {"id": ""}, {"other": "invalid"}
        ]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(models.httpx, "AsyncClient", lambda **kwargs:
                        real_client(transport=httpx.MockTransport(respond), **kwargs))
    result = await models.discover_models("opencode-go", "https://attacker.invalid/v1")
    assert result == {"provider": "opencode-go", "base_url": "https://hermes.invalid/v1",
                      "models": [{"id": "alpha", "name": "Alpha"}, {"id": "zeta", "name": "zeta"}],
                      "default_model": ""}
    request = requests[0]
    assert str(request.url) == "https://hermes.invalid/v1/models"
    assert request.headers["Authorization"] == "Bearer fake-hermes-token"
    assert request.headers["User-Agent"] == "emma-studio/1.0"
    assert request.headers["x-opencode-session"]
    assert "fake-hermes-token" not in json.dumps(result)


@pytest.mark.parametrize("status", [401, 403, 500])
async def test_discovery_failure_never_returns_upstream_secret_body(hermes_home, monkeypatch, status):
    write_credentials(hermes_home)
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text="fake-sensitive-upstream-content"))
    monkeypatch.setattr(models.httpx, "AsyncClient", lambda **kwargs:
                        real_client(transport=transport, **kwargs))
    with pytest.raises(models.ModelDiscoveryError) as error:
        await models.discover_models("opencode-go")
    assert "fake-sensitive-upstream-content" not in str(error.value)
    assert "fake-hermes-token" not in str(error.value)


def test_legacy_persona_migration_and_cloud_selection_roundtrip(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE personas (
            id TEXT PRIMARY KEY, name TEXT NOT NULL,
            personality_prompt TEXT NOT NULL DEFAULT '', ref_image TEXT, voice_sample TEXT,
            voice_profile TEXT, vrm_model TEXT, language TEXT NOT NULL DEFAULT 'German',
            created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
        INSERT INTO personas(id, name, created_at, updated_at) VALUES ('legacy', 'Emma', 1, 1);
    """)
    conn.close()
    store = Store(db)
    try:
        persona = store.get_persona("legacy")
        assert (persona["llm_provider"], persona["llm_base_url"], persona["llm_model"]) == ("configured", "", "")
        store.update_persona("legacy", llm_provider="configured", llm_base_url="http://local.invalid/v1", llm_model="local-model")
        store.update_persona("legacy", llm_provider="opencode-go", llm_model="cloud-model")
        created = store.create_persona("Cloud", llm_provider="opencode-go", llm_base_url="http://local.invalid/v1", llm_model="other-model")
        assert created["llm_base_url"] == ""
    finally:
        store.close()
    reopened = Store(db)
    try:
        persona = reopened.get_persona("legacy")
        assert (persona["llm_provider"], persona["llm_base_url"], persona["llm_model"]) == ("opencode-go", "", "cloud-model")
        reopened.update_persona("legacy", name="Renamed")
        assert reopened.get_persona("legacy")["llm_model"] == "cloud-model"
        assert reopened.get_persona(created["id"])["llm_model"] == "other-model"
    finally:
        reopened.close()
