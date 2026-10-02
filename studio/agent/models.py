"""Server-side model selection and OpenAI-compatible model discovery."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

from .. import config

GO_BASE_URL = "https://opencode.ai/zen/go/v1"
_DEFAULT_SESSION_ID = uuid.uuid4().hex


class ModelConfigurationError(ValueError):
    """An actionable, credential-free model selection error."""


class ModelDiscoveryError(RuntimeError):
    """A sanitized failure while contacting a model provider."""


@dataclass(frozen=True)
class ModelEndpoint:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    provider: str

    def headers(self, session_id: str | None = None) -> dict[str, str]:
        headers = {"User-Agent": "emma-studio/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider == "opencode-go":
            headers["x-opencode-session"] = session_id or _DEFAULT_SESSION_ID
        return headers


def _text(value, field_name: str) -> str:
    if not isinstance(value, str):
        raise ModelConfigurationError(f"{field_name} must be a string.")
    return value.strip()


def _base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment)
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ModelConfigurationError("Model base URL must be an HTTP(S) URL without credentials, query or fragment.")
    return value.rstrip("/")


def _origin(base_url: str) -> tuple[str, str, int]:
    parsed = urlsplit(base_url)
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname, port


def _priority(entry: dict) -> float:
    try:
        return float(entry.get("priority", 0))
    except (TypeError, ValueError):
        return 0


def _go_settings() -> tuple[str, str]:
    """Load credentials only on the server; environment overrides Hermes."""
    key = os.getenv("STUDIO_OPENCODE_GO_API_KEY", "").strip()
    base = os.getenv("STUDIO_OPENCODE_GO_BASE_URL", "").strip()
    home = Path(os.getenv("HERMES_HOME", "~/.hermes")).expanduser()
    entry: dict = {}
    auth_error = False
    try:
        auth = json.loads((home / "auth.json").read_text(encoding="utf-8"))
        pool = auth.get("credential_pool", {}).get("opencode-go", [])
        if isinstance(pool, list):
            candidates = [item for item in pool if isinstance(item, dict)
                          and isinstance(item.get("access_token"), str)
                          and item["access_token"].strip()]
            if candidates:
                entry = min(candidates, key=_priority)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, AttributeError, TypeError):
        auth_error = True
    if not key:
        key = entry.get("access_token", "").strip()
    if not key:
        try:
            key = (dotenv_values(home / ".env").get("OPENCODE_GO_API_KEY") or "").strip()
        except (OSError, ValueError):
            pass
    if not key:
        if auth_error:
            raise ModelConfigurationError("Cannot read Hermes credentials. Repair HERMES_HOME/auth.json or set STUDIO_OPENCODE_GO_API_KEY.")
        raise ModelConfigurationError("OpenCode Go needs a server credential: set STUDIO_OPENCODE_GO_API_KEY or sign in through Hermes.")
    if not base:
        stored_base = entry.get("base_url")
        base = stored_base.strip() if isinstance(stored_base, str) else ""
    return _base_url(base or GO_BASE_URL), key


def _selection(persona: dict | None) -> tuple[str, str, str]:
    persona = persona or {}
    provider = _text(persona.get("llm_provider", "configured"), "llm_provider")
    if provider not in {"configured", "opencode-go"}:
        raise ModelConfigurationError("Unknown model provider. Choose configured or opencode-go.")
    base = _text(persona.get("llm_base_url", ""), "llm_base_url")
    model = _text(persona.get("llm_model", ""), "llm_model")
    return provider, base, model


def resolve_endpoint(persona: dict | None = None) -> ModelEndpoint:
    provider, base, model = _selection(persona)
    if provider == "configured":
        configured_base = _base_url(config.LLM_BASE_URL)
        selected_base = _base_url(base) if base else configured_base
        # Persona overrides may select another server, never another credential recipient.
        configured_key = config.LLM_API_KEY or os.getenv("OPENAI_API_KEY", "")
        api_key = configured_key if _origin(selected_base) == _origin(configured_base) else ""
        return ModelEndpoint(selected_base, model or config.LLM_MODEL, api_key, provider)
    if not model:
        raise ModelConfigurationError("Choose an OpenCode Go model before saving this persona.")
    # A persona URL is deliberately ignored here: it must never receive Go secrets.
    base, key = _go_settings()
    return ModelEndpoint(base, model, key, provider)


def validate_selection(persona: dict) -> dict[str, str]:
    """Validate a merged selection and normalize its persisted fields."""
    provider, base, model = _selection(persona)
    resolve_endpoint(persona)
    return {"llm_provider": provider,
            "llm_base_url": _base_url(base) if provider == "configured" and base else "",
            "llm_model": model}


async def discover_models(provider: str, base_url: str = "") -> dict:
    """Return public model metadata, never credentials or upstream error bodies."""
    selected, base, _ = _selection({"llm_provider": provider, "llm_base_url": base_url})
    if selected == "opencode-go":
        base, key = _go_settings()
        endpoint = ModelEndpoint(base, "", key, selected)
        default_model = ""
    else:
        endpoint = resolve_endpoint({"llm_provider": selected, "llm_base_url": base})
        default_model = endpoint.model
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
            response = await client.get(f"{endpoint.base_url}/models", headers=endpoint.headers())
    except httpx.HTTPError:
        raise ModelDiscoveryError("Model discovery could not reach the provider. Check the server connection and base URL.") from None
    if response.status_code != 200:
        if response.status_code in {401, 403}:
            raise ModelDiscoveryError("Model discovery authentication failed. Check the provider credential on the server.")
        raise ModelDiscoveryError(f"Model discovery failed (HTTP {response.status_code}).")
    try:
        payload = response.json()
        rows = payload["data"]
        if not isinstance(rows, list):
            raise ValueError
        models = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_id = row.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            model_id = model_id.strip()
            name = row.get("name")
            models.setdefault(model_id, {"id": model_id,
                                         "name": name if isinstance(name, str) and name else model_id})
    except (ValueError, KeyError, TypeError):
        raise ModelDiscoveryError("The provider returned an invalid model list; an OpenAI-compatible /models endpoint is required.") from None
    return {"provider": endpoint.provider, "base_url": endpoint.base_url,
            "models": [models[key] for key in sorted(models)], "default_model": default_model}
