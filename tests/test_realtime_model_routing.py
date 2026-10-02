"""Exercise provider switching through the real HTTP streaming request path."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event, Thread

import pytest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from studio import config
from studio.agent.models import resolve_endpoint
from studio.speech.chat_completions_language_model import ChatCompletionsApiModelHandler
from studio.deliver.realtime.runtime_config import RuntimeConfig
from studio.speech.chat import Chat, make_user_message
from studio.speech.core.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk


@pytest.fixture
def providers(monkeypatch):
    calls = []
    compaction_entered = Event()
    compaction_release = Event()

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body, dict(self.headers)))
            cloud = self.path == "/go/v1/chat/completions"
            valid = (
                body["model"] == ("cloud-model" if cloud else "local-model")
                and self.headers["Authorization"] == ("Bearer cloud-secret" if cloud else "Bearer none")
                and ("chat_template_kwargs" not in body if cloud else
                     body.get("chat_template_kwargs") == {"enable_thinking": False})
            )
            if not valid:
                self.send_error(400, "Provider-specific request mismatch")
                return
            if not body.get("stream", False):
                compaction_entered.set()
                if not compaction_release.wait(5):
                    self.send_error(504, "Compaction test was not released")
                    return
                summary = "CLOUD SUMMARY" if cloud else "LOCAL SUMMARY"
                response = {
                    "id": "compact_fixture", "object": "chat.completion", "created": 1,
                    "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": json.dumps({
                            "user_summary": summary, "assistant_summary": summary,
                        }),
                    }}],
                }
                payload = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            chunk = {
                "id": "chat_fixture", "object": "chat.completion.chunk", "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "delta": {"content": "CLOUD." if cloud else "LOCAL."},
                             "finish_reason": "stop"}],
            }
            payload = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(config, "LLM_BASE_URL", base + "/local/v1")
    monkeypatch.setattr(config, "LLM_MODEL", "local-model")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("STUDIO_OPENCODE_GO_BASE_URL", base + "/go/v1")
    monkeypatch.setenv("STUDIO_OPENCODE_GO_API_KEY", "cloud-secret")
    try:
        yield calls, compaction_entered, compaction_release
    finally:
        compaction_release.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def _turn(handler, cfg):
    cfg.chat.add_item(make_user_message("Hello"))
    request = GenerateResponseRequest(
        runtime_config=cfg,
        response=RealtimeResponseCreateParams(output_modalities=["text"]),
    )
    output = list(handler.process(request))
    terminal = next(item for item in output if isinstance(item, EndOfResponse))
    assert terminal.error is None
    # The realtime send loop finalizes provisional history once the response is
    # delivered. Reproduce that boundary here so compaction can actually start.
    cfg.chat.finalize_provisional_generation(request.response_key)
    return "".join(item.text for item in output if isinstance(item, LLMResponseChunk))


def test_cloud_then_configured_pool_reuse_does_not_leak_route_or_session(providers):
    calls, _, _ = providers
    # A non-loopback configured endpoint with no configured/ambient key cannot
    # even construct an OpenAI client. Cloud selection must leave it unused.
    handler = ChatCompletionsApiModelHandler(Event(), Queue(), Queue(), setup_kwargs={
        "base_url": "https://configured.invalid/v1", "model_name": "unused",
        "warmup_on_setup": False, "disable_thinking": True,
    })
    try:
        cloud = RuntimeConfig()
        cloud.bind_llm(resolve_endpoint({"llm_provider": "opencode-go", "llm_model": "cloud-model"}),
                       "call-cloud-one")
        assert _turn(handler, cloud) == "CLOUD."
        assert _turn(handler, cloud) == "CLOUD."
        assert "cloud-secret" not in repr(cloud)
        assert "cloud-secret" not in repr(cloud.model_dump())
        assert "cloud-secret" not in cloud.session.model_dump_json()
        assert calls[0][2]["x-opencode-session"] == calls[1][2]["x-opencode-session"] == "call-cloud-one"
        assert calls[0][2]["User-Agent"] == "emma-studio/1.0"

        # Switching persona on an existing connection changes the endpoint too.
        cloud.bind_llm(resolve_endpoint())
        assert _turn(handler, cloud) == "LOCAL."
        assert calls[-1][0] == "/local/v1/chat/completions"
        assert "x-opencode-session" not in calls[-1][2]

        handler.on_session_end()
        next_cloud = RuntimeConfig()
        next_cloud.bind_llm(resolve_endpoint({"llm_provider": "opencode-go", "llm_model": "cloud-model"}),
                            "call-cloud-two")
        assert _turn(handler, next_cloud) == "CLOUD."
        assert calls[-1][2]["x-opencode-session"] == "call-cloud-two"
    finally:
        handler.cleanup()


def test_active_cloud_compaction_keeps_original_route_after_pool_session_end(providers):
    calls, entered, release = providers
    handler = ChatCompletionsApiModelHandler(Event(), Queue(), Queue(), setup_kwargs={
        "base_url": "https://configured.invalid/v1", "model_name": "unused",
        "warmup_on_setup": False, "disable_thinking": True, "compact_history": True,
    })
    cloud = RuntimeConfig(chat=Chat(2))
    cloud.bind_llm(resolve_endpoint({"llm_provider": "opencode-go", "llm_model": "cloud-model"}),
                   "call-compacting")
    try:
        for _ in range(3):
            assert _turn(handler, cloud) == "CLOUD."
        assert entered.wait(2)
        handler.on_session_end()
        local = RuntimeConfig()
        local.bind_llm(resolve_endpoint(), "call-local")
        assert _turn(handler, local) == "LOCAL."
        release.set()
        cloud.chat._compact_thread.join(timeout=2)
        messages = cloud.chat.to_transformers_chat()
        assert messages[0]["content"] == "CLOUD SUMMARY"
        assert messages[1]["content"] == "CLOUD SUMMARY"
        compact_call = next(call for call in calls if not call[1].get("stream", False))
        assert compact_call[0] == "/go/v1/chat/completions"
        assert compact_call[2]["x-opencode-session"] == "call-compacting"
    finally:
        release.set()
        if cloud.chat._compact_thread is not None:
            cloud.chat._compact_thread.join(timeout=2)
        handler.cleanup()
