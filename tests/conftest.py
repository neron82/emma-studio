"""Shared test guards.

A few tests exercise fact retrieval against the real local bge-m3 embedding
service; they skip when it is not running so a fresh checkout stays green.
"""
from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest

from studio import config


def _embed_service_up() -> bool:
    parsed = urlparse(config.EMBED_URL)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=1.0):
            return True
    except OSError:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_embed: requires the local bge-m3 embedding service")


def pytest_collection_modifyitems(config, items):
    if _embed_service_up():
        return
    skip = pytest.mark.skip(reason="requires the local bge-m3 embedding service")
    for item in items:
        if "needs_embed" in item.keywords:
            item.add_marker(skip)
