#!/usr/bin/env python3
"""Headless view-contract test for Emma Studio.

Uses Chrome headless with --dump-dom on each hash URL and asserts the
rendered DOM contains the view's key element ids.
"""
import json
import os
import subprocess
import sys

VIEWS = {
    "facetime": ["ftMsg", "ftCall", "ftMute", "ftHangup", "engineBadge", "vrmCanvas", "chatLog"],
    "call": ["micBtn2", "handsfreeBtn", "hangupBtn", "callMsg", "callStatus", "chatLog", "txToggle"],
    "videos": ["vidTopic", "vidMode", "vidGo", "vidGrid"],
    "personas": ["pPrompt", "pLang", "pVoiceProfile", "npName"],
    "facts": ["factNew", "factQ", "factList"],
}

BASE = os.environ.get("STUDIO_BASE_URL", "https://127.0.0.1:8890/#")
failures = []

for view, ids in VIEWS.items():
    dom = subprocess.run(
        ["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
         "--ignore-certificate-errors", "--virtual-time-budget=5000", "--dump-dom", BASE + view],
        capture_output=True, text=True, timeout=60,
    ).stdout
    missing = [i for i in ids if f'id="{i}"' not in dom]
    status = "OK" if not missing else f"MISSING {missing}"
    print(f"{view:10s} {status}")
    if missing:
        failures.append((view, missing))

if failures:
    print("\nFAILED:", failures)
    sys.exit(1)
print("\nAll views render their contract elements.")
