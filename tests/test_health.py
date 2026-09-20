"""Tests for src/ai_env/health.py."""
from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pytest

from ai_env.health import HEALTH_ENDPOINTS, check_url, run_health_check


class TestCheckUrl:
    def test_returns_ok_when_urlopen_succeeds(self):
        fake_response = type("R", (), {
            "status": 200,
            "read": lambda self: b'{"ok":true}',
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: False,
        })()
        with patch("ai_env.health.urllib.request.urlopen", return_value=fake_response):
            result = check_url("svc", "http://example.invalid")
        assert result["ok"] == "true"
        assert result["status"] == "200"
        assert "body" in result
        assert result["name"] == "svc"

    def test_returns_error_when_urlopen_raises(self):
        with patch(
            "ai_env.health.urllib.request.urlopen",
            side_effect=urllib.error.URLError("nope"),
        ):
            result = check_url("svc", "http://example.invalid")
        assert result["ok"] == "false"
        assert "error" in result
        assert "nope" in result["error"]


class TestRunHealthCheck:
    def test_aggregates_results(self):
        # Patch the underlying check_url to return deterministic values
        with patch("ai_env.health.check_url", return_value={"ok": "true", "status": "200"}):
            results = run_health_check()
        assert set(results.keys()) == set(HEALTH_ENDPOINTS.keys())
        for value in results.values():
            assert value["ok"] == "true"

    def test_marks_failures(self):
        with patch("ai_env.health.check_url", return_value={"ok": "false", "error": "x"}):
            results = run_health_check()
        for value in results.values():
            assert value["ok"] == "false"
            assert value["error"] == "x"
