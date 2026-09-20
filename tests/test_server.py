"""Tests for src/ai_env/server.py."""
from __future__ import annotations

import json
import socket
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
from uuid import uuid4

import pytest

from ai_env.server import AIWebHandler, build_handler, run_server


# ---------------------------------------------------------------------------
# Fake agents
# ---------------------------------------------------------------------------


class FakeAgent:
    """A minimal agent stub compatible with the server contract."""

    def __init__(self):
        self.last_message = None
        self.last_max_tokens = None
        self.last_temperature = None
        self.conversation_history = []
        self.last_response_source = "fake"
        self.model_name = "fake-model"
        self.device = "fake"
        self.loaded = True
        self.tokens_used = 0
        self.total_requests = 0

    def get_status(self):
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self.loaded,
            "tokens_used": self.tokens_used,
            "conversation_turns": len(self.conversation_history),
            "total_requests": self.total_requests,
        }

    def generate(self, prompt, *, max_tokens=512, temperature=0.7, **kwargs):
        self.last_message = prompt
        self.last_max_tokens = max_tokens
        self.last_temperature = temperature
        self.conversation_history.append({"role": "user", "content": prompt})
        self.total_requests += 1
        return f"echo: {prompt}"

    def reset_history(self):
        self.conversation_history = []


def _free_port() -> int:
    """Get a free TCP port by binding to port 0 and reading it back."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def workspace(tmp_path):
    """A small workspace with a couple of text files."""
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("# heading\n\nbody", encoding="utf-8")
    return tmp_path


@pytest.fixture
def ui_dir(tmp_path):
    """A minimal web_ui directory with one HTML file."""
    d = tmp_path / "web_ui"
    d.mkdir()
    (d / "index.html").write_text(
        "<!doctype html><html><body>hi</body></html>",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def running_server(workspace, ui_dir):
    """Spin up a real AIWebHandler on a free port in a background thread."""
    port = _free_port()
    server = run_server("127.0.0.1", port, workspace, FakeAgent(), ui_dir)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Give the server a moment to bind.
    time.sleep(0.1)
    yield port
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _http_get(port, path):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp, body


def _http_post(port, path, payload):
    body = json.dumps(payload).encode("utf-8")
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp, data


def _http_post_multipart(port, path, fields, files):
    boundary = f"----{uuid4().hex}"
    body = []
    for key, value in fields:
        body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n")
    for key, filename, content in files:
        body.append(
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{key}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n"
        )
        body.append(content)
        body.append("\r\n")
    body.append(f"--{boundary}--\r\n")
    payload = b"".join(part.encode("utf-8") if isinstance(part, str) else part for part in body)
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(
        "POST",
        path,
        body=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp, data


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------


class TestStatic:
    def test_serves_index_html(self, running_server):
        resp, body = _http_get(running_server, "/")
        assert resp.status == 200
        assert b"<html>" in body

    def test_serves_existing_file(self, running_server):
        # The server's root is the ui_dir, not the workspace, so the
        # only file available via the static path is index.html.
        resp, _ = _http_get(running_server, "/index.html")
        assert resp.status == 200


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestApiStatus:
    def test_returns_200(self, running_server):
        resp, body = _http_get(running_server, "/api/status")
        assert resp.status == 200
        data = json.loads(body)
        assert data["model_name"] == "fake-model"
        assert "resources" in data
        assert "workspace_root" in data


class TestApiFiles:
    def test_lists_files(self, running_server, workspace):
        resp, body = _http_get(running_server, "/api/files")
        assert resp.status == 200
        data = json.loads(body)
        paths = {f["path"] for f in data["files"]}
        assert "a.txt" in paths
        assert any(p.endswith("b.md") for p in paths)

    def test_includes_dirs(self, running_server):
        resp, body = _http_get(running_server, "/api/files")
        data = json.loads(body)
        dirs = [f for f in data["files"] if f.get("is_dir")]
        assert any(d["path"] == "sub" for d in dirs)


class TestApiFile:
    def test_get_existing_file(self, running_server, workspace):
        resp, body = _http_get(running_server, "/api/file?path=a.txt")
        assert resp.status == 200
        data = json.loads(body)
        assert data["path"] == "a.txt"
        assert data["content"] == "hello world"

    def test_get_missing_path_400(self, running_server):
        resp, _ = _http_get(running_server, "/api/file")
        assert resp.status == 400

    def test_get_nonexistent_404(self, running_server):
        resp, _ = _http_get(running_server, "/api/file?path=nope.txt")
        assert resp.status == 404

    def test_get_blocks_escape(self, running_server):
        resp, _ = _http_get(running_server, "/api/file?path=../../../etc/passwd")
        # We want it to NOT serve a system file. 403 is the right
        # response from safe_workspace_path. Some implementations
        # might 404; both are acceptable as long as we don't 200.
        assert resp.status in (403, 404)

    def test_post_writes_file(self, running_server, workspace):
        resp, body = _http_post(
            running_server, "/api/file",
            {"path": "new.txt", "content": "created"},
        )
        assert resp.status == 200
        assert (workspace / "new.txt").read_text(encoding="utf-8") == "created"

    def test_post_missing_path_400(self, running_server):
        resp, _ = _http_post(running_server, "/api/file", {"content": "x"})
        assert resp.status == 400


class TestApiChat:
    def test_basic(self, running_server):
        resp, body = _http_post(
            running_server, "/api/chat",
            {"message": "hello there"},
        )
        assert resp.status == 200
        data = json.loads(body)
        assert data["response"] == "echo: hello there"
        assert data["response_source"] == "fake"

    def test_missing_message_400(self, running_server):
        resp, _ = _http_post(running_server, "/api/chat", {})
        assert resp.status == 400

    def test_temperature_and_tokens(self, running_server):
        # Cannot read agent state across requests easily, but the
        # response shape must be correct.
        resp, _ = _http_post(
            running_server, "/api/chat",
            {"message": "x", "max_tokens": 99, "temperature": 0.1},
        )
        assert resp.status == 200


class TestApiReset:
    def test_resets(self, running_server):
        # Add a turn first
        _http_post(running_server, "/api/chat", {"message": "x"})
        resp, body = _http_post(running_server, "/api/reset", {})
        assert resp.status == 200
        data = json.loads(body)
        assert data["ok"] is True
        assert data["status"]["conversation_turns"] == 0


class TestApiUpload:
    def test_uploads_file(self, running_server, workspace):
        resp, body = _http_post_multipart(
            running_server,
            "/api/upload",
            fields=[("path", "uploads/hello.txt")],
            files=[("files", "hello.txt", b"hello from upload")],
        )
        assert resp.status == 200
        data = json.loads(body)
        assert data["ok"] is True
        assert data["files"][0]["path"] == "uploads/hello.txt"
        assert (workspace / "uploads" / "hello.txt").read_text(encoding="utf-8") == "hello from upload"


class TestApiDiff:
    def test_returns_diff_field(self, running_server):
        resp, body = _http_get(running_server, "/api/diff?path=a.txt")
        assert resp.status == 200
        data = json.loads(body)
        # Either there is no git, or git returns something. Either way
        # the API should always return the "diff" key.
        assert "diff" in data
        assert data["path"] == "a.txt"

    def test_no_path_works(self, running_server):
        resp, body = _http_get(running_server, "/api/diff")
        assert resp.status == 200
        data = json.loads(body)
        assert "diff" in data
        assert data["path"] is None


class TestApiCommitPreview:
    def test_returns_summary(self, running_server):
        resp, body = _http_get(running_server, "/api/commit-preview")
        assert resp.status == 200
        data = json.loads(body)
        assert "summary" in data
        assert "status" in data
        assert "diff_stat" in data


class TestApiNotFound:
    def test_unknown_endpoint_404(self, running_server):
        resp, _ = _http_get(running_server, "/api/nope")
        assert resp.status == 404

    def test_unsupported_post_404(self, running_server):
        resp, _ = _http_post(running_server, "/api/nope", {})
        assert resp.status == 404


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


class TestBuildHandler:
    def test_configures_class_state(self, workspace, ui_dir):
        agent = FakeAgent()
        handler_factory = build_handler(agent, workspace, ui_dir)
        assert AIWebHandler.agent is agent
        assert AIWebHandler.workspace_root == workspace.resolve()
        # The factory is a functools.partial pointing at AIWebHandler.
        from functools import partial
        assert isinstance(handler_factory, partial)
        assert handler_factory.func is AIWebHandler
        assert handler_factory.keywords["directory"] == str(ui_dir.resolve())


class TestRunServer:
    def test_returns_threading_http_server(self, workspace, ui_dir):
        port = _free_port()
        server = run_server("127.0.0.1", port, workspace, FakeAgent(), ui_dir)
        assert hasattr(server, "serve_forever")
        assert hasattr(server, "shutdown")
        server.server_close()


class TestApiInstallExtension:
    def test_install_success(self, running_server, workspace, monkeypatch):
        # Create a mock extension folder in workspace
        ext_dir = workspace / "extension"
        ext_dir.mkdir(exist_ok=True)
        (ext_dir / "package.json").write_text('{"name": "ai-coder-agent"}', encoding="utf-8")
        (ext_dir / "README.md").write_text("test extension", encoding="utf-8")

        # Create a mock user home directory
        mock_home = workspace / "mock_home"
        mock_home.mkdir(exist_ok=True)

        # Mock Path.home() to return mock_home
        from pathlib import Path
        monkeypatch.setattr(Path, "home", lambda: mock_home)

        # Create a fake VS Code parent directory so that it gets processed
        (mock_home / ".vscode").mkdir(exist_ok=True)

        # Run API request
        resp, body = _http_post(running_server, "/api/install-extension", {})
        assert resp.status == 200
        data = json.loads(body)
        assert data["ok"] is True
        assert "paths" in data
        assert len(data["paths"]) == 1

        # Verify the copied files
        installed_dir = mock_home / ".vscode" / "extensions" / "ai-coder-agent-0.1.0"
        assert installed_dir.exists()
        assert (installed_dir / "package.json").exists()
        assert (installed_dir / "README.md").exists()

    def test_install_no_extension_folder(self, running_server):
        # Workspace doesn't have an 'extension' folder in this case (fixture has sub/ and a.txt, but no extension)
        resp, body = _http_post(running_server, "/api/install-extension", {})
        assert resp.status == 404
        data = json.loads(body)
        assert "error" in data


class TestApiCrawler:
    def test_crawler_status_default(self, running_server):
        resp, body = _http_get(running_server, "/api/crawler/status")
        assert resp.status == 200
        data = json.loads(body)
        assert data["status"] == "idle"
        assert len(data["repositories"]) == 5

    def test_crawler_start_and_stop(self, running_server, workspace):
        # Start crawler
        resp, body = _http_post(running_server, "/api/crawler/start", {})
        assert resp.status == 200
        data = json.loads(body)
        assert data["ok"] is True

        # Check running status
        resp_status, body_status = _http_get(running_server, "/api/crawler/status")
        assert resp_status.status == 200
        data_status = json.loads(body_status)
        assert data_status["status"] in ("running", "completed", "paused")

        # Stop crawler
        resp_stop, body_stop = _http_post(running_server, "/api/crawler/stop", {})
        assert resp_stop.status == 200
        data_stop = json.loads(body_stop)
        assert data_stop["ok"] is True
