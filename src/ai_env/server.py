"""HTTP request handler and server bootstrap for the AI coding agent Web UI.

This module is the single source of truth for:

  * :class:`AIWebHandler` - the ``SimpleHTTPRequestHandler`` subclass that
    serves the bundled static files and the JSON API surface
    (``/api/status``, ``/api/files``, ``/api/file``, ``/api/chat``,
    ``/api/reset``, ``/api/diff``, ``/api/commit-preview``).
  * :func:`build_handler` - returns a ``functools.partial`` suitable
    for :class:`http.server.ThreadingHTTPServer`.
  * :func:`run_server` - one-call convenience that wires up the
    handler with a workspace root and an agent, then blocks in
    ``serve_forever``.

The handler is intentionally class-level: the agent and workspace root
are set on the class (not on an instance) so that ``SimpleHTTPRequestHandler``
can be instantiated by the server with the ``directory=...`` keyword.
The class-level attributes are configured once in
:func:`run_server` / :func:`build_handler` before the first request.

Dependency graph (after the modular refactor):

    web.py   --(imports)-->  server.py
    server.py --(imports)-->  workspace.py  (git, disk, system_snapshot)
    server.py --(imports)-->  data.py       (only if you add /api/data
                                            endpoints in the future)
"""
from __future__ import annotations

import functools
import json
import threading
from email import policy
from email.parser import BytesParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

from .workspace import (
    disk_usage,
    git_diff,
    preview_commit,
    safe_workspace_path,
    system_snapshot,
    write_workspace_file,
    should_ignore,
)

CRAWLER_WORKER = None


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class AIWebHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the Web UI: static files + JSON API.

    Class-level attributes (set by :func:`build_handler` / :func:`run_server`
    before the server starts):

      * ``agent`` - any object exposing ``get_status()``, ``generate()``,
        ``reset_history()``, ``last_response_source`` - typically an
        :class:`ai_env.ollama_agent.OllamaCodingAgent` or
        :class:`ai_env.agent.CodingAgent`.
      * ``workspace_root`` - :class:`pathlib.Path` of the directory the
        user is browsing/editing.
    """

    agent: Any = None
    workspace_root: Optional[Path] = None
    _agent_lock = threading.Lock()

    # Silence the default per-request stderr access log.
    def log_message(self, format: str, *args: object) -> None:  # pragma: no cover
        return

    # ----- low-level response helpers -----

    def send_json(self, payload: dict, status: int = 200) -> None:
        """Serialize ``payload`` as JSON and write the response."""
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict:
        """Read a JSON request body. Returns ``{}`` on empty/malformed input."""
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    # ----- request dispatch -----

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            return self._route_api_get(parsed)
        if parsed.path in ("/instructions.html", "/instructions"):
            instructions_path = Path(__file__).resolve().parents[2] / "instructions.html"
            if instructions_path.exists():
                try:
                    content = instructions_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return
                except Exception:  # pragma: no cover
                    pass
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            return self._route_api_post(parsed)
        self.send_json({"error": "Unsupported POST path"}, status=404)

    # ----- API surface -----

    def _route_api_get(self, parsed) -> None:
        path = parsed.path
        if path == "/api/status":
            self.api_status()
        elif path == "/api/files":
            self.api_files()
        elif path == "/api/file":
            self.api_file_get(parsed)
        elif path == "/api/diff":
            self.api_diff(parsed)
        elif path == "/api/commit-preview":
            self.api_commit_preview()
        elif path == "/api/crawler/status":
            self.api_crawler_status()
        elif path == "/api/sessions":
            self.api_sessions_get(parsed)
        else:
            self.send_json({"error": "Endpoint not found"}, status=404)

    def _route_api_post(self, parsed) -> None:
        path = parsed.path
        if path == "/api/chat":
            self.api_chat()
        elif path == "/api/reset":
            self.api_reset()
        elif path == "/api/file":
            self.api_file_post()
        elif path == "/api/complete":
            self.api_complete()
        elif path == "/api/upload":
            self.api_upload()
        elif path == "/api/shutdown":
            self.api_shutdown()
        elif path == "/api/install-extension":
            self.api_install_extension()
        elif path == "/api/crawler/start":
            self.api_crawler_start()
        elif path == "/api/crawler/stop":
            self.api_crawler_stop()
        elif path == "/api/sessions":
            self.api_sessions_post()
        elif path == "/api/sessions/delete":
            self.api_sessions_delete()
        else:
            self.send_json({"error": "Endpoint not found"}, status=404)

    # ----- concrete handlers -----

    def api_status(self) -> None:
        if self.agent is None or self.workspace_root is None:
            self.send_json({"error": "Agent not initialized"}, status=500)
            return
        with self._agent_lock:
            status = self.agent.get_status()
        status["resources"] = system_snapshot()
        status["resources"].update(disk_usage(self.workspace_root))
        status["workspace_root"] = str(self.workspace_root)
        self.send_json(status)

    def api_files(self) -> None:
        if self.workspace_root is None:
            self.send_json({"error": "Workspace root not configured"}, status=500)
            return
        if not self.workspace_root.exists():
            self.send_json({"files": []})
            return
        files: list[dict] = []
        for item in sorted(self.workspace_root.rglob("*")):
            if should_ignore(item):
                continue
            try:
                rel_path = item.relative_to(self.workspace_root).as_posix()
            except ValueError:
                continue
            files.append({"path": rel_path, "is_dir": item.is_dir()})
            if len(files) >= 5000:
                break
        self.send_json({"files": files})

    def api_file_get(self, parsed) -> None:
        if self.workspace_root is None:
            self.send_json({"error": "Workspace root not configured"}, status=500)
            return
        params = parse_qs(parsed.query)
        path_values = params.get("path", [])
        if not path_values:
            self.send_json({"error": "Missing file path"}, status=400)
            return
        try:
            target = safe_workspace_path(self.workspace_root, path_values[0])
        except PermissionError as exc:
            self.send_json({"error": str(exc)}, status=403)
            return
        if not target.exists() or not target.is_file():
            self.send_json({"error": f"File not found: {path_values[0]}"}, status=404)
            return
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            self.send_json({"error": str(exc)}, status=500)
            return
        self.send_json({"path": path_values[0], "content": content})

    def api_file_post(self) -> None:
        if self.workspace_root is None:
            self.send_json({"error": "Workspace root not configured"}, status=500)
            return
        payload = self._read_json_body()
        path_value = payload.get("path")
        content = payload.get("content")
        if not path_value or content is None:
            self.send_json({"error": "Missing path or content"}, status=400)
            return
        try:
            write_workspace_file(self.workspace_root, path_value, content)
            self.send_json({"ok": True, "path": path_value})
        except PermissionError as exc:
            self.send_json({"error": str(exc)}, status=403)
        except OSError as exc:
            self.send_json({"error": str(exc)}, status=500)

    def api_upload(self) -> None:
        if self.workspace_root is None:
            self.send_json({"error": "Workspace root not configured"}, status=500)
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self.send_json({"error": "Expected multipart/form-data upload"}, status=400)
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            self.send_json({"error": "Missing upload body"}, status=400)
            return

        raw_body = self.rfile.read(length)
        if not raw_body:
            self.send_json({"error": "Missing upload body"}, status=400)
            return

        try:
            parser = BytesParser(policy=policy.default)
            message = parser.parsebytes(
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + raw_body
            )
            boundary = message.get_boundary()
            if not boundary:
                raise ValueError("Multipart boundary not found")
        except Exception as exc:  # noqa: BLE001
            self.send_json({"error": str(exc)}, status=400)
            return

        upload_target = ""
        uploaded_files: list[dict[str, Any]] = []
        for part in message.iter_parts():
            part_name = part.get_param("name", header="content-disposition")
            if part_name == "path":
                upload_target = (part.get_payload(decode=True) or b"").decode("utf-8", errors="replace").strip().replace("\\", "/")
                continue
            if part.get_content_disposition() != "form-data":
                continue
            filename = part.get_filename()
            if not filename:
                continue

            raw_name = Path(filename).name
            target_rel = raw_name
            if upload_target:
                if upload_target.endswith("/"):
                    target_rel = f"{upload_target}{raw_name}"
                elif Path(upload_target).suffix == "" and "/" not in upload_target and "\\" not in upload_target:
                    target_rel = f"{upload_target}/{raw_name}"
                else:
                    target_rel = upload_target

            try:
                target = safe_workspace_path(self.workspace_root, target_rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                content = part.get_payload(decode=True)
                if content is None:
                    content = part.get_payload().encode("utf-8")
                target.write_bytes(content)
                uploaded_files.append({"path": target_rel, "size": len(content)})
            except PermissionError as exc:
                self.send_json({"error": str(exc)}, status=403)
                return
            except OSError as exc:
                self.send_json({"error": str(exc)}, status=500)
                return

        if not uploaded_files:
            self.send_json({"error": "No files were uploaded"}, status=400)
            return
        self.send_json({"ok": True, "files": uploaded_files})

    def api_chat(self) -> None:
        if self.agent is None or self.workspace_root is None:
            self.send_json({"error": "Agent not initialized or workspace root not configured"}, status=500)
            return
        payload = self._read_json_body()
        message = (payload.get("message") or "").strip()
        if not message:
            self.send_json({"error": "Missing message"}, status=400)
            return
        max_tokens = int(payload.get("max_tokens", 512))
        temperature = float(payload.get("temperature", 0.7))
        session_id = payload.get("session_id")
        
        import time
        import uuid
        sessions = self._read_sessions()
        if session_id and session_id in sessions:
            session_data = sessions[session_id]
        else:
            session_id = str(uuid.uuid4())
            session_data = {
                "id": session_id,
                "title": message[:30] + "..." if len(message) > 30 else message,
                "updated_at": time.time(),
                "messages": []
            }
            sessions[session_id] = session_data

        try:
            with self._agent_lock:
                self.agent.conversation_history = list(session_data.get("messages", []))
                
                response = self.agent.generate(
                    message, max_tokens=max_tokens, temperature=temperature,
                )
                
                session_data["messages"] = self.agent.conversation_history
                if len(session_data["messages"]) <= 2 and session_data.get("title") == "New Chat":
                    session_data["title"] = message[:30] + "..." if len(message) > 30 else message
                session_data["updated_at"] = time.time()
                sessions[session_id] = session_data
                self._write_sessions(sessions)
                
                status = self.agent.get_status()
                source = getattr(self.agent, "last_response_source", None)
            self.send_json({
                "response": response,
                "status": status,
                "response_source": source,
                "session_id": session_id
            })
        except Exception as exc:  # noqa: BLE001 - surface upstream errors to UI
            self.send_json({"error": str(exc)}, status=500)

    # ----- chat sessions helpers -----

    def _get_sessions_file(self) -> Path:
        return self.workspace_root / ".ai_chat_history.json"

    def _read_sessions(self) -> dict:
        sf = self._get_sessions_file()
        if sf.exists():
            try:
                return json.loads(sf.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _write_sessions(self, data: dict) -> None:
        sf = self._get_sessions_file()
        sf.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def api_sessions_get(self, parsed) -> None:
        if self.workspace_root is None:
            self.send_json({"error": "Workspace root not configured"}, status=500)
            return
        params = parse_qs(parsed.query)
        session_id = params.get("id", [None])[0]
        sessions = self._read_sessions()
        
        if session_id:
            if session_id in sessions:
                self.send_json(sessions[session_id])
            else:
                self.send_json({"error": "Session not found"}, status=404)
            return
            
        session_list = []
        for sid, sdata in sessions.items():
            session_list.append({
                "id": sid,
                "title": sdata.get("title", "New Chat"),
                "updated_at": sdata.get("updated_at", 0)
            })
        session_list.sort(key=lambda x: x["updated_at"], reverse=True)
        self.send_json({"sessions": session_list})

    def api_sessions_post(self) -> None:
        if self.workspace_root is None:
            self.send_json({"error": "Workspace root not configured"}, status=500)
            return
        import uuid
        import time
        session_id = str(uuid.uuid4())
        sessions = self._read_sessions()
        sessions[session_id] = {
            "id": session_id,
            "title": "New Chat",
            "updated_at": time.time(),
            "messages": []
        }
        self._write_sessions(sessions)
        self.send_json(sessions[session_id])

    def api_sessions_delete(self) -> None:
        if self.workspace_root is None:
            self.send_json({"error": "Workspace root not configured"}, status=500)
            return
        payload = self._read_json_body()
        session_id = payload.get("id")
        if not session_id:
            self.send_json({"error": "Missing session id"}, status=400)
            return
        sessions = self._read_sessions()
        if session_id in sessions:
            del sessions[session_id]
            self._write_sessions(sessions)
        self.send_json({"ok": True})

    def api_complete(self) -> None:
        if self.agent is None:
            self.send_json({"error": "Agent not initialized"}, status=500)
            return
        payload = self._read_json_body()
        prompt = payload.get("prompt", "")
        if not prompt:
            self.send_json({"error": "Missing prompt"}, status=400)
            return
        max_tokens = int(payload.get("max_tokens", 128))
        temperature = float(payload.get("temperature", 0.2))
        try:
            with self._agent_lock:
                if hasattr(self.agent, "complete"):
                    response = self.agent.complete(prompt, max_tokens=max_tokens, temperature=temperature)
                else:
                    response = self.agent.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            self.send_json({"response": response})
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def api_reset(self) -> None:
        if self.agent is None:
            self.send_json({"error": "Agent not initialized"}, status=500)
            return
        with self._agent_lock:
            self.agent.reset_history()
            status = self.agent.get_status()
        self.send_json({"ok": True, "status": status})

    def api_shutdown(self) -> None:
        import threading
        self.send_json({"ok": True, "message": "Server shutting down..."})
        threading.Thread(target=self.server.shutdown).start()

    def api_install_extension(self) -> None:
        if self.workspace_root is None:
            self.send_json({"error": "Workspace root not configured"}, status=500)
            return

        import shutil
        src_dir = self.workspace_root / "extension"
        if not src_dir.exists():
            self.send_json({"error": "Extension directory not found"}, status=404)
            return

        from pathlib import Path
        home = Path.home()
        ext_dirs = [
            home / ".vscode" / "extensions",
            home / ".vscode-insiders" / "extensions"
        ]

        # Use directories where parent exists (meaning VS Code or VS Code Insiders was installed/run),
        # otherwise fallback to standard VS Code extension directory.
        valid_dirs = [d for d in ext_dirs if d.parent.exists()]
        if not valid_dirs:
            valid_dirs = [ext_dirs[0]]

        installed_paths = []
        error_msg = None

        for d in valid_dirs:
            target_dir = d / "ai-coder-agent-0.1.0"
            try:
                d.mkdir(parents=True, exist_ok=True)
                if target_dir.exists():
                    if target_dir.is_dir():
                        shutil.rmtree(target_dir)
                    else:
                        target_dir.unlink()

                ignore_patterns = shutil.ignore_patterns("node_modules", "src", ".git", "*.ts", ".gitignore")
                shutil.copytree(src_dir, target_dir, ignore=ignore_patterns)
                installed_paths.append(str(target_dir))
            except Exception as e:
                error_msg = str(e)

        if not installed_paths:
            msg = f"VS Code directories not found or write failed. {error_msg or ''}".strip()
            self.send_json({"error": msg}, status=500)
        else:
            self.send_json({
                "ok": True,
                "message": f"Successfully installed extension to {len(installed_paths)} location(s). Please restart or reload VS Code.",
                "paths": installed_paths
            })

    def api_crawler_status(self) -> None:
        if self.workspace_root is None:
            self.send_json({"error": "Workspace root not configured"}, status=500)
            return

        state_file = self.workspace_root / "crawler_state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                self.send_json(state)
                return
            except Exception:
                pass

        # Return default state
        self.send_json({
            "status": "idle",
            "current_index": 0,
            "repositories": [
                "psf/requests",
                "bottlepy/bottle",
                "pallets/click",
                "encode/starlette",
                "pytest-dev/pytest"
            ],
            "completed": [],
            "logs": []
        })

    def api_crawler_start(self) -> None:
        global CRAWLER_WORKER
        if self.workspace_root is None:
            self.send_json({"error": "Workspace root not configured"}, status=500)
            return

        if CRAWLER_WORKER and CRAWLER_WORKER.is_alive():
            self.send_json({"error": "Crawler is already running"}, status=400)
            return

        from .crawler import CrawlerTrainerWorker
        state_file = self.workspace_root / "crawler_state.json"
        checkpoint_dir = self.workspace_root / "checkpoints"

        CRAWLER_WORKER = CrawlerTrainerWorker(self.workspace_root, state_file, checkpoint_dir)
        CRAWLER_WORKER.start()
        self.send_json({"ok": True, "message": "Crawler started"})

    def api_crawler_stop(self) -> None:
        global CRAWLER_WORKER
        if not CRAWLER_WORKER or not CRAWLER_WORKER.is_alive():
            self.send_json({"ok": True, "message": "Crawler is not running"})
            return

        CRAWLER_WORKER.stop_event.set()
        CRAWLER_WORKER.join(timeout=2)
        self.send_json({"ok": True, "message": "Crawler stopped/paused"})

    def api_diff(self, parsed) -> None:
        if self.workspace_root is None:
            self.send_json({"error": "Workspace root not configured"}, status=500)
            return
        params = parse_qs(parsed.query)
        path_values = params.get("path", [])
        target = path_values[0] if path_values else None
        diff_text = git_diff(self.workspace_root, target)
        if not diff_text:
            diff_text = "No git diff available for this file or no changes detected."
        self.send_json({"diff": diff_text, "path": target})

    def api_commit_preview(self) -> None:
        if self.workspace_root is None:
            self.send_json({"error": "Workspace root not configured"}, status=500)
            return
        preview = preview_commit(self.workspace_root)
        self.send_json(preview.to_dict())


# ---------------------------------------------------------------------------
# Server bootstrap helpers
# ---------------------------------------------------------------------------


def build_handler(
    agent: Any,
    workspace_root: Path,
    ui_dir: Path,
) -> Callable[..., AIWebHandler]:
    """Configure the class-level state and return a handler factory.

    The returned object is suitable as the second argument to
    :class:`http.server.ThreadingHTTPServer`.
    """
    workspace_root = Path(workspace_root).resolve()
    ui_dir = Path(ui_dir).resolve()
    AIWebHandler.agent = agent
    AIWebHandler.workspace_root = workspace_root
    return functools.partial(AIWebHandler, directory=str(ui_dir))


def run_server(
    host: str,
    port: int,
    workspace_root: Path,
    agent: Any,
    ui_dir: Path,
) -> ThreadingHTTPServer:
    """Build, bind, and return a :class:`ThreadingHTTPServer`.

    Does **not** enter ``serve_forever()``; the caller drives the loop
    (which makes the function easier to test). Use this with:

        server = run_server(...)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
    """
    handler_class = build_handler(agent, workspace_root, ui_dir)
    server = ThreadingHTTPServer((host, port), handler_class)
    return server


__all__ = ["AIWebHandler", "build_handler", "run_server"]
