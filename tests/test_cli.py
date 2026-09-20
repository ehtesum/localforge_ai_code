"""Tests for src/ai_env/cli.py."""
from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from ai_env import cli
from ai_env.cli import (
    check_prereqs,
    config_command,
    create_parser,
    health_check,
    model_command,
    start_services,
    stop_services,
    train_command,
)


def _ns(**kwargs) -> argparse.Namespace:
    """Build a minimal argparse.Namespace with sensible defaults."""
    return argparse.Namespace(**kwargs)


class TestCreateParser:
    def test_top_level_commands(self):
        parser = create_parser()
        # Required subparsers: invoking with --help for any should succeed
        for cmd in ("start", "stop", "health", "check", "config", "serve-web"):
            try:
                args = parser.parse_args([cmd])
            except SystemExit:
                pytest.fail(f"Command {cmd!r} was rejected")
            assert args.command == cmd


class TestCheckPrereqs:
    def test_prints_paths_for_each_command(self, capsys):
        with patch("ai_env.cli.find_executable", side_effect=lambda n: f"/usr/bin/{n}"):
            check_prereqs(_ns())
        out = capsys.readouterr().out
        for cmd in ("python", "ollama", "roocode", "code"):
            assert cmd in out
            assert "/usr/bin/" + cmd in out

    def test_marks_missing(self, capsys):
        with patch("ai_env.cli.find_executable", return_value=None):
            check_prereqs(_ns())
        out = capsys.readouterr().out
        assert "not found" in out


class TestHealthCheck:
    def test_prints_status(self, capsys):
        with patch(
            "ai_env.cli.run_health_check",
            return_value={"Ollama": {"ok": "true", "status": "200"},
                          "Roo Code": {"ok": "false", "error": "down"}},
        ):
            health_check(_ns())
        out = capsys.readouterr().out
        assert "[OK] Ollama" in out
        assert "[FAIL] Roo Code" in out


class TestStartServices:
    def test_raises_when_ollama_missing(self):
        with patch("ai_env.cli.check_installed", return_value=False):
            with pytest.raises(RuntimeError):
                start_services(_ns())

    def test_skips_roocode_when_not_installed(self, capsys):
        with patch("ai_env.cli.check_installed", return_value=True), \
             patch("ai_env.cli.serve") as serve, \
             patch("ai_env.cli.find_executable", return_value=None):
            start_services(_ns())
        serve.assert_called_once()
        out = capsys.readouterr().out
        assert "Skip Roo Code" in out

    def test_starts_roocode_when_installed(self):
        with patch("ai_env.cli.check_installed", return_value=True), \
             patch("ai_env.cli.serve"), \
             patch("ai_env.cli.find_executable", return_value="/usr/bin/roocode"), \
             patch("ai_env.cli._start_process") as start_proc:
            start_services(_ns())
        start_proc.assert_called_once()
        args = start_proc.call_args[0][0]
        assert args[0] == "/usr/bin/roocode"
        assert "serve" in args


class TestStopServices:
    def test_invokes_taskkill_on_windows(self, monkeypatch):
        monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
        with patch("ai_env.cli.subprocess.run",
                   return_value=type("R", (), {"returncode": 0})()) as run:
            stop_services(_ns())
        assert run.call_count == 2
        for call in run.call_args_list:
            assert call.args[0][0] == "taskkill"
            assert call.args[0][1] == "/IM"

    def test_uses_pkill_on_linux(self, monkeypatch):
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli.shutil, "which", lambda n: "/usr/bin/pkill" if n == "pkill" else None)
        with patch("ai_env.cli.subprocess.run",
                   return_value=type("R", (), {"returncode": 0})()) as run:
            stop_services(_ns())
        assert run.call_count == 2
        for call in run.call_args_list:
            assert call.args[0][0] == "/usr/bin/pkill"
            assert call.args[0][1] == "-f"
            assert call.args[0][2] in ("ollama", "roocode")

    def test_uses_killall_when_no_pkill(self, monkeypatch):
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        def fake_which(name):
            return "/usr/bin/killall" if name == "killall" else None
        monkeypatch.setattr(cli.shutil, "which", fake_which)
        with patch("ai_env.cli.subprocess.run",
                   return_value=type("R", (), {"returncode": 0})()) as run:
            stop_services(_ns())
        for call in run.call_args_list:
            assert call.args[0][0] == "/usr/bin/killall"

    def test_handles_missing_processes(self, capsys, monkeypatch):
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli.shutil, "which", lambda n: "/usr/bin/pkill" if n == "pkill" else None)
        with patch("ai_env.cli.subprocess.run",
                   return_value=type("R", (), {"returncode": 1})()):
            stop_services(_ns())
        out = capsys.readouterr().out
        assert "not found" in out

    def test_no_pkill_or_killall_reports_clearly(self, capsys, monkeypatch):
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli.shutil, "which", lambda n: None)
        with patch("ai_env.cli.subprocess.run") as run:
            stop_services(_ns())
        # No subprocess.run should be called when neither tool is present.
        assert run.call_count == 0
        out = capsys.readouterr().out
        assert "cannot stop" in out.lower() or "not found" in out.lower()


class TestStopNamed:
    def test_returns_true_when_taskkill_succeeds(self, monkeypatch):
        monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
        with patch("ai_env.cli.subprocess.run",
                   return_value=type("R", (), {"returncode": 0})()) as run:
            assert cli._stop_named("ollama") is True
        assert run.call_args.args[0] == ["taskkill", "/IM", "ollama.exe", "/F"]

    def test_returns_false_when_taskkill_fails(self, monkeypatch):
        monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
        with patch("ai_env.cli.subprocess.run",
                   return_value=type("R", (), {"returncode": 1})()):
            assert cli._stop_named("ollama") is False


class TestModelCommand:
    def test_list(self, capsys):
        with patch("ai_env.cli.list_models", return_value="models:\n- foo"):
            model_command(_ns(subcommand="list"))
        out = capsys.readouterr().out
        assert "foo" in out

    def test_pull(self):
        with patch("ai_env.cli.pull_model", return_value=0) as pull:
            model_command(_ns(subcommand="pull", model="m", device="cpu", precision="int4"))
        pull.assert_called_once_with("m", device="cpu", precision="int4")

    def test_remove(self):
        with patch("ai_env.cli.remove_model", return_value=0) as rm:
            model_command(_ns(subcommand="remove", model="m"))
        rm.assert_called_once_with("m")

    def test_unknown_subcommand(self, capsys):
        with patch("ai_env.cli.list_models"), \
             patch("ai_env.cli.pull_model"), \
             patch("ai_env.cli.remove_model"):
            # Subcommand is None -> the if-chain falls through and returns None
            assert model_command(_ns(subcommand=None)) is None


class TestConfigCommand:
    def test_prints_path(self, capsys):
        from pathlib import Path
        expected_path = Path("/tmp/roo.json")
        with patch("ai_env.cli.ensure_roocode_config", return_value=expected_path):
            config_command(_ns())
        out = capsys.readouterr().out
        assert str(expected_path) in out


class TestTrainCommand:
    def test_prepare_data(self, tmp_path):
        with patch("ai_env.cli.prepare_data") as prep:
            train_command(_ns(
                subcommand="prepare-data",
                output_dir=str(tmp_path),
                max_chars=1000,
                validation_split=0.2,
                seed=7,
            ))
        prep.assert_called_once_with(
            output_dir=tmp_path,
            max_chars=1000,
            val_split=0.2,
            seed=7,
        )

    def test_prepare_web(self):
        from pathlib import Path
        with patch("ai_env.cli.prepare_web_data") as prep:
            train_command(_ns(
                subcommand="prepare-web",
                output_dir="/tmp/out",
                urls=["http://a", "http://b"],
                max_chars=2000,
                validation_split=0.1,
                seed=1,
            ))
        prep.assert_called_once_with(
            output_dir=Path("/tmp/out"),
            urls=["http://a", "http://b"],
            max_chars=2000,
            val_split=0.1,
            seed=1,
        )

    def test_unknown_subcommand(self):
        with pytest.raises(RuntimeError):
            train_command(_ns(subcommand="nope", output_dir="/tmp", max_chars=1,
                              validation_split=0.1, seed=0, urls=[]))
