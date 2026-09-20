from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_env.crawler import CrawlerTrainerWorker


@pytest.fixture
def temp_workspace(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    state_file = workspace_root / "crawler_state.json"
    checkpoint_dir = workspace_root / "checkpoints"
    checkpoint_dir.mkdir()
    return workspace_root, state_file, checkpoint_dir


class TestCrawlerTrainerWorker:
    def test_state_load_default(self, temp_workspace):
        root, state_file, checkpoint_dir = temp_workspace
        worker = CrawlerTrainerWorker(root, state_file, checkpoint_dir)
        state = worker.load_state()
        assert state["status"] == "idle"
        assert len(state["repositories"]) == 5
        assert state["current_index"] == 0
        assert state["completed"] == []
        assert state["logs"] == []

    def test_state_load_existing(self, temp_workspace):
        root, state_file, checkpoint_dir = temp_workspace
        custom_state = {
            "status": "paused",
            "current_index": 2,
            "repositories": ["a/b", "c/d"],
            "completed": ["a/b"],
            "logs": ["log 1"],
        }
        state_file.write_text(json.dumps(custom_state), encoding="utf-8")

        worker = CrawlerTrainerWorker(root, state_file, checkpoint_dir)
        state = worker.load_state()
        assert state == custom_state

    def test_state_save(self, temp_workspace):
        root, state_file, checkpoint_dir = temp_workspace
        worker = CrawlerTrainerWorker(root, state_file, checkpoint_dir)
        test_state = {"data": "test"}
        worker.save_state(test_state)

        saved = json.loads(state_file.read_text(encoding="utf-8"))
        assert saved == test_state

    @patch("urllib.request.urlopen")
    @patch("ai_env.crawler.CrawlerTrainerWorker.train_on_data")
    @patch("ai_env.data.prepare_repo_dataset")
    def test_run_success(
        self, mock_prepare, mock_train, mock_urlopen, temp_workspace
    ):
        root, state_file, checkpoint_dir = temp_workspace
        # Set up a small list of repos to process
        state_file.write_text(
            json.dumps({
                "status": "idle",
                "current_index": 0,
                "repositories": ["test/repo"],
                "completed": [],
                "logs": [],
            }),
            encoding="utf-8",
        )

        # Mock download response
        mock_response = MagicMock()
        mock_response.read.return_value = b"zip-file-bytes"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Mock dataset preparation output
        class MockResult:
            train_path = Path("/tmp/train.jsonl")
            validation_path = Path("/tmp/val.jsonl")
        mock_prepare.return_value = MockResult()

        # Mock extraction zip utility
        with patch("zipfile.ZipFile") as mock_zip:
            mock_zip_instance = mock_zip.return_value.__enter__.return_value
            # We mock the directories listing
            with patch("pathlib.Path.iterdir") as mock_iterdir:
                dummy_dir = MagicMock(spec=Path)
                dummy_dir.is_dir.return_value = True
                dummy_dir.name = "test-repo-commit"
                mock_iterdir.return_value = [dummy_dir]

                worker = CrawlerTrainerWorker(root, state_file, checkpoint_dir)
                worker.run()

        # Verify state is updated to completed
        final_state = json.loads(state_file.read_text(encoding="utf-8"))
        assert final_state["status"] == "completed"
        assert final_state["current_index"] == 1
        assert "test/repo" in final_state["completed"]
        assert len(final_state["logs"]) > 0

    @patch("urllib.request.urlopen")
    def test_run_fallback_simulation(self, mock_urlopen, temp_workspace):
        root, state_file, checkpoint_dir = temp_workspace
        # Force download failure to trigger mock simulation fallback
        mock_urlopen.side_effect = Exception("Network down")

        state_file.write_text(
            json.dumps({
                "status": "idle",
                "current_index": 0,
                "repositories": ["test/repo"],
                "completed": [],
                "logs": [],
            }),
            encoding="utf-8",
        )

        worker = CrawlerTrainerWorker(root, state_file, checkpoint_dir)
        # Mock time.sleep to run quickly
        with patch("time.sleep"):
            worker.run()

        final_state = json.loads(state_file.read_text(encoding="utf-8"))
        assert final_state["status"] == "completed"
        assert final_state["current_index"] == 1
        assert "test/repo" in final_state["completed"]
        assert any("[MOCK] Running training" in log for log in final_state["logs"])
